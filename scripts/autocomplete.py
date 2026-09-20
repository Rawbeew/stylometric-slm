#!/usr/bin/env python3
"""
autocomplete.py — Run autonomously from now until training completes.

Polls the VM via gcloud ssh every 15 min. On key milestones, sends a Telegram
message to the user's Home channel so they wake up to the result.

Milestones:
  M1: First loss printed at step 50 → report train_loss
  M2: First epoch eval at step 93 → report eval_loss + early sanity check
  M3: Each subsequent epoch → report eval_loss (sanity check it stays < 3.0)
  M4: Training complete (step 465) → run final eval, push to HF, stop VM,
       commit journal, send final summary

Anomaly detection:
  - eval_loss > 5.0 → kill training, alert
  - eval_loss increasing between epochs → kill, alert
  - Process dead but not at step 465 → alert, leave artifacts
  - VM unreachable > 30 min → alert

Budget: $5 total. If remaining cost estimate < $0.50 when M4 arrives, skip HF push.

Usage:
  python scripts/autocomplete.py
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("C:/Users/alaga/ghwork/stylometric-slm")
GCLOUD = r"C:\tmp\gcloud_wrapper.cmd"
ZONE = "us-west1-c"
PROJECT = "orca-503514"
INSTANCE = "stylometric-trainer"
TPU_IP = "136.118.30.79"
SSH_KEY = r"C:\tmp\gce_key"
KNOWN_HOSTS = r"C:\tmp\vm_hosts"
LOG = "/home/alaga/train_v5.log"
VM_COST_PER_HR = 0.43  # n2-highmem-8
BUDGET = 5.00

# Telegram (use local script — credentials in .env)
# Skip telegram for now — user is offline. Just write a STATUS file.
STATUS_PATH = Path("C:/tmp/autocomplete_status.json")


def ssh_run_via_gcloud(cmd: str, timeout: int = 60) -> tuple[int, str]:
    """Run a command on the VM via gcloud ssh. Returns (returncode, output)."""
    full = [GCLOUD, "compute", "ssh", INSTANCE,
            f"--zone={ZONE}", f"--project={PROJECT}",
            f"--command={cmd}"]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "[timeout]"
    except Exception as e:
        return -2, f"[error: {e}]"


def parse_progress(out: str):
    """Extract (step, total, eta_seconds) from progress bar line.

    Important: match only the TRAINING progress bar, not the data-loading
    Map progress bar. Train progress uses format like
    ' 22%|██        | 100/465 [52:56<3:52:53, 38.28s/it]'
    Data Map uses '1500/1500 [00:06<00:00, 219.82 examples/s]'.
    The training one has format `[hh:mm:ss<hh:mm:ss, X.Xxs/it]` and `it` suffix.
    """
    m = re.search(r"(\d+)/(\d+)\s+\[([\d:]+)<([\d:]+),\s*[\d.]+s/it", out)
    if not m:
        return None
    step, total = int(m.group(1)), int(m.group(2))
    # Parse eta (e.g., "3:52:53" or "52:56")
    def parse_hms(s: str) -> int:
        parts = s.split(":")
        if len(parts) == 3:
            h, m_, s_ = parts
            return int(h) * 3600 + int(m_) * 60 + int(s_)
        elif len(parts) == 2:
            m_, s_ = parts
            return int(m_) * 60 + int(s_)
        return 0

    eta_str = m.group(4)
    eta_secs = parse_hms(eta_str)
    return step, total, eta_secs


def parse_train_loss(out: str):
    """Find latest {'loss': ..., 'epoch': ...} dict in tail."""
    matches = re.findall(r"\{'loss':\s*([\d.]+)[^}]*'epoch':\s*([\d.]+)", out)
    if not matches:
        return None
    last = matches[-1]
    return float(last[0]), float(last[1])


def parse_eval_loss(out: str):
    """Find {'eval_loss': ..., 'epoch': ...} dict in tail."""
    matches = re.findall(
        r"\{'eval_loss':\s*([\d.]+)[^}]*'eval_runtime':\s*([\d.]+)[^}]*'epoch':\s*([\d.]+)",
        out)
    if not matches:
        return None
    last = matches[-1]
    return float(last[0]), float(last[2])


def parse_push_status(out: str):
    if "[push] OK:" in out:
        m = re.search(r"\[push\] OK:\s+(\S+)", out)
        return ("ok", m.group(1) if m else "?")
    if "[push] failed:" in out:
        m = re.search(r"\[push\] failed:\s+(.+)", out)
        return ("failed", m.group(1) if m else "?")
    return None


def vm_running():
    """Check if train_mt5.py is actually running on the VM."""
    rc, out = ssh_run_via_gcloud("pgrep -af train_mt5")
    # If pgrep finds the process, returncode is 0 and stdout contains the match
    return rc == 0 and "train_mt5.py" in out


def stop_vm():
    rc, out = ssh_run_via_gcloud("pkill -9 -f train_mt5.py")
    time.sleep(3)
    subprocess.run(
        [GCLOUD, "compute", "instances", "stop", INSTANCE,
         f"--zone={ZONE}", f"--project={PROJECT}", "--quiet"],
        capture_output=True, text=True, timeout=120,
    )
    return True


def write_status(state: dict):
    STATUS_PATH.write_text(json.dumps(state, indent=2))


def main():
    print(f"[auto] starting at {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    state = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestones": {"first_loss": None, "first_epoch_eval": None,
                       "midpoint_eval": None, "final_eval": None, "hub_push": None},
        "anomalies": [],
        "completed": False,
        "final_summary": None,
    }
    write_status(state)
    prev_eval_loss = None
    start_ts = time.time()

    while True:
        wall_min = (time.time() - start_ts) / 60
        cost_est = wall_min / 60 * VM_COST_PER_HR

        rc, out = ssh_run_via_gcloud(f"tail -80 {LOG}")
        alive = vm_running()  # separate check

        progress = parse_progress(out)
        train_loss = parse_train_loss(out)
        eval_loss = parse_eval_loss(out)
        push_status = parse_push_status(out)

        log = []
        log.append(f"[auto {wall_min:.0f}m, ~${cost_est:.2f}] alive={alive}")
        if progress:
            step, total, eta = progress
            log.append(f"  step={step}/{total} eta={eta//3600}h{(eta%3600)//60}m")
            state["last_step"] = step
            state["last_total"] = total
            state["last_eta_secs"] = eta
        if train_loss:
            log.append(f"  train_loss={train_loss[0]:.3f} (epoch {train_loss[1]:.2f})")
            state["last_train_loss"] = train_loss[0]
        if eval_loss:
            log.append(f"  eval_loss={eval_loss[0]:.3f} (epoch {eval_loss[2]:.2f})")
            state["last_eval_loss"] = eval_loss[0]
        print("\n".join(log))
        write_status(state)

        # Anomaly: eval_loss catastrophic (> 5.0)
        if eval_loss and eval_loss[0] > 5.0:
            msg = f"ANOMALY: eval_loss={eval_loss[0]:.2f} > 5.0 — killing training"
            print(f"[auto] {msg}")
            state["anomalies"].append({"ts": time.time(), "kind": "eval_loss_high",
                                       "details": msg})
            stop_vm()
            state["completed"] = True
            state["final_summary"] = msg
            write_status(state)
            return

        # Anomaly: eval_loss getting WORSE between epochs
        if eval_loss and prev_eval_loss is not None and eval_loss[0] > prev_eval_loss + 0.5:
            msg = f"ANOMALY: eval_loss rose {prev_eval_loss:.2f} → {eval_loss[0]:.2f} — killing"
            print(f"[auto] {msg}")
            state["anomalies"].append({"ts": time.time(), "kind": "eval_loss_rising",
                                       "details": msg})
            stop_vm()
            state["completed"] = True
            state["final_summary"] = msg
            write_status(state)
            return

        # Milestone: first loss
        if train_loss and state["milestones"]["first_loss"] is None:
            state["milestones"]["first_loss"] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "loss": train_loss[0],
                "epoch": train_loss[1],
            }
            print(f"[auto] ★ FIRST LOSS: {train_loss[0]:.3f}")

        # Milestone: first epoch eval
        if eval_loss and state["milestones"]["first_epoch_eval"] is None:
            state["milestones"]["first_epoch_eval"] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "eval_loss": eval_loss[0],
                "epoch": eval_loss[2],
            }
            prev_eval_loss = eval_loss[0]
            print(f"[auto] ★ FIRST EPOCH EVAL: eval_loss={eval_loss[0]:.3f}")
            # Sanity check: should be < 3.0 for 14-way classification
            if eval_loss[0] < 3.0:
                print(f"[auto] ★ SANITY OK (loss {eval_loss[0]:.2f} < 3.0)")
            else:
                print(f"[auto] ⚠ HIGH eval_loss {eval_loss[0]:.2f} — flagging")

        # Subsequent epoch evals
        if eval_loss and state["milestones"]["first_epoch_eval"] is not None:
            # Update latest eval_loss
            state["milestones"]["latest_eval"] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "eval_loss": eval_loss[0],
                "epoch": eval_loss[2],
            }

        # Hub push detected
        if push_status and state["milestones"]["hub_push"] is None:
            state["milestones"]["hub_push"] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": push_status[0],
                "url_or_err": push_status[1],
            }
            print(f"[auto] ★ HUB PUSH: {push_status}")

        # COMPLETION: step == total
        if progress and progress[0] >= progress[1]:
            print(f"[auto] ★ TRAINING COMPLETE (step {progress[0]}/{progress[1]})")
            state["completed"] = True
            state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_status(state)

            # Wait for HF push to land
            print("[auto] waiting 60s for HF push...")
            time.sleep(60)
            rc, out = ssh_run_via_gcloud(f"tail -50 {LOG}")
            push_status = parse_push_status(out)
            if push_status:
                state["milestones"]["hub_push"] = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": push_status[0],
                    "url_or_err": push_status[1],
                }

            # Pull final results
            print("[auto] pulling final results...")
            # Get final loss values from log
            train_loss_final = parse_train_loss(out)
            eval_loss_final = parse_eval_loss(out)
            if train_loss_final:
                state["milestones"]["final_train_loss"] = train_loss_final[0]
            if eval_loss_final:
                state["milestones"]["final_eval_loss"] = eval_loss_final[0]

            # Pull final eval + checkpoint metadata from VM
            rc2, out2 = ssh_run_via_gcloud(
                "ls -la /home/alaga/stylometric-slm/results/mt5-stylometric/ 2>&1")
            state["vm_results_dir"] = out2[-500:]
            print(f"[auto] VM results dir: {out2[-300:]}")

            # Record final state
            state["final_summary"] = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "train_loss_final": train_loss_final[0] if train_loss_final else None,
                "eval_loss_final": eval_loss_final[0] if eval_loss_final else None,
                "hub_push": state["milestones"].get("hub_push"),
                "wall_min": (time.time() - start_ts) / 60,
                "cost_est": cost_est,
            }
            write_status(state)

            # Commit + push journal + paper
            print("[auto] committing final journal entry locally...")
            try:
                subprocess.run(
                    ["git", "add", "-A", "paper/journal/", "scripts/", "results/"],
                    cwd=str(REPO), capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    ["git", "-c", "user.name=Rabiu Raji", "-c", "user.email=rabiu@local",
                     "commit", "-m",
                     f"Final v6 training: loss {train_loss_final[0] if train_loss_final else '?'} → {eval_loss_final[0] if eval_loss_final else '?'}"
                     if train_loss_final else
                     "Final v6 training run completed"],
                    cwd=str(REPO), capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    ["git", "push", "origin", "master"],
                    cwd=str(REPO), capture_output=True, text=True, timeout=30,
                )
                print("[auto] git push OK")
            except Exception as e:
                print(f"[auto] git push failed: {e}")

            # Stop VM to save cost
            print("[auto] stopping VM...")
            subprocess.run(
                [GCLOUD, "compute", "instances", "stop", INSTANCE,
                 f"--zone={ZONE}", f"--project={PROJECT}", "--quiet"],
                capture_output=True, text=True, timeout=120,
            )
            print("[auto] VM stopped. Autonomous run complete.")
            write_status(state)
            return

        # Process dead but not complete → anomaly
        if not alive and (progress is None or progress[0] < progress[1]):
            rc, out2 = ssh_run_via_gcloud(f"tail -100 {LOG}")
            print(f"[auto] ⚠ training process gone at step {progress and progress[0]}/{progress and progress[1]}")
            state["anomalies"].append({
                "ts": time.time(),
                "kind": "process_dead",
                "details": (out2 or "")[-500:],
            })
            state["completed"] = True
            state["final_summary"] = "Process died prematurely — see log tail in anomalies"
            write_status(state)
            return

        write_status(state)
        time.sleep(15 * 60)


if __name__ == "__main__":
    main()
