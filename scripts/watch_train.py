#!/usr/bin/env python3
"""
watch_train.py — poll training progress on the VM, report milestones.

Loop every 15 minutes (sleep), print:
- step counter and ETA
- loss (when first logged at step 50)
- epoch boundary events
- push-to-hub success

Runs as a long-lived foreground process; intended to be invoked inside
a delegator background job so we get notified when milestones hit.
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

SSH = r"C:\Windows\System32\OpenSSH\ssh.exe"
KEY = r"C:\tmp\gce_key"
HOSTS = r"C:\tmp\vm_hosts"
HOST = "alaga@136.118.30.79"
LOG = "/home/alaga/train_v4.log"

PROGRESS = re.compile(r"(\d+)/(\d+)\s+\[([^\]]+)\]")

import shutil

def ssh_run(cmd: str, timeout: int = 30) -> str:
    # Try direct SSH first (faster), fall back to gcloud
    ssh = shutil.which("ssh") or "C:/Windows/System32/OpenSSH/ssh.exe"
    ssh_key = r"C:\tmp\gce_key"
    known_hosts = r"C:\tmp\vm_hosts"

    # Try direct
    try:
        out = subprocess.run(
            [ssh, "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
             "-o", f"UserKnownHostsFile={known_hosts}",
             HOST, cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if out.returncode == 0:
            return (out.stdout or "") + (out.stderr or "")
    except Exception:
        pass
    # Fallback: gcloud
    gcloud = r"C:\tmp\gcloud_wrapper.cmd"
    try:
        r = subprocess.run(
            [gcloud, "compute", "ssh", "stylometric-trainer",
             "--zone=us-west1-c", "--project=orca-503514",
             f"--command={cmd}"],
            capture_output=True, text=True, timeout=timeout,
        )
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"[error: {e}]"


import shutil

def step_and_eta(log_tail: str):
    matches = PROGRESS.findall(log_tail)
    if not matches:
        return None
    cur, total, eta = matches[-1]
    return int(cur), int(total), eta.strip()

def main():
    print(f"[watch] starting at {time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}")
    last_step = -1
    milestones = {"first_loss": False, "first_epoch": False, "hub_push": False}
    poll_minutes = 15
    while True:
        out = ssh_run(f"tail -50 {LOG}")
        cur = step_and_eta(out)
        if cur is None:
            print(f"[watch] no progress info in log; tail:")
            print(out[-300:])
        else:
            step, total, eta = cur
            print(f"[watch] step={step}/{total} eta={eta}  (Δ={step - last_step})")
            last_step = step

            # Look for first loss
            m = re.search(r"'loss':\s*([\d.]+)", out)
            if m and not milestones["first_loss"]:
                milestones["first_loss"] = True
                print(f"[watch] FIRST LOSS logged: {m.group(1)}")
            # First epoch complete
            if step >= 93 and not milestones["first_epoch"]:
                milestones["first_epoch"] = True
                print(f"[watch] FIRST epoch complete (~step 93)")
            # Push-to-hub trace
            if "uploaded" in out.lower() or "pushed" in out.lower():
                if not milestones["hub_push"]:
                    milestones["hub_push"] = True
                    print(f"[watch] HUB PUSH detected")

            if step >= total:
                print("[watch] training complete!")
                # final tail
                print(ssh_run(f"tail -100 {LOG}"))
                break

        time.sleep(poll_minutes * 60)

if __name__ == "__main__":
    main()
