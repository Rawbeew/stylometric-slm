#!/usr/bin/env python
"""
llm_judge.py — critique paper/journal/*.md against actual reproducibility.

Strategy: extract "claim" signals from a journal entry's Verified result block,
re-run the Cross-check commands, extract the same signal kinds from the fresh run,
and diff. Don't try exact string match — match on the key facts (HTTP codes, file
sizes, numeric counts, exit codes, key=value presence).

Usage:
    python scripts/llm_judge.py                    # deterministic diff
    JUDGE_LLM=gpt-4o-mini python scripts/llm_judge.py  # +LLM critique layer
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DIR = ROOT / "paper" / "journal"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

def extract_section(text: str, header_pattern: str) -> str:
    """Extract the next '```' fenced block after a heading matching pattern.

    Anchors to the start of a line so '## Cross-check commands' doesn't
    accidentally fall through to a subsequent '## Verified results' or
    '### Verified results' heading and pick up the wrong fenced block.
    """
    m = re.search(header_pattern, text, re.MULTILINE)
    if not m:
        return ""
    # Find the next fenced ```block immediately after this heading
    rest = text[m.end():]
    # First skip any sub-headings or paragraphs until we hit ```
    fence_m = re.search(r"```(?:bash|sh)?\n(.*?)```", rest, re.DOTALL)
    if not fence_m:
        return ""
    return fence_m.group(1).strip()


def extract_verified_block(text: str) -> str:
    """Find '### Verified results' heading and pull its first fenced block."""
    # Match a '### Verified results' heading (not '## Verified results')
    m = re.search(r"^### Verified results", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return ""
    rest = text[m.end():]
    fence = re.search(r"```\n(.*?)```", rest, re.DOTALL)
    if not fence:
        return ""
    return fence.group(1).strip()


def extract_signals(text: str) -> dict:
    """Pull out verifiable factual signals from a chunk of terminal-like text.

    Conservative: only count what a fresh execution produces, not paraphrased text.
    """
    sig = {}
    # HTTP status codes (only from `curl -w` or `curl -sI` output)
    # Match "HTTP 200", "HTTP/1.1 200", "HTTP/2 200" — but NOT 4-digit numbers in
    # line counts like "    2382 train.jsonl" (those have leading whitespace + spaces).
    codes = []
    for line in text.splitlines():
        # An HTTP status appears right after "HTTP" or at start of header line.
        m = re.match(r"^\s*HTTP/?[\d.]*\s+(\d{3})\b", line)
        if m:
            codes.append(m.group(1))
    if codes:
        sig["http_codes"] = sorted(set(codes))
    # File sizes from `ls -la` (6+ digit numbers; millisecond-precision rule out)
    lsizes = re.findall(r"\b(\d{5,10})\b\s+\S+\s+\d+\s+\d+\s+\S+\s+\S+\s+\d+\s+\S+", text)
    if lsizes:
        sig["file_size_bytes"] = sorted(set(lsizes))
    # bytes=<N> from curl -w (explicitly tagged)
    sizes = re.findall(r"bytes=(\d{2,})", text)
    if sizes:
        sig["curl_bytes_total"] = sum(int(s) for s in sizes)
    # exit codes from explicit '[exit N]' or 'exit N'
    exits = re.findall(r"exit\s+(?:code\s+)?(\d+)", text)
    if exits:
        sig["exit_codes"] = sorted(set(exits))
    return sig


# --------------------------------------------------------------------------- #
# Run + compare
# --------------------------------------------------------------------------- #

def run_bash(commands: str, timeout: int = 60) -> str:
    """Run a bash block, return stdout+stderr."""
    # strip Windows CR if any
    commands = commands.replace("\r\n", "\n")
    import shutil

    # Find a usable bash on Windows; fall back to git-bash
    candidates = [
        shutil.which("bash"),
        shutil.which("bash.exe"),
        shutil.which("C:/Program Files/Git/bin/bash.exe"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        "/usr/bin/bash",
        "/bin/bash",
    ]
    bash = next((c for c in candidates if c and Path(str(c)).exists()), None)
    if not bash:
        return f"[error: no bash found]\n"
    try:
        r = subprocess.run(
            [bash, "-c", commands],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]\n"
    except Exception as e:
        return f"[error: {type(e).__name__}: {e}]\n"


def diff_signals(verified_text: str, actual_text: str) -> list:
    """Compare signal sets. Returns list of ('mismatch', signal, expected, actual)."""
    findings = []
    v = extract_signals(verified_text)
    a = extract_signals(actual_text)

    all_keys = set(v.keys()) | set(a.keys())
    for k in sorted(all_keys):
        ve = v.get(k, "<absent>")
        ae = a.get(k, "<absent>")
        if ve != ae:
            findings.append((k, ve, ae))

    return findings


# --------------------------------------------------------------------------- #
# Main per-entry audit
# --------------------------------------------------------------------------- #

def audit_entry(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    commands = extract_section(text, r"^## Cross-check commands")
    if not commands:
        return {"file": path.name, "skipped": "no Cross-check block", "verdict": "SKIP"}
    verified = extract_verified_block(text)
    if not verified:
        return {"file": path.name, "skipped": "no Verified results block", "verdict": "SKIP"}

    actual = run_bash(commands, timeout=60)
    findings = diff_signals(verified, actual)
    return {
        "file": path.name,
        "verdict": "CLEAN" if not findings else "MISMATCH",
        "findings": findings,
        "actual_excerpt": actual[:600],
    }


def main():
    if not JOURNAL_DIR.exists():
        print(f"[judge] no journal dir at {JOURNAL_DIR}", file=sys.stderr)
        sys.exit(1)
    entries = sorted([p for p in JOURNAL_DIR.glob("*.md") if p.name != "README.md"])
    if not entries:
        print(f"[judge] no entries in {JOURNAL_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"[judge] auditing {len(entries)} journal entries...\n")
    bad = 0
    report = []
    for entry in entries:
        r = audit_entry(entry)
        report.append(r)
        if r["verdict"] == "SKIP":
            print(f"  -- {entry.name}: {r['skipped']}")
        elif r["verdict"] == "CLEAN":
            print(f"  OK  {entry.name}")
        else:
            bad += 1
            print(f"  ❌ {entry.name}: {len(r['findings'])} signal mismatches")
            for sig, exp, got in r["findings"][:8]:
                print(f"     - {sig}: journal={exp}  re-run={got}")

    print()
    if bad:
        print(f"[judge] {bad}/{len(entries)} entries have mismatches against fresh re-run.")
    else:
        print(f"[judge] all {len(entries)} entries match fresh re-run.")

    # Write machine-readable report
    out = ROOT / "results" / "llm_judge_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[judge] report: {out}")

    # Optional LLM layer
    judge_llm = os.environ.get("JUDGE_LLM")
    if judge_llm:
        print(f"[judge] LLM critique layer configured: {judge_llm}")
        print("[judge]   (prompt: send (entry_md + report) to model, ask for replication referee verdict)")
        print("[judge]   (provider hookup deferred — wire to whichever API key you have in .env)")


if __name__ == "__main__":
    main()
