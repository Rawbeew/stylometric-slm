# verify.py — byte-level verification of project scripts and notebooks.
# Computes SHA256 of every file in scripts/ and notebooks/ + the README,
# confirms Python syntax and notebook JSON parse. Exits non-zero on any failure.
#
# Usage:
#   python verify.py                # verify the repo at ./scripts, ./notebooks
#   python verify.py --root /path   # verify a different checkout
import argparse
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent

TARGETS = [
    "README.md",
    "scripts/pull_gutenberg.py",
    "scripts/pull_ctext.py",
    "scripts/build_split.py",
    "scripts/clean_passages.py",
    "scripts/train_classifier.py",
    "scripts/train_mt5.py",
    "scripts/baseline_local_cpu.py",
    "scripts/audit_corpus.py",
    "scripts/audit_corpus_v2.py",
    "scripts/audit_corpus_v3.py",
    "scripts/audit_corpus_final.py",
    "scripts/llm_judge.py",
    "scripts/push_to_hf.py",
    "scripts/upload_to_zenodo.py",
    "notebooks/finetune_mt5.ipynb",
    "notebooks/one_click_train.ipynb",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="project root to verify (default: directory containing verify.py)")
    args = ap.parse_args()
    root = args.root.resolve()

    print(f"{'file':<42} {'bytes':>9}  {'sha256':<12}  status")
    print("-" * 86)

    failures: list[str] = []
    total_bytes = 0
    n_files = 0

    for rel in TARGETS:
        p = root / rel
        label = rel
        if not p.exists():
            print(f"{label:<42} {'MISSING':>9}  {'-':<12}  FAIL")
            failures.append(rel)
            continue
        size = p.stat().st_size
        total_bytes += size
        n_files += 1
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
        status = "OK"
        try:
            if rel.endswith(".py"):
                compile(p.read_text(encoding="utf-8"), str(p), "exec")
            elif rel.endswith(".ipynb"):
                json.loads(p.read_text(encoding="utf-8"))
        except (SyntaxError, json.JSONDecodeError) as e:
            status = f"FAIL: {type(e).__name__}: {e}"
            failures.append(rel)
        print(f"{label:<42} {size:>9}  {digest:<12}  {status}")

    print("-" * 86)
    print(f"verified: {n_files - len(failures)}/{n_files} files, {total_bytes} bytes")
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())