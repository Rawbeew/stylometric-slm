"""
push_to_hf.py
Uploads corpus/ and splits/ to a Hugging Face Hub dataset repo. Reads
HF_TOKEN from environment (set via `huggingface-cli login` or
`export HF_TOKEN=...`). Never echoes the token, never embeds it.

Usage:
  export HF_TOKEN=hf_...   # in your shell, or ~/.bashrc
  huggingface-cli login    # alternative: prompts and saves locally
  python scripts/push_to_hf.py --repo your-username/stylometric-slm-corpus
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo",
        required=True,
        help="Target HF dataset repo, e.g. your-username/stylometric-slm-corpus",
    )
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Run `huggingface-cli login` or `export HF_TOKEN=...`")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo (idempotent — no error if it already exists)
    create_repo(
        args.repo,
        repo_type="dataset",
        private=args.private,
        token=token,
        exist_ok=True,
    )
    print(f"repo ready: https://huggingface.co/datasets/{args.repo}")

    # Upload corpus/ and splits/ as two folders in the dataset repo
    for sub in ("corpus", "splits"):
        src = REPO_ROOT / sub
        if not src.exists():
            print(f"  ! skipping {sub}/ (does not exist locally)")
            continue
        print(f"  uploading {sub}/")
        upload_folder(
            folder_path=str(src),
            repo_id=args.repo,
            repo_type="dataset",
            path_in_repo=sub,
            token=token,
            commit_message=f"upload {sub}/",
        )

    print("done.")
    print(f"view at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
