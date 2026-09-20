"""
build_split.py
Walks corpus/{lang}/{author}/passage_*.txt, builds an 80/20 train/eval split
per author (held-out by passage, never mixed across authors), writes
splits/train.jsonl and splits/eval.jsonl.

Each line is JSON: {"author": str, "language": str, "passage_id": str,
"text": str, "n_words": int}

The split is deterministic (hash-based) so re-runs produce the same split.

Usage:
  python scripts/build_split.py
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus"
SPLITS_DIR = REPO_ROOT / "splits"
HELD_OUT_FRAC = 0.20  # 20% held-out per author


def hash_to_bucket(s: str) -> float:
    """Deterministic hash → [0, 1) for split assignment."""
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def word_count(text: str) -> int:
    """Count words, language-aware (Chinese characters count as words)."""
    # Strip whitespace runs
    t = text.strip()
    if not t:
        return 0
    # Heuristic: Chinese text has very few spaces; European has many.
    space_frac = sum(1 for c in t if c == " ") / max(1, len(t))
    if space_frac < 0.05:
        # Chinese — count CJK characters as words
        return sum(1 for c in t if "\u4e00" <= c <= "\u9fff")
    return len(t.split())


def walk_corpus():
    """Yield (lang, author, passage_id, text) tuples."""
    for lang_dir in sorted(CORPUS_DIR.iterdir()):
        if not lang_dir.is_dir():
            continue
        lang = lang_dir.name
        for author_dir in sorted(lang_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            author = author_dir.name
            for passage_file in sorted(author_dir.glob("passage_*.txt")):
                pid = passage_file.stem  # "passage_0042"
                text = passage_file.read_text(encoding="utf-8", errors="ignore")
                yield lang, author, pid, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="stylometric-slm-v1")
    args = ap.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLITS_DIR / "train.jsonl"
    eval_path = SPLITS_DIR / "eval.jsonl"

    n_train = 0
    n_eval = 0
    train_counts = {}
    eval_counts = {}

    with train_path.open("w", encoding="utf-8") as ftrain, eval_path.open(
        "w", encoding="utf-8"
    ) as feval:
        for lang, author, pid, text in walk_corpus():
            nw = word_count(text)
            # Deterministic split: hash the (seed + pid) into [0,1)
            bucket = hash_to_bucket(f"{args.seed}::{lang}::{author}::{pid}")
            row = {
                "author": author,
                "language": lang,
                "passage_id": pid,
                "text": text,
                "n_words": nw,
            }
            if bucket < HELD_OUT_FRAC:
                feval.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_eval += 1
                eval_counts[(lang, author)] = eval_counts.get((lang, author), 0) + 1
            else:
                ftrain.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_train += 1
                train_counts[(lang, author)] = train_counts.get((lang, author), 0) + 1

    print(f"wrote {n_train} train + {n_eval} eval passages")
    print(f"  train: {train_path.relative_to(REPO_ROOT)}")
    print(f"  eval:  {eval_path.relative_to(REPO_ROOT)}")
    print()
    print("per-author breakdown (train / eval):")
    authors = sorted(set(list(train_counts.keys()) + list(eval_counts.keys())))
    for lang, author in authors:
        t = train_counts.get((lang, author), 0)
        e = eval_counts.get((lang, author), 0)
        print(f"  {lang}/{author:<20s} {t:5d} / {e:5d}")


if __name__ == "__main__":
    main()
