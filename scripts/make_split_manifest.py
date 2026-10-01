#!/usr/bin/env python3
"""make_split_manifest.py — turn the split into a verifiable replication package.

The critique: "a rebuild script is not a replication package." This writes:

  1. replication/split_manifest.json — every corpus passage file with its
     SHA-256, word count, language, author, and split assignment, plus the
     hash-bucket formula so the split is auditable by hand.
  2. replication/eval_set.sha256 — the exact eval passage hashes (705).
  3. replication/train_set.sha256 — the exact train passage hashes (2810).

With these, a reviewer can: (a) re-run pull_gutenberg.py, (b) diff their
corpus hashes against the manifest, (c) verify the hash-bucket split assigns
each passage identically, (d) run the baselines and get the same numbers.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "corpus"
OUT = REPO / "replication"
OUT.mkdir(exist_ok=True)
SPLITS = REPO / "splits"


def sha256_file(p):
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def hash_to_bucket(s):
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


# split assignment per passage id (as built by build_split.py)
assign = {}
for split in ("train", "eval"):
    for line in open(SPLITS / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        assign[(r["language"], r["author"], r["passage_id"])] = split

manifest = {"generated": "2026-10-01", "split_rule": {
    "method": "sha256(passage_id)[:8] as hex / 0xFFFFFFFF -> [0,1); bucket < 0.20 = eval",
    "held_out_frac": 0.20,
    "note": "deterministic; no RNG, so the seed question is vacuous: the split is a pure hash function of passage IDs",
}, "passages": []}

files = sorted(CORPUS.glob("*/*/passage_*.txt"))
print(f"hashing {len(files)} passage files...")
for i, p in enumerate(files):
    lang, author = p.parts[-3], p.parts[-2]
    pid = p.stem
    split = assign.get((lang, author, pid), "MISSING")
    manifest["passages"].append({
        "lang": lang, "author": author, "passage_id": pid,
        "split": split,
        "sha256": sha256_file(p),
        "n_words": len(p.read_text(encoding="utf-8", errors="ignore").split()),
    })
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(files)}")

(OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

eval_lines, train_lines = [], []
for e in manifest["passages"]:
    line = f"{e['sha256']}  {e['lang']}/{e['author']}/{e['passage_id']}"
    (eval_lines if e["split"] == "eval" else train_lines).append(line)
(OUT / "eval_set.sha256").write_text("\n".join(eval_lines) + "\n", encoding="utf-8")
(OUT / "train_set.sha256").write_text("\n".join(train_lines) + "\n", encoding="utf-8")

n_eval = sum(1 for e in manifest["passages"] if e["split"] == "eval")
n_train = sum(1 for e in manifest["passages"] if e["split"] == "train")
n_missing = sum(1 for e in manifest["passages"] if e["split"] == "MISSING")
print(f"\nmanifest: {len(manifest['passages'])} passages")
print(f"  eval={n_eval} train={n_train} missing={n_missing}")
assert n_eval == 705 and n_train == 2810 and n_missing == 0, "SPLIT MISMATCH vs published numbers"
print("matches published split exactly (705 eval / 2810 train)")
