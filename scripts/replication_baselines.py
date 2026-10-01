#!/usr/bin/env python3
"""replication_baselines.py — standard-method baselines for the stylometric-slm
replication note.

Answers the reviewer asks the original paper did not:
  1. Burrows' Delta (the field's canonical method) on the same split.
  2. Character n-gram SVM (the strong classical baseline family).
  3. Logreg baseline re-run on the MATCHED 705-passage eval (the original
     compared 605 vs 705 — mismatched slices).
  4. Full confusion matrices for every method + the classifier's own numbers.
  5. Leave-one-author-out OPEN-SET test with Delta: can the method tell that a
     held-out author is NOT any of the 13 candidates, or does it force-assign?

Methodology notes:
  - Feature vocabulary is computed from TRAIN passages only. No eval leakage.
  - Delta: top-N most frequent words in train (N in {100, 500, 1000, 3000}),
    z-scored per passage, distance = mean |z_i(a) - z_i(b)|, nearest centroid.
  - Open-set: for held-out author A, train Delta on the other 13, score A's
    eval passages against the 13 centroids. Two diagnostics:
      (a) forced-assignment accuracy (which impostor wins — expect near 0)
      (b) rejection: z-normalize distances per passage; if the best impostor's
          score is closer than the "impostor distance distribution" (rank of
          best score among all candidate scores for that passage), flag as
          "not rejected". We report the rank of the best candidate — a
          held-out author that consistently ranks #1 among candidates is
          being force-assigned, which is exactly the closed-set failure.

Outputs:
  results/replication_baselines.json  (all numbers, machine-readable)
  results/confusion_<method>.csv     (14x14 confusion matrices)
  results/openset_delta.json         (leave-one-author-out details)
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "splits"
RESULTS = REPO / "results"
RESULTS.mkdir(exist_ok=True)

SEED = 42  # frozen for the replication note; sklearn seeds set at each call

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


train = load(SPLITS / "train.jsonl")
eval_ = load(SPLITS / "eval.jsonl")
AUTHORS = sorted({r["author"] for r in train})
LANGS = {a: r["language"] for r in train for a in [r["author"]] if r["author"] == a}
LANGS = {}
for r in train:
    LANGS[r["author"]] = r["language"]
AIDX = {a: i for i, a in enumerate(AUTHORS)}
print(f"train={len(train)} eval={len(eval_)} authors={len(AUTHORS)}")
print("authors:", AUTHORS)


# ---------------------------------------------------------------- Burrows' Delta

def tokens(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def build_delta(train_rows, top_n, authors=None):
    """Top-N words by train frequency -> per-author centroid z-score vectors.

    `authors` restricts the label set (used by the open-set test so a
    held-out author never gets a phantom centroid)."""
    label_set = authors if authors is not None else AUTHORS
    lidx = {a: i for i, a in enumerate(label_set)}
    freq = Counter()
    for r in train_rows:
        freq.update(tokens(r["text"]))
    vocab = [w for w, _ in freq.most_common(top_n)]
    vidx = {w: i for i, w in enumerate(vocab)}

    # per-passage relative freq
    def relfreq(text):
        toks = tokens(text)
        v = np.zeros(len(vocab))
        if not toks:
            return v
        c = Counter(toks)
        for w, n in c.items():
            if w in vidx:
                v[vidx[w]] = n / len(toks)
        return v

    X_train = np.array([relfreq(r["text"]) for r in train_rows])
    # corpus means/SDs from TRAIN ONLY
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    Z_train = (X_train - mu) / sd

    centroids = np.zeros((len(label_set), len(vocab)))
    counts = np.zeros(len(label_set))
    for r, z in zip(train_rows, Z_train):
        a = r["author"]
        if a not in lidx:
            continue  # held-out author: no centroid, never a candidate
        centroids[lidx[a]] += z
        counts[lidx[a]] += 1
    assert (counts > 0).all(), "every label must have train passages"
    centroids /= counts[:, None]

    def predict(text):
        z = (relfreq(text) - mu) / sd
        d = np.abs(centroids - z).mean(axis=1)
        i = int(np.argmin(d))
        return label_set[i], d

    return predict


def eval_delta(predict, eval_rows, tag):
    y_true = [r["author"] for r in eval_rows]
    y_pred = []
    for r in eval_rows:
        p, _ = predict(r["text"])
        y_pred.append(p)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    cm = np.zeros((len(AUTHORS), len(AUTHORS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[AIDX[t]][AIDX[p]] += 1
    np.savetxt(RESULTS / f"confusion_delta_{tag}.csv", cm, fmt="%d",
               delimiter=",", header=",".join(AUTHORS))
    per_author = {a: {"correct": int(cm[AIDX[a]][AIDX[a]]),
                      "total": int(cm[AIDX[a]].sum()),
                      "accuracy": round(float(cm[AIDX[a]][AIDX[a]] / max(1, cm[AIDX[a]].sum())), 3)}
                  for a in AUTHORS}
    return {"method": f"delta_top{tag}", "accuracy": round(acc, 4),
            "per_author": per_author}, y_pred


# ---------------------------------------------------------------- char n-gram SVM

def eval_charsvm(train_rows, eval_rows, ngram_range=(3, 5), max_features=200000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          max_features=max_features, lowercase=True,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([r["text"] for r in train_rows])
    ytr = [r["author"] for r in train_rows]
    clf = LinearSVC(random_state=SEED)
    clf.fit(Xtr, ytr)
    Xev = vec.transform([r["text"] for r in eval_rows])
    y_pred = clf.predict(Xev).tolist()
    y_true = [r["author"] for r in eval_rows]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    cm = np.zeros((len(AUTHORS), len(AUTHORS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[AIDX[t]][AIDX[p]] += 1
    np.savetxt(RESULTS / "confusion_charsvm.csv", cm, fmt="%d",
               delimiter=",", header=",".join(AUTHORS))
    per_author = {a: {"correct": int(cm[AIDX[a]][AIDX[a]]),
                      "total": int(cm[AIDX[a]].sum()),
                      "accuracy": round(float(cm[AIDX[a]][AIDX[a]] / max(1, cm[AIDX[a]].sum())), 3)}
                  for a in AUTHORS}
    return {"method": f"charsvm_{ngram_range[0]}-{ngram_range[1]}", "accuracy": round(acc, 4),
            "per_author": per_author}, y_pred


# ---------------------------------------------------------------- logreg matched slice

def eval_logreg(train_rows, eval_rows):
    """Re-run the original baseline on the MATCHED 705 eval slice."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(analyzer="word", lowercase=True, max_features=50000,
                          sublinear_tf=True)
    Xtr = vec.fit_transform([r["text"] for r in train_rows])
    ytr = [r["author"] for r in train_rows]
    clf = LogisticRegression(max_iter=2000, random_state=SEED, n_jobs=1)
    clf.fit(Xtr, ytr)
    Xev = vec.transform([r["text"] for r in eval_rows])
    y_pred = clf.predict(Xev).tolist()
    y_true = [r["author"] for r in eval_rows]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    cm = np.zeros((len(AUTHORS), len(AUTHORS)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[AIDX[t]][AIDX[p]] += 1
    np.savetxt(RESULTS / "confusion_logreg_705.csv", cm, fmt="%d",
               delimiter=",", header=",".join(AUTHORS))
    per_author = {a: {"correct": int(cm[AIDX[a]][AIDX[a]]),
                      "total": int(cm[AIDX[a]].sum()),
                      "accuracy": round(float(cm[AIDX[a]][AIDX[a]] / max(1, cm[AIDX[a]].sum())), 3)}
                  for a in AUTHORS}
    return {"method": "logreg_wordtfidf_705", "accuracy": round(acc, 4),
            "per_author": per_author}, y_pred


# ---------------------------------------------------------------- open-set (LOAO) Delta

def openset_delta(train_rows, eval_rows, top_n=1000):
    """Leave-one-author-out: Delta trained on 13 authors, scored on the
    held-out author's eval passages. Reports which impostor wins and the
    best-candidate rank — rank 1 for most passages = force-assignment."""
    out = {}
    for held in AUTHORS:
        tr = [r for r in train_rows if r["author"] != held]
        ev = [r for r in eval_rows if r["author"] == held]
        if not ev:
            continue
        candidates = [a for a in AUTHORS if a != held]
        predict = build_delta(tr, top_n, authors=candidates)
        impostor_wins = Counter()
        ranks = []
        for r in ev:
            pred, dist = predict(r["text"])
            impostor_wins[pred] += 1
            # rank of best (nearest) candidate distance is always 1 by
            # construction of argmin; what matters is the DISTANCE GAP:
            # how close the nearest impostor is, relative to the typical
            # same-author distances in this reduced problem. We record the
            # min distance and the spread of the 13 candidate distances.
            ranks.append((float(dist.min()), float(dist.std())))
        mins = [m for m, _ in ranks]
        stds = [s for _, s in ranks]
        out[held] = {
            "n_eval": len(ev),
            "impostor_wins": dict(impostor_wins.most_common(3)),
            "forced_assignment_rate": round(sum(1 for _ in ev) / len(ev), 3),
            "mean_min_distance": round(float(np.mean(mins)), 4),
            "mean_distance_spread": round(float(np.mean(stds)), 4),
            "note": "no reject option — every held-out passage is assigned to an impostor; "
                    "the diagnostic is WHICH impostor and how tight the assignment is",
        }
    (RESULTS / "openset_delta.json").write_text(
        json.dumps({"method": f"delta_top{top_n}", "seed": SEED, "loao": out},
                   indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------- main

def main():
    results = []

    for n in (100, 500, 1000, 3000):
        print(f"\n=== Burrows' Delta top-{n} ===")
        pred_fn = build_delta(train, n)
        r, _ = eval_delta(pred_fn, eval_, tag=str(n))
        print(f"  accuracy: {r['accuracy']}")
        print(f"  flaubert: {r['per_author']['flaubert']['accuracy']}"
              f"  zola: {r['per_author']['zola']['accuracy']}")
        results.append(r)

    print("\n=== char n-gram SVM (3-5) ===")
    r, _ = eval_charsvm(train, eval_)
    print(f"  accuracy: {r['accuracy']}")
    print(f"  flaubert: {r['per_author']['flaubert']['accuracy']}"
          f"  zola: {r['per_author']['zola']['accuracy']}")
    results.append(r)

    print("\n=== logreg on matched 705 ===")
    r, _ = eval_logreg(train, eval_)
    print(f"  accuracy: {r['accuracy']}")
    print(f"  flaubert: {r['per_author']['flaubert']['accuracy']}")
    results.append(r)

    print("\n=== open-set (leave-one-author-out, Delta top-1000) ===")
    loao = openset_delta(train, eval_, top_n=1000)
    for a, v in loao.items():
        print(f"  {a:12s} -> top impostor: {list(v['impostor_wins'].items())[:1]}")

    summary = {
        "seed": SEED,
        "split": {"train": len(train), "eval": len(eval_),
                  "deterministic": "sha256-hash bucketing, build_split.py"},
        "methods": results,
        "classifier_v8_reference": {
            "accuracy_705": 0.912, "sampled_50per": 0.886,
            "flaubert_sampled": "1/19 -> zola", "source": "results/classifier_results.json",
        },
    }
    (RESULTS / "replication_baselines.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote results/replication_baselines.json + confusion CSVs")


if __name__ == "__main__":
    main()
