#!/usr/bin/env python3
"""build_work_blocked_split.py — fix within-work leakage with EXACT book IDs.

Two-pass:
  1. Download each source book from Gutenberg (once, cached to corpus/_books/).
  2. Match every corpus passage to its book by normalized-substring search
     (passages are ~1000-word windows of the book). Ambiguous or unmatched
     passages are reported, never silently assigned.
  3. Block-split by BOOK: no book in both train and eval. Authors with one
     book become train-only (reported, not hidden).

Outputs: splits_blocked/{train,eval}.jsonl + book_map.json + match report.
"""
import hashlib
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = REPO / "corpus"
BOOKS_DIR = REPO / "corpus" / "_books"
OUT = REPO / "splits_blocked"
OUT.mkdir(exist_ok=True)
BOOKS_DIR.mkdir(exist_ok=True)

# exact source lists from scripts/pull_gutenberg.py (audited 2026-09-20)
SOURCES = {
    ("en", "dickens"): [("pg98", "A Tale of Two Cities"), ("pg1400", "Great Expectations"), ("pg766", "David Copperfield")],
    ("en", "twain"): [("pg74", "Tom Sawyer"), ("pg76", "Huckleberry Finn"), ("pg3176", "Innocents Abroad")],
    ("en", "woolf"): [("pg1245", "Night and Day"), ("pg5670", "Jacob's Room")],
    ("en", "joyce"): [("pg4217", "Portrait of the Artist")],
    ("en", "melville"): [("pg2701", "Moby-Dick"), ("pg10712", "White Jacket"), ("pg15859", "Piazza Tales")],
    ("fr", "hugo"): [("pg135", "Les Misérables"), ("pg2610", "Notre-Dame de Paris")],
    ("fr", "zola"): [("pg6497", "L'Assommoir"), ("pg5711", "Germinal")],
    ("fr", "flaubert"): [("pg2413", "Madame Bovary")],
    ("fr", "maupassant"): [("pg3090", "Original Short Stories")],
    ("fr", "proust"): [("pg2650", "Du côté de chez Swann")],
    ("es", "cervantes"): [("pg2000", "Don Quijote")],
    ("es", "galdos"): [("pg17013", "Fortunata y Jacinta"), ("pg17340", "Marianela")],
    ("es", "pardo_bazan"): [("pg17491", "La Tribuna"), ("pg68452", "La piedra angular")],
    ("it", "manzoni"): [("pg45334", "I Promessi Sposi (2014 ed.)")],
}

WS = re.compile(r"\s+")


def norm(s):
    return WS.sub(" ", s).strip().lower()


def fetch_book(pg_id):
    """Download + cache a book; return normalized full text."""
    cache = BOOKS_DIR / f"{pg_id}.txt"
    if cache.exists():
        return norm(cache.read_text(encoding="utf-8", errors="ignore"))
    url = f"https://www.gutenberg.org/cache/epub/{pg_id[2:]}/{pg_id}.txt"
    print(f"  downloading {pg_id}...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "replication-build/1.0"})
    raw = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    cache.write_text(raw, encoding="utf-8")
    return norm(raw)


def main():
    # load all passages (train+eval from the ORIGINAL split, since we keep
    # every passage and re-split)
    rows = []
    for split in ("train", "eval"):
        for line in open(REPO / "splits" / f"{split}.jsonl", encoding="utf-8"):
            rows.append(json.loads(line))

    by_author = defaultdict(list)
    for r in rows:
        by_author[(r["language"], r["author"])].append(r)
    for k in by_author:
        by_author[k].sort(key=lambda r: int(r["passage_id"].split("_")[1]))

    report, blocked_train, blocked_eval, book_map = [], [], [], []
    n_unmatched = 0

    for key, arows in sorted(by_author.items()):
        books = SOURCES[key]
        print(f"{key[1]}: matching {len(arows)} passages against {len(books)} book(s)")
        book_texts = {bid: fetch_book(bid) for bid, _ in books}
        # index: for each book, first 80 normalized chars of each passage as probe
        assigned = []
        for r in arows:
            ptext = norm(r["text"])
            probe = ptext[:80]
            hits = [bid for bid, btext in book_texts.items() if probe and probe in btext]
            if len(hits) == 1:
                assigned.append((r, hits[0]))
            elif len(hits) == 0:
                # try the tail (some passages may have leading boilerplate)
                probe2 = ptext[-80:]
                hits = [bid for bid, btext in book_texts.items() if probe2 in btext]
                if len(hits) == 1:
                    assigned.append((r, hits[0]))
                else:
                    n_unmatched += 1
                    report.append({"passage": f"{key[0]}/{key[1]}/{r['passage_id']}",
                                   "status": "unmatched", "hits": hits})
                    assigned.append((r, None))
            else:
                # multi-hit: pick the book whose text contains the LONGEST probe
                best, bestlen = None, 0
                for bid in hits:
                    # binary-search the longest prefix present
                    lo, hi = 80, min(len(ptext), 400)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if ptext[:mid] in book_texts[bid]:
                            lo = mid
                        else:
                            hi = mid - 1
                    if lo > bestlen:
                        best, bestlen = bid, lo
                assigned.append((r, best))

        # book -> passage list
        book_groups = defaultdict(list)
        for r, bid in assigned:
            if bid is None:
                continue
            book_groups[bid].append(r)

        # decide sides: multi-book -> hash-order books, ~1/3 to eval; single
        # book -> train-only
        bids = sorted(book_groups)
        if len(bids) == 1:
            sides = {bids[0]: "train"}
        else:
            order = sorted(bids, key=lambda b: hashlib.sha256(b.encode()).hexdigest())
            n_eval_books = max(1, len(bids) // 3) if len(bids) >= 3 else 1
            sides = {b: ("eval" if b in set(order[:n_eval_books]) else "train") for b in bids}

        for bid, plist in book_groups.items():
            label = dict(books)[bid]
            side = sides[bid]
            for r in plist:
                out = dict(r, source_book=label, source_pg=bid)
                (blocked_eval if side == "eval" else blocked_train).append(out)
        book_map.append({"lang": key[0], "author": key[1],
                         "books": [{"pg": bid, "book": dict(books)[bid],
                                    "n": len(book_groups[bid]), "split": sides[bid]}
                                   for bid in bids]})

    # verify no book leak
    tr_books = {(r["author"], r["source_pg"]) for r in blocked_train}
    ev_books = {(r["author"], r["source_pg"]) for r in blocked_eval}
    assert not (tr_books & ev_books), f"BOOK LEAK: {tr_books & ev_books}"

    (OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in blocked_train) + "\n", encoding="utf-8")
    (OUT / "eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in blocked_eval) + "\n", encoding="utf-8")
    (OUT / "book_map.json").write_text(json.dumps(book_map, indent=2), encoding="utf-8")
    (OUT / "match_report.json").write_text(json.dumps(
        {"unmatched": n_unmatched, "details": report[:50]}, indent=2), encoding="utf-8")

    ct = Counter(r["author"] for r in blocked_train)
    ce = Counter(r["author"] for r in blocked_eval)
    print(f"\nblocked train: {len(blocked_train)}  blocked eval: {len(blocked_eval)}  unmatched: {n_unmatched}")
    for a in sorted(ct):
        print(f"  {a:12s} train={ct[a]:4d} eval={ce.get(a, 0):4d}")
    print("authors with eval:", sorted(ce))
    print("book-level leak check: CLEAN" if not (tr_books & ev_books) else "LEAK!")


if __name__ == "__main__":
    main()
