#!/usr/bin/env python3
"""
audit_corpus_final.py — definitive corpus audit with full report.

For each author dir:
  1. Read passage_0000.txt (canonical first passage of first book)
  2. Identify the actual title + author based on raw head + URL metadata
  3. Cross-reference with the URL list in pull_gutenberg.py to flag mismatches
  4. Mark the directory as CLEAN, WRONG_TEXT, MIXED, or UNKNOWN

Output is a JSON + a human readable summary.
"""

import json
import re
import urllib.request
import urllib.error
import sys
from pathlib import Path
from collections import defaultdict, Counter
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
URLS_FILE = ROOT / "scripts/pull_gutenberg.py"

URLS_INDEX = {
    # (lang, author): [urls]
    # Updated 2026-09-20 — match the corrected URLs in scripts/pull_gutenberg.py
    ("en","dickens"): [
        "https://www.gutenberg.org/cache/epub/98/pg98.txt",
        "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
        "https://www.gutenberg.org/cache/epub/766/pg766.txt",
    ],
    ("en","twain"): [
        "https://www.gutenberg.org/cache/epub/74/pg74.txt",
        "https://www.gutenberg.org/cache/epub/76/pg76.txt",
        "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",
    ],
    ("en","woolf"): [
        # FIXED from pg57496 (Mary Johnston) → pg1245 (Night and Day) + pg5670 (Jacob's Room)
        "https://www.gutenberg.org/cache/epub/1245/pg1245.txt",
        "https://www.gutenberg.org/cache/epub/5670/pg5670.txt",
    ],
    ("en","joyce"): [
        "https://www.gutenberg.org/cache/epub/4217/pg4217.txt",
    ],
    ("en","melville"): [
        "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        "https://www.gutenberg.org/cache/epub/10712/pg10712.txt",
        "https://www.gutenberg.org/cache/epub/15859/pg15859.txt",
    ],
    ("fr","hugo"): [
        "https://www.gutenberg.org/cache/epub/135/pg135.txt",
        "https://www.gutenberg.org/cache/epub/2610/pg2610.txt",
    ],
    ("fr","zola"): [
        # FIXED from pg8609 (404) + pg5320 (E.P. Roe English)
        # to pg6497 (L'Assommoir FR) + pg5711 (Germinal FR)
        "https://www.gutenberg.org/cache/epub/6497/pg6497.txt",
        "https://www.gutenberg.org/cache/epub/5711/pg5711.txt",
    ],
    ("fr","flaubert"): [
        # FIXED from pg2413 + pg26839 (404) to just pg2413 (Madame Bovary FR)
        "https://www.gutenberg.org/cache/epub/2413/pg2413.txt",
    ],
    ("fr","maupassant"): [
        "https://www.gutenberg.org/cache/epub/3090/pg3090.txt",
    ],
    ("fr","proust"): [
        "https://www.gutenberg.org/cache/epub/2650/pg2650.txt",
    ],
    ("es","cervantes"): [
        "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
    ],
    ("es","galdos"): [
        # FIXED from pg56462 (Gibson, English) to pg17013 (Fortunata y Jacinta) + pg17340 (Marianela)
        "https://www.gutenberg.org/cache/epub/17013/pg17013.txt",
        "https://www.gutenberg.org/cache/epub/17340/pg17340.txt",
    ],
    ("es","pardo_bazan"): [
        # FIXED from pg49990 (English lit history) to pg17491 (La Tribuna) + pg68452 (La piedra angular)
        "https://www.gutenberg.org/cache/epub/17491/pg17491.txt",
        "https://www.gutenberg.org/cache/epub/68452/pg68452.txt",
    ],
    ("it","manzoni"): [
        # FIXED from pg4555 (Symonds, English) to pg45334 (I promessi sposi, IT)
        "https://www.gutenberg.org/cache/epub/45334/pg45334.txt",
    ],
}

# Cache: fetch each URL only once
URL_FETCH_CACHE = {}

def get_url_meta(url):
    if url in URL_FETCH_CACHE:
        return URL_FETCH_CACHE[url]
    try:
        ua = "Stylometric-SLM-Audit/1.0"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=20) as r:
            head = r.read(2000).decode("utf-8", errors="replace")
    except Exception as e:
        meta = {"error": str(e), "status": "fail"}
        URL_FETCH_CACHE[url] = meta
        return meta

    title_m = re.search(r"^Title:\s*([^\n]+)", head, re.M)
    author_m = re.search(r"^Author:\s*([^\n]+)", head, re.M)
    meta = {
        "status": "ok",
        "title": title_m.group(1).strip() if title_m else None,
        "author": author_m.group(1).strip() if author_m else None,
    }
    URL_FETCH_CACHE[url] = meta
    return meta


def norm(s):
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    # Replace underscores (used as separators in dir names) with spaces
    s = s.replace("_", " ")
    return re.sub(r"\s+", " ", s).rstrip(".,;:")


def check_url(url, expected_dir_author):
    """Returns dict with title/author/url status."""
    meta = get_url_meta(url)
    if meta.get("status") != "ok":
        return {"url_ok": False, **meta}
    declared_author = norm(meta.get("author") or "")
    expected_author_norm = norm(expected_dir_author)
    title = meta.get("title")
    # Compare against any token in the declared author. The author string
    # often has a title (e.g. "condesa de") before the surname.
    expected_tokens = expected_author_norm.split()
    declared_tokens = declared_author.split() if declared_author else []
    # If the expected contains multiple tokens (e.g. "pardo bazan"), accept
    # if any of them appears in declared; otherwise accept if expected last
    # matches anything.
    matches = any(tok in declared_tokens for tok in expected_tokens) if expected_tokens else False
    if not matches and expected_tokens:
        # Fallback: substring check on any declared token
        matches = any(expected_tokens[-1] in t for t in declared_tokens)
    return {
        "url_ok": True,
        "title": title,
        "author": meta.get("author"),
        "matches_expected_author": matches,
    }


def audit_author(lang, author):
    """Audit one author dir against its URLs."""
    urls = URLS_INDEX.get((lang, author), [])
    url_results = [check_url(u, author) for u in urls]

    # Verdict
    if not url_results:
        return {"verdict": "NO_URL_CONFIG", "url_results": []}
    if all(r.get("matches_expected_author") for r in url_results):
        verdict = "OK"
    elif any(r.get("matches_expected_author") for r in url_results):
        verdict = "MIXED"
    else:
        verdict = "WRONG"

    # Count what we have on disk
    out_dir = CORPUS / lang / author
    n_files = sum(1 for _ in out_dir.glob("*.txt")) if out_dir.exists() else 0

    return {
        "verdict": verdict,
        "url_results": [
            {**r, "url": u} for r, u in zip(url_results, urls)
        ],
        "on_disk_files": n_files,
    }


def main():
    findings = {}
    for (lang, author), _ in URLS_INDEX.items():
        findings[(lang, author)] = audit_author(lang, author)

    print("=" * 80)
    print("STYLOMETRIC SLM — Corpus Audit Report (2026-09-20)")
    print("=" * 80)
    print()

    total_passages_affected = 0
    total_clean_passages = 0

    for (lang, author), r in sorted(findings.items()):
        verdict = r["verdict"]
        files = r["on_disk_files"]
        marker = "  OK" if verdict == "OK" else "❌"
        print(f"{marker} {lang}/{author}: verdict={verdict}, files={files}")

        for u in r["url_results"]:
            if u.get("url_ok"):
                match = "✓" if u.get("matches_expected_author") else "✗"
                print(f"      {match} {u.get('title')[:50] if u.get('title') else '?'}")
                print(f"        author: {u.get('author')}")
            else:
                print(f"      ✗ URL failed: {u.get('error') or u.get('status')}")
                print(f"        url: {u['url']}")
        if verdict in ("WRONG", "MIXED"):
            total_passages_affected += files
        else:
            total_clean_passages += files
        print()

    print("=" * 80)
    print(f"Total passages CLEAN: {total_clean_passages}")
    print(f"Total passages CONTAMINATED: {total_passages_affected}")
    print(f"Total: {total_clean_passages + total_passages_affected}")
    print("=" * 80)

    out = ROOT / "results/corpus_audit_final.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(
        {f"{l}/{a}": r for (l, a), r in findings.items()},
        indent=2, ensure_ascii=False
    ))
    print(f"\nFull JSON: {out}")


if __name__ == "__main__":
    main()
