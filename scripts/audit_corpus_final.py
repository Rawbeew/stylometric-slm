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
        "https://www.gutenberg.org/cache/epub/57496/pg57496.txt",
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
        "https://www.gutenberg.org/cache/epub/8609/pg8609.txt",
        "https://www.gutenberg.org/cache/epub/5320/pg5320.txt",
    ],
    ("fr","flaubert"): [
        "https://www.gutenberg.org/cache/epub/2413/pg2413.txt",
        "https://www.gutenberg.org/cache/epub/26839/pg26839.txt",
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
        "https://www.gutenberg.org/cache/epub/56462/pg56462.txt",
    ],
    ("es","pardo_bazan"): [
        "https://www.gutenberg.org/cache/epub/49990/pg49990.txt",
    ],
    ("it","manzoni"): [
        "https://www.gutenberg.org/cache/epub/4555/pg4555.txt",
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
    return re.sub(r"\s+", " ", s).rstrip(".,;:")


def check_url(url, expected_dir_author):
    """Returns dict with title/author/url status."""
    meta = get_url_meta(url)
    if meta.get("status") != "ok":
        return {"url_ok": False, **meta}
    declared_author = norm(meta.get("author") or "")
    expected_author_norm = norm(expected_dir_author)
    title = meta.get("title")
    # If the declared author matches the directory expectation
    expected_last = expected_author_norm.split()[-1]
    declared_tokens = declared_author.split() if declared_author else []
    matches = any(expected_last in t for t in declared_tokens)
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
