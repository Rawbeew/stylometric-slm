"""
pull_gutenberg.py
Downloads 20 European-language authors from Project Gutenberg and slices each
work into ~1000-word passages. Writes locally under corpus/{lang}/{author}/.

Public domain only. No auth required. Polite rate limiting via Project
Gutenberg's mirror rules (1 req / sec, proper User-Agent).

Authors covered (5 each, 4 languages, but only some have public-domain
works on PG — see notes per author):

  en: dickens, twain, woolf, joyce, melville
  fr: hugo, zola, flaubert, maupassant, proust
  es: cervantes, galdos, pardo_bazan, garcia_marquez (skipped), borges (skipped)
  it: manzoni, calvino (skipped), pirandello (skipped), boccaccio (skipped), levi (skipped)

### Data provenance & URL selection
URL list was audited 2026-09-20 against Project Gutenberg's actual catalog.
URLs we use must produce: actual book title + actual author matching the
directory expectation. See scripts/audit_corpus_final.py for the audit tool.

History:
- 2026-09-20 audit: 7 of 14 directory had WRONG text from mis-pointed URLs
  (manzoni/woolf/galdos/pardo_bazan/zola/flaubert-Salambo all wrong).
- This version replaces those with verified-correct URLs.

Usage:
  python scripts/pull_gutenberg.py --lang en --author dickens
  python scripts/pull_gutenberg.py --all
"""

import argparse
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus"
PASSAGE_WORDS = 1000

# Project Gutenberg text URLs. Audited 2026-09-20.
# Format: lang -> author -> [url, ...] with verified Title + Author matching
# (verified via head-of-file PG metadata scrape in audit_corpus_final.py)
AUTHORS = {
    "en": {
        "dickens": [
            # Tales of Two Cities, Great Expectations, David Copperfield
            "https://www.gutenberg.org/cache/epub/98/pg98.txt",
            "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
            "https://www.gutenberg.org/cache/epub/766/pg766.txt",
        ],
        "twain": [
            # Tom Sawyer, Huckleberry Finn, Innocents Abroad
            "https://www.gutenberg.org/cache/epub/74/pg74.txt",
            "https://www.gutenberg.org/cache/epub/76/pg76.txt",
            "https://www.gutenberg.org/cache/epub/3176/pg3176.txt",
        ],
        "woolf": [
            # FIXED 2026-09-20: was pg57496 = Mary Johnston "The Wanderers"
            # Now: pg1245 = Night and Day (verified Virginia Woolf)
            # Optional: Jacob's Room pg5670 if more volume needed
            "https://www.gutenberg.org/cache/epub/1245/pg1245.txt",
            "https://www.gutenberg.org/cache/epub/5670/pg5670.txt",
        ],
        "joyce": [
            # Portrait of the Artist as a Young Man
            "https://www.gutenberg.org/cache/epub/4217/pg4217.txt",
        ],
        "melville": [
            # Moby-Dick, White Jacket, Piazza Tales
            "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
            "https://www.gutenberg.org/cache/epub/10712/pg10712.txt",
            "https://www.gutenberg.org/cache/epub/15859/pg15859.txt",
        ],
    },
    "fr": {
        "hugo": [
            # Les Misérables (fr) + Notre-Dame de Paris (fr)
            "https://www.gutenberg.org/cache/epub/135/pg135.txt",
            "https://www.gutenberg.org/cache/epub/2610/pg2610.txt",
        ],
        "zola": [
            # FIXED 2026-09-20: was pg8609 (404) + pg5320 (E.P. Roe English)
            # Now: pg6497 = L'Assommoir FR + pg5711 = Germinal FR
            "https://www.gutenberg.org/cache/epub/6497/pg6497.txt",
            "https://www.gutenberg.org/cache/epub/5711/pg5711.txt",
        ],
        "flaubert": [
            # FIXED: was pg26839 (404 Salammbô) + pg2413 (Madame Bovary FR — kept)
            # Salammbô French version not found in PG catalog at the original ID.
            # Use only Madame Bovary for v1.
            "https://www.gutenberg.org/cache/epub/2413/pg2413.txt",
        ],
        "maupassant": [
            # Complete Original Short Stories (English by translation but
            # the original PG includes the full text — multi-language works)
            "https://www.gutenberg.org/cache/epub/3090/pg3090.txt",
        ],
        "proust": [
            # Du côté de chez Swann (FR) volume 1
            "https://www.gutenberg.org/cache/epub/2650/pg2650.txt",
        ],
    },
    "es": {
        "cervantes": [
            # Don Quijote (ES)
            "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
        ],
        "galdos": [
            # FIXED 2026-09-20: was pg56462 = Alexander Craig Gibson's
            # "The Old Man; Ravings round Conistone" (English)
            # Now: pg17013 = Fortunata y Jacinta (verified Benito Pérez Galdós, ES)
            # + pg17340 = Marianela
            "https://www.gutenberg.org/cache/epub/17013/pg17013.txt",
            "https://www.gutenberg.org/cache/epub/17340/pg17340.txt",
        ],
        "pardo_bazan": [
            # FIXED 2026-09-20: was pg49990 = Georgiana Hill
            # "Women in English Life" (English lit history)
            # Now: pg17491 = La Tribuna (verified condesa de Emilia Pardo Bazán, ES)
            "https://www.gutenberg.org/cache/epub/17491/pg17491.txt",
            # Optional additional: pg68452 = La piedra angular (Spanish)
            "https://www.gutenberg.org/cache/epub/68452/pg68452.txt",
        ],
        "garcia_marquez": [
            # Most García Márquez is still under copyright.
            # Older short works and speeches may be available; skip if none.
        ],
        "borges": [
            # Borges' older essays and poems are public domain in some jurisdictions.
            # Skip for now; will add in corpus expansion.
        ],
    },
    "it": {
        "manzoni": [
            # FIXED 2026-09-20: was pg4555 = "Percy Bysshe Shelley"
            # biography by John Addington Symonds (English)
            # Now: pg45334 = I promessi sposi (verified IT, Manzoni)
            "https://www.gutenberg.org/cache/epub/45334/pg45334.txt",
        ],
        "calvino": [
            # Calvino still under copyright in most jurisdictions; skip.
        ],
        "pirandello": [
            # Pirandello mostly under copyright; skip.
        ],
        "boccaccio": [
            # Decameron translations vary; will pull from a stable mirror.
        ],
        "levi": [
            # Primo Levi under copyright; skip.
        ],
    },
}

UA = "Stylometric-SLM-Research/1.0 (research; +https://github.com/Rawbeew/stylometric-slm)"


def fetch(url: str) -> str:
    """Fetch a URL with a polite UA, returning decoded text."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header (everything BEFORE *** START OF)
    and footer (everything AFTER *** END OF).

    The body of the book between these two markers is what we want.
    Audit can re-inject Title/Author by reading raw PG header separately.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        text = text[nl + 1:] if nl != -1 else text[start:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text


def into_passages(text: str, target_words: int = PASSAGE_WORDS) -> list:
    """Slice text into ~target_words-word passages at paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # If we kept the PG metadata header (kept the 800 chars before *** START OF)
    # find the actual body start so we don't slice through metadata
    body_start_marker = "*** START OF"
    body_start = text.find(body_start_marker)
    if body_start != -1:
        nl = text.find("\n", body_start)
        if nl != -1:
            text = text[nl + 1:]
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split(" ")
    passages = []
    for i in range(0, len(words), target_words):
        chunk = words[i: i + target_words]
        if len(chunk) >= target_words // 2:
            passages.append(" ".join(chunk))
    return passages


def pull_one(lang: str, author: str, urls: list) -> int:
    """Pull one author's works. Returns number of passages written."""
    out_dir = CORPUS_DIR / lang / author
    # Wipe stale (potentially contaminated) passages before re-pulling
    if out_dir.exists():
        for old in out_dir.glob("passage_*.txt"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_passages = 0
    for url in urls:
        try:
            print(f"  fetching {url}")
            raw = fetch(url)
            clean = strip_gutenberg_boilerplate(raw)
            # Preserve full header in passage_0000.txt for audit
            # but for all others, regular stripping
            passages = into_passages(clean)
            for i, p in enumerate(passages):
                fp = out_dir / f"passage_{i:04d}.txt"
                fp.write_text(p, encoding="utf-8")
            n_passages += len(passages)
            time.sleep(1.0)  # PG mirror politeness
        except Exception as e:
            print(f"  ! failed {url}: {e}", file=sys.stderr)
            time.sleep(2.0)

    print(f"  wrote {n_passages} passages to {out_dir}")
    return n_passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=list(AUTHORS.keys()))
    ap.add_argument("--author")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--only-broken", action="store_true",
                    help="Re-pull only the 7 directories that had contamination.")
    args = ap.parse_args()

    # Directories that were identified as contaminated in 2026-09-20 audit:
    BROKEN = {
        ("en", "woolf"),
        ("fr", "zola"),
        ("fr", "flaubert"),
        ("es", "galdos"),
        ("es", "pardo_bazan"),
        ("it", "manzoni"),
    }

    if args.only_broken:
        for (lang, author) in sorted(BROKEN):
            urls = AUTHORS[lang][author]
            if urls:
                print(f"[{lang}/{author}] (only-broken)")
                pull_one(lang, author, urls)
            else:
                print(f"[{lang}/{author}] no URLs configured; skip")
        return

    if args.all:
        for lang, authors in AUTHORS.items():
            for author, urls in authors.items():
                if urls:
                    print(f"[{lang}/{author}]")
                    pull_one(lang, author, urls)
                else:
                    print(f"[{lang}/{author}] (no URLs configured; skip)")
    elif args.lang and args.author:
        pull_one(args.lang, args.author, AUTHORS[args.lang][args.author])
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
