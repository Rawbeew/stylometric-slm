#!/usr/bin/env python3
"""
audit_corpus_v3.py — pattern-based attribution audit.

Looks at first 800 chars of text (after stripping PG boilerplate) and
extracts the canonical "TITLE by AUTHOR" pattern. Patterns supported:
  - "TITLE By AUTHOR"
  - "TITLE by AUTHOR"
  - "TITLE por AUTHOR" (Spanish)
  - "TITLE par AUTHOR" (French)
  - "TITLE de AUTHOR" (some Romance cases)
  - "TITLE BY JOHN ADDINGTON SYMONDS" (all-caps)
  - Plain "AUTHOR" name alone (cover-page style)

Reports per-file mismatches and group statistics.
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

# Match `TITLE by AUTHOR` flexibly
# Title: capitalized words and common punctuation, len 4..100
# Author: capitalized name(s), may include accents, hyphens, periods, 2-5 tokens
BY_PATTERN = re.compile(
    r"(?P<title>[A-Z][\wÀ-ſ ,;:\-_'\.!\?&]{3,90}?)\s+"
    r"(?:by|par|por|di|de)\s+"
    r"(?P<author>[A-Z][\wÀ-ſ'\.\-]+(?:\s+[A-Z][\wÀ-ſ'\.\-]+){1,4})"
)

# All-caps variant: "PERCY BYSSHE SHELLEY BY JOHN ADDINGTON SYMONDS"
BY_PATTERN_UPPER = re.compile(
    r"(?P<title>[A-Z][A-Z0-9 ,;:\-_'\.!\?&]{3,90}?)\s+"
    r"(?:BY|PAR|POR|DI|DE)\s+"
    r"(?P<author>[A-Z][A-Z'\.\-]+(?:\s+[A-Z][A-Z'\.\-]+){1,4})"
)


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        text = text[nl + 1:] if nl != -1 else text[start:]
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text


def find_attribution(text: str) -> dict:
    body = strip_gutenberg(text)
    head = body[:1500]
    scan = head.replace("\n", " | ")

    for label, rx in [("byline", BY_PATTERN), ("uppercase", BY_PATTERN_UPPER)]:
        m = rx.search(scan)
        if m:
            t = m.group("title").strip().rstrip(".,;:")
            a = m.group("author").strip().rstrip(".,;:")
            if len(t) >= 4 and len(a) >= 4:
                return {"title": t, "author": a, "method": label}

    return {"title": None, "author": None, "method": "none"}


def norm(s: str) -> str:
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".,;:")
    return s


DIR_TO_CANON = {
    "dickens": "charles dickens",
    "joyce": "james joyce",
    "melville": "herman melville",
    "twain": "mark twain",
    "woolf": "virginia woolf",
    "cervantes": "miguel de cervantes",
    "galdos": "benito perez galdos",
    "pardo_bazan": "emilia pardo bazan",
    "flaubert": "gustave flaubert",
    "hugo": "victor hugo",
    "maupassant": "guy de maupassant",
    "proust": "marcel proust",
    "zola": "emile zola",
    "manzoni": "alessandro manzoni",
}


def check(exp_dir: str, attr: dict) -> str:
    if attr["author"] is None:
        return "NO_HEADER"
    declared = norm(attr["author"])
    expected = DIR_TO_CANON.get(exp_dir, exp_dir)
    exp_norm = norm(expected)
    # Core check: declared contains expected's last token (last name)
    expected_last = exp_norm.split()[-1]
    declared_tokens = declared.split()
    if any(expected_last in t for t in declared_tokens):
        return "OK"
    # Partial — author has different last name entirely
    return "WRONG_DIR"


def audit_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    attr = find_attribution(text)
    verdict = check(path.parent.name, attr)
    return {
        "path": str(path.relative_to(ROOT)),
        "lang": path.parent.parent.name,
        "expected": path.parent.name,
        "verdict": verdict,
        "title": attr["title"],
        "detected_author": attr["author"],
        "method": attr["method"],
    }


def main():
    if not CORPUS.exists():
        sys.exit(1)

    findings = [audit_one(f)
                for lang_dir in sorted(CORPUS.iterdir())
                if lang_dir.is_dir()
                for author_dir in sorted(lang_dir.iterdir())
                if author_dir.is_dir()
                for f in sorted(author_dir.glob("*.txt"))]

    counts = Counter(f["verdict"] for f in findings)
    print(f"[v3] {len(findings)} files inspected")
    for v, n in counts.most_common():
        print(f"  {v:12} {n}")

    print("\nPer-author breakdown:")
    by_dir = defaultdict(Counter)
    for f in findings:
        by_dir[f"{f['lang']}/{f['expected']}"][f["verdict"]] += 1
    for k in sorted(by_dir):
        st = by_dir[k]
        flag = " ⚠" if st.get("WRONG_DIR", 0) > 0 else ""
        print(f"  {k:18} {dict(st)}{flag}")

    wrong = [f for f in findings if f["verdict"] == "WRONG_DIR"]
    if wrong:
        print(f"\n{len(wrong)} files have WRONG_DIR:")
        grp = defaultdict(list)
        for f in wrong:
            key = (f["lang"], f["expected"], f["detected_author"] or "?")
            grp[key].append(Path(f["path"]).name)
        for (lang, exp, decl), files in sorted(grp.items()):
            print(f"\n  ❌ {lang}/{exp} → declared '{decl}' ({len(files)} files)")
            for fn in files[:8]:
                print(f"     - {fn}")
            if len(files) > 8:
                print(f"     ... +{len(files) - 8} more")

    out = ROOT / "results" / "corpus_audit_v3.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    print(f"\n[v3] full: {out}")


if __name__ == "__main__":
    main()
