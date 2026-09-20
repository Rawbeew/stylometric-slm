#!/usr/bin/env python3
"""
audit_corpus_v2.py — STAGE 1: Title-only detection.

For each file, ONLY inspect the first 1000 chars of the body (after stripping
any Project Gutenberg boilerplate). Try to find the canonical "TITLE by AUTHOR"
attribution line. Then compare AUTHOR to the directory name.

Distinguishes:
  - OK:           attribution matches directory
  - WRONG_DIR:    attribution present, names a DIFFERENT known author
  - NO_HEADER:    no clear attribution (probably mid-book excerpt)
  - LOW_QUALITY:  text is mostly Gutenberg metadata (skip)

The previous v1 used "any mention" which caused false positives on names
mentioned in narrative prose (e.g., Dickens quoting Goethe). This v2 only looks
at the canonical title+byline.
"""

import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

# Canonical author name with regex that matches in title/byline context only
# (each pattern explicitly includes "by " or end-of-title marker)
TITLE_AUTHOR_PATTERNS = [
    # canonical [First Last] "by [First Last]" capture
    (None, re.compile(
        r"(?P<title>[A-Z][A-Za-z0-9 ,;:\-_'\.!?]{3,80}?)\s+by\s+(?P<author>[A-Z][a-zA-Záéíóúñüçëÿâêîôûäöüßøåæœ\,\s]+?)(?:\s*\[|\s*Translated|\s*$|\n)",
        re.M,
    )),
]

# Standard PG header pattern: "Title: X\nAuthor: Y"
PG_HEADER = re.compile(
    r"Title:\s*(?P<title>[^\n]+)\n(?:.*\n)*?Author:\s*(?P<author>[^\n]+)",
    re.M,
)

# Cover-page style: TITLE on its own line, followed within next few lines by AUTHOR NAME
# Detect via lines that look like names (2-3 capitalized words, ASCII)
LINE_AUTHOR_NAME = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s*$")
LINE_AUTHOR_NAME_FULL = re.compile(
    r"^[A-Z][a-záéíóúñüçëÿâêîôûäöüßøåæœ]+(?:\s+[A-Z][a-záéíóúñüçëÿâêîôûäöüßøåæœ]+){1,3}\s*$"
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
    """Extract author/title attribution from first ~1500 chars after PG stripping."""
    body = strip_gutenberg(text)
    head = body[:1500]

    # Method 1: PG metadata header
    m = PG_HEADER.search(head)
    if m:
        return {"title": m.group("title").strip(), "author": m.group("author").strip(), "method": "pg_header"}

    # Method 2: "TITLE by AUTHOR" pattern (must appear early, within first 30 lines)
    lines = head.split("\n")[:30]
    scan = "\n".join(lines)
    m = re.search(
        r"^(?P<title>[A-Z][A-Za-z0-9 ,;:\-_'\.!?\d]{3,90}?)\s+by\s+(?P<author>[A-Z][a-zA-Záéíóúñüçëÿâêîôûäöüßøåæœ\,\s]+?)(?:\s*$|\s*\n|\s*Translated|\s*\[)",
        scan,
        re.M,
    )
    if m:
        title = m.group("title").strip()
        author = m.group("author").strip()
        # Filter out nonsense: title too short or too long, no capital letters
        if 5 <= len(title) <= 100 and len(author) >= 5 and re.search(r"[A-Z]", author):
            return {"title": title, "author": author, "method": "byline"}

    # Method 3: cover page (look for first non-metadata line, then find name)
    nonmeta_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("[")
                     and not l.strip().startswith("Produced")
                     and not l.strip().startswith("Updated")
                     and not l.strip().startswith("E-text")
                     and "Project Gutenberg" not in l]
    candidates = []
    for i, line in enumerate(nonmeta_lines[:8]):
        if LINE_AUTHOR_NAME_FULL.match(line):
            candidates.append((i, line))
    if candidates:
        # Use the first name-like line as the author, the line before it as title
        i, author = candidates[0]
        title = nonmeta_lines[i - 1] if i > 0 else "(no title)"
        return {"title": title, "author": author, "method": "cover_page"}

    return {"title": None, "author": None, "method": "none"}


# Quick map of directory name → expected author canonical name
# (lowercase forms we expect)
DIR_TO_CANON = {
    "dickens": "dickens",
    "joyce": "joyce",
    "melville": "melville",
    "twain": "twain",
    "woolf": "woolf",
    "cervantes": "cervantes",
    "galdos": "galdos",
    "pardo_bazan": "pardo bazan",
    "flaubert": "flaubert",
    "hugo": "hugo",
    "maupassant": "maupassant",
    "proust": "proust",
    "zola": "zola",
    "manzoni": "manzoni",
}


def norm(s: str) -> str:
    s = s.lower().strip()
    # strip accents for comparison
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    # collapse whitespace
    s = re.sub(r"\s+", " ", s)
    # strip trailing "letter", punctuation
    s = s.rstrip(".,;:")
    return s


def check_attribution(exp_dirname: str, attribution: dict) -> str:
    expected = DIR_TO_CANON.get(exp_dirname, exp_dirname)
    if attribution["author"] is None:
        return "NO_HEADER"
    declared = norm(attribution["author"])

    # Is the declared name reasonably close to expected?
    expected_norm = norm(expected)
    # simple substring check (most authors single-token)
    if declared == expected_norm:
        return "OK"
    if expected_norm in declared or declared in expected_norm:
        return "OK"
    # special: 'perez galdos' vs 'galdos'
    if expected_norm in ("galdos", "perez galdos", "benito perez galdos"):
        if "galdos" in declared or "gald" in declared:
            return "OK"
    if "pardo" in expected_norm and "pardo" in declared:
        return "OK"
    return "WRONG_DIR"


def audit_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    attr = find_attribution(text)
    verdict = check_attribution(path.parent.name, attr)

    return {
        "path": str(path.relative_to(ROOT)),
        "lang": path.parent.parent.name,
        "expected": path.parent.name,
        "verdict": verdict,
        "title_detected": attr["title"],
        "author_detected": attr["author"],
        "method": attr["method"],
    }


def main():
    if not CORPUS.exists():
        sys.exit(1)

    findings = []
    for lang_dir in sorted(CORPUS.iterdir()):
        if not lang_dir.is_dir():
            continue
        for author_dir in sorted(lang_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            for f in sorted(author_dir.glob("*.txt")):
                findings.append(audit_one(f))

    # Aggregate
    counts = Counter(f["verdict"] for f in findings)
    print(f"[audit-v2] {len(findings)} files inspected")
    print("  Verdict counts:")
    for v, n in counts.most_common():
        print(f"    {v:11} {n}")

    # Per-author
    print("\n  Per-author breakdown:")
    by_dir = defaultdict(lambda: Counter())
    for f in findings:
        key = f"{f['lang']}/{f['expected']}"
        by_dir[key][f["verdict"]] += 1
    for key in sorted(by_dir):
        st = by_dir[key]
        flag = " ⚠" if st["WRONG_DIR"] > 0 else ""
        print(f"    {key:18} {dict(st)}{flag}")

    # Show real misattributions
    wrong = [f for f in findings if f["verdict"] == "WRONG_DIR"]
    if wrong:
        print(f"\n  {len(wrong)} files have WRONG_DIR:")
        # Group by (expected_dir, declared_author) for compactness
        groups = defaultdict(list)
        for f in wrong:
            key = (f["lang"], f["expected"], f["author_detected"])
            groups[key].append(Path(f["path"]).name)
        print(f"\n  Grouped (lang/expected_dir) -> declared_author:")
        for (lang, exp, decl), files in sorted(groups.items()):
            print(f"\n    ❌ {lang}/{exp} → declared '{decl}' ({len(files)} files)")
            for fn in files[:5]:
                print(f"       - {fn}")
            if len(files) > 5:
                print(f"       ... and {len(files) - 5} more")

    out = ROOT / "results" / "corpus_audit_v2.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    print(f"\n[audit-v2] full report: {out}")


if __name__ == "__main__":
    main()
