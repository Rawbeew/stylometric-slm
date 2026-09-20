#!/usr/bin/env python3
"""
audit_corpus.py — proper-text-only forensic corpus audit.

For each file in corpus/<lang>/<author>/, look at the actual text and decide:
  - "OK" if title and language match the directory
  - "MISATTRIBUTED" if the title clearly says another known author
  - "NO_TITLE" if we can't determine

Strategy:
  1. Look in the first 2000 characters (after stripping any Gutenberg boilerplate)
  2. Extract the first non-trivial line that looks like a title
  3. Match against known author name patterns (full directory cross-check)

Used to find data contamination before training a stylometric model — if
twain text sits in melville/, the model learns wrong associations.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"

# (canonical name, regex) — order: longest first to avoid partial matches
KNOWN_AUTHORS = [
    ("balzac",       re.compile(r"\bhonoré\s+de\s+balzac|\bbalzac\b", re.I)),
    ("baudelaire",   re.compile(r"\bbaudelaire\b", re.I)),
    ("bronte",       re.compile(r"\bbront[ëe]\b", re.I)),
    ("byron",        re.compile(r"\blord\s+byron\b|\bbyron\b", re.I)),
    ("cervantes",    re.compile(r"\bmiguel\s+de\s+cervantes|\bcervantes\s+saavedra|\bcervantes\b", re.I)),
    ("chekhov",      re.compile(r"\bchekhov\b|\bchehov\b", re.I)),
    ("darwin",       re.compile(r"\bcharles\s+darwin|\bdarwin\b", re.I)),
    ("dickens",      re.compile(r"\bcharles\s+dickens|\bdickens\b", re.I)),
    ("dostoyevsky",  re.compile(r"\bdostoyevsky\b|\bdostoevsky\b", re.I)),
    ("eliot",        re.compile(r"\bgeorge\s+eliot\b|\bmary\s+ann\s+evans\b", re.I)),
    ("flaubert",     re.compile(r"\bgustave\s+flaubert|\bflaubert\b", re.I)),
    ("franklin",     re.compile(r"\bbenjamin\s+franklin\b", re.I)),
    ("freud",        re.compile(r"\bsigmund\s+freud\b", re.I)),
    ("galdos",       re.compile(r"\bpérez\s+galdós|\bbenito\s+p[ée]rez\s+gald[óo]s|\bgaldos\b", re.I)),
    ("goethe",       re.compile(r"\bgoethe\b", re.I)),
    ("gogol",        re.compile(r"\bgogol\b", re.I)),
    ("hardy",        re.compile(r"\bthomas\s+hardy\b|\bhardy\b", re.I)),
    ("hawthorne",    re.compile(r"\bnathaniel\s hawthorne\b|\bhawthorne\b", re.I)),
    ("homero",       re.compile(r"\bhomer\b|\bhomero\b", re.I)),
    ("hugo",         re.compile(r"\bvictor\s+hugo\b|\bhugo\b", re.I)),
    ("james",        re.compile(r"\bhenry\s+james\b", re.I)),
    ("joyce",        re.compile(r"\bjames\s+joyce\b|\bjoyce\b", re.I)),
    ("kafka",        re.compile(r"\bkafka\b", re.I)),
    ("kipling",      re.compile(r"\brudyard\s+kipling\b|\bkipling\b", re.I)),
    ("leopardi",     re.compile(r"\bleopardi\b", re.I)),
    ("liszt",        re.compile(r"\bliszt\b", re.I)),
    ("mann",         re.compile(r"\bthomas\s+mann\b", re.I)),
    ("manzoni",      re.compile(r"\balessandro\s+manzoni\b|\bmanzoni\b", re.I)),
    ("maupassant",   re.compile(r"\bguy\s+de\s+maupassant\b|\bmaupassant\b", re.I)),
    ("melville",     re.compile(r"\bherman\s+melville\b|\bmelville\b", re.I)),
    ("milton",       re.compile(r"\bmilton\b", re.I)),
    ("moravia",      re.compile(r"\bmoravia\b", re.I)),
    ("neruda",       re.compile(r"\bneruda\b", re.I)),
    ("pardo_bazan",  re.compile(r"\bemilia\s+pardo\s+baz[áa]n\b|\bpardo\s+baz[áa]n\b|\bpardo_bazan\b", re.I)),
    ("poe",          re.compile(r"\bedgar\s+allan\s+poe\b|\bpoe\b", re.I)),
    ("pound",        re.compile(r"\bpezza\s+(?:ezra\s+)?pound\b|\bpound\b", re.I)),
    ("proust",       re.compile(r"\bmarcel\s+proust\b|\bproust\b", re.I)),
    ("roe",          re.compile(r"\be\.?\s*p\.?\s*roe\b|\b[eE]\.\s*P\.\s*Roe\b", re.I)),
    ("shelley",      re.compile(r"\bpercy\s+bysshe\s+shelley\b|\bshelley\b", re.I)),
    ("stendhal",     re.compile(r"\bstendhal\b", re.I)),
    ("stevenson",    re.compile(r"\brosert\s+louis\s+stevenson\b|\bstevenson\b", re.I)),
    ("symonds",      re.compile(r"\bsymonds\b", re.I)),
    ("tolstoy",      re.compile(r"\btolstoy\b|\btolstoi\b", re.I)),
    ("twain",        re.compile(r"\bmark\s+twain\b|\btwain\b", re.I)),
    ("verne",        re.compile(r"\bjules\s+verne\b|\bverne\b", re.I)),
    ("virgilio",     re.compile(r"\bvirgilio\b|\bvirgil\b", re.I)),
    ("vonnegut",     re.compile(r"\bvonnegut\b", re.I)),
    ("wilde",        re.compile(r"\boscar\s+wilde\b|\bwilde\b", re.I)),
    ("woolf",        re.compile(r"\bvirginia\s+woolf\b|\bwoolf\b", re.I)),
    ("zola",         re.compile(r"\b[eé]mile\s+zola\b|\bzola\b", re.I)),
]


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer if present."""
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start != -1:
        nl = text.find("\n", start)
        text = text[nl + 1:] if nl != -1 else text[start:]
    if end != -1:
        text = text[:end]
    return text


def get_first_meaningful_lines(text: str, n_lines: int = 10) -> list:
    body = strip_gutenberg(text).strip()
    lines = [l for l in body.split("\n") if l.strip() and not l.startswith("[")]
    return lines[:n_lines]


def detect_authors(text: str) -> set:
    """Find all known authors whose name appears in the first 2000 chars."""
    body = strip_gutenberg(text)[:2000]
    out = set()
    for canon, rx in KNOWN_AUTHORS:
        if rx.search(body):
            out.add(canon)
    return out


def audit_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    head_lines = get_first_meaningful_lines(text, n_lines=10)
    authors = detect_authors(text)

    lang = path.parent.parent.name
    expected_author = path.parent.name

    # Verdict
    if expected_author in authors:
        verdict = "OK"
        # But if there's a foreign author also, that's a misattribution flag
        foreign = authors - {expected_author}
        if foreign:
            verdict = "MIXED"
    else:
        if authors:
            verdict = "MISATTRIBUTED"
        else:
            verdict = "UNKNOWN_TITLE"

    return {
        "path": str(path.relative_to(ROOT)),
        "lang": lang,
        "expected": expected_author,
        "authors_mentioned": sorted(authors),
        "head_first_lines": head_lines[:5],
        "verdict": verdict,
    }


def main():
    if not CORPUS.exists():
        print(f"[audit] corpus not at {CORPUS}", file=sys.stderr)
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

    # Counts
    by_verdict = defaultdict(int)
    by_dir = defaultdict(lambda: {"total": 0, "misat": 0, "mixed": 0})
    for f in findings:
        by_verdict[f["verdict"]] += 1
        key = (f["lang"], f["expected"])
        by_dir[key]["total"] += 1
        if f["verdict"] == "MISATTRIBUTED":
            by_dir[key]["misat"] += 1
        elif f["verdict"] == "MIXED":
            by_dir[key]["mixed"] += 1

    print(f"[audit] {len(findings)} files inspected\n")
    print(f"  verdict counts:")
    for k, v in sorted(by_verdict.items(), key=lambda kv: -kv[1]):
        print(f"    {k:15} {v}")

    print(f"\n  per-author breakdown (misatt = MISATTRIBUTED, mixed = MIXED):")
    for (lang, exp), st in sorted(by_dir.items()):
        flag = ""
        if st["misat"] or st["mixed"]:
            flag = f"  ⚠ MISAT={st['misat']} MIXED={st['mixed']}"
        print(f"    {lang:3}/{exp:12} {st['total']:5} files{flag}")

    # Show details for any flagged
    flagged = [f for f in findings if f["verdict"] in ("MISATTRIBUTED", "MIXED")]
    if flagged:
        print(f"\n  {len(flagged)} flagged files:")
        for f in flagged[:50]:  # first 50
            print(f"    ❌ {f['verdict']:13}  {f['lang']}/{f['expected']}/{Path(f['path']).name}")
            print(f"        authors: {f['authors_mentioned']}")
            print(f"        head: {f['head_first_lines'][0][:120] if f['head_first_lines'] else None}")

    out = ROOT / "results" / "corpus_audit.json"
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(findings, indent=2, ensure_ascii=False))
    print(f"\n[audit] full report: {out}")


if __name__ == "__main__":
    main()
