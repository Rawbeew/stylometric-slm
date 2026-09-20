#!/usr/bin/env python3
"""
clean_passages.py — Drop the first N passages of each author that may contain
the author's name in the text (title pages, dedications, prefaces that quote
the author, table of contents, transcriber notes, etc.).

Default: drop first 3 passages. Also drop ANY passage whose first 1000 chars
contain the author's surname — these are still contaminated (e.g., prefaces
that go past p0002).
"""
import re
import sys
from pathlib import Path

ROOT = Path("C:/Users/alaga/ghwork/stylometric-slm")
CORPUS = ROOT / "corpus"

# Author surname regexes (lowercase, accent-stripped)
# These match first ~10 chars of the surname
AUTHOR_PATTERNS = {
    # EN
    "dickens": r"dickens",
    "twain": r"twain|mark twain|samuel clemens|s.l. clemens",
    "melville": r"melville|herman melv",
    "woolf": r"woolf|virginia woolf",
    "joyce": r"joyce|james joyce",
    # FR
    "hugo": r"hugo|victor hugo",
    "maupassant": r"maupassant|guy de maupassant",
    "flaubert": r"flaubert|gustave flaub",
    "proust": r"proust|marcel proust",
    "zola": r"zola|[ée]mile zola|[ée]mile z",
    # ES
    "cervantes": r"cervantes|miguel de cervantes",
    "galdos": r"gald[oó]s|benito p[ée]rez gald[oó]s|gald[oó]s",
    "pardo_bazan": r"pardo|baz[áa]n|emilia pardo",
    # IT
    "manzoni": r"manzoni|alessandro manzoni",
}

def norm(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace."""
    import unicodedata
    s = text.lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", s)


def main():
    drop_first = 3  # always drop first 3 passages per author (title/preface)
    stats = {}
    total_kept = 0
    total_dropped = 0

    for lang_dir in sorted(CORPUS.iterdir()):
        if not lang_dir.is_dir():
            continue
        for author_dir in sorted(lang_dir.iterdir()):
            if not author_dir.is_dir():
                continue
            author = author_dir.name
            # Pick pattern
            pat = AUTHOR_PATTERNS.get(author)
            if pat is None:
                print(f"  WARN: no pattern for {author}, skipping safety drop")
                pat = author.replace("_", r"[ _]")
            regex = re.compile(pat, re.IGNORECASE)

            files = sorted(author_dir.glob("passage_*.txt"))
            kept = 0
            dropped = 0
            reasons = []

            for i, f in enumerate(files):
                # Always drop first 3
                if i < drop_first:
                    dropped += 1
                    reasons.append((f.name, "always_drop_first_3"))
                    f.unlink()
                    continue

                # Safety check: any passage whose first 1000 chars mention author
                txt = f.read_text(encoding="utf-8", errors="replace")[:1000]
                if regex.search(norm(txt)):
                    dropped += 1
                    reasons.append((f.name, "author_name_in_text"))
                    f.unlink()
                    continue

                kept += 1

            stats[f"{lang_dir.name}/{author}"] = (kept, dropped, reasons)
            total_kept += kept
            total_dropped += dropped

    print(f"\n{'='*60}")
    print(f"{'Lang/Author':<25s} {'Kept':>6s} {'Dropped':>8s}  Why dropped")
    for k, (k_, d_, rs) in stats.items():
        if d_ > 0:
            sample = ", ".join(f"{n}({r})" for n, r in rs[:3])
            print(f"  {k:<23s} {k_:>6d} {d_:>8d}  {sample}")
        else:
            print(f"  {k:<23s} {k_:>6d} {d_:>8d}  -")
    print(f"\nTotal kept: {total_kept}")
    print(f"Total dropped: {total_dropped}")


if __name__ == "__main__":
    main()
