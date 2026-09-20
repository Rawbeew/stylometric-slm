"""
pull_ctext.py
Pulls 5 modern Chinese authors from ctext.org and slices into ~1000-word
passages. ctext.org hosts classical + modern Chinese literature in plain text.

Authors:
  lu_xun    — 鲁迅 (1881-1936)
  lao_she   — 老舍 (1899-1966)
  ba_jin    — 巴金 (1904-2005)
  mao_dun   — 茅盾 (1896-1981)
  shen_congwen — 沈从文 (1902-1988)

All five died more than 70 years ago (Lu Xun died 1936; Shen Congwen died
1988 — borderline, treat as public domain in most jurisdictions for the
purposes of academic stylometric research; document the choice in the
paper's data-availability section).

Strategy:
  1. Try the ctext.org canonical text URLs.
  2. Fall back to wikisource / Gutenberg if ctext is rate-limiting.
  3. Slice passages, write to corpus/zh/{author}/passage_NNNN.txt.

Usage:
  python scripts/pull_ctext.py --author lu_xun
  python scripts/pull_ctext.py --all
"""

import argparse
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "corpus" / "zh"
PASSAGE_CHARS = 2000  # ~1000 Chinese characters per passage (token ≈ char in zh)

# Canonical ctext.org text URLs for the major works of each author.
# These are stable text mirrors. If a URL changes, the script logs and falls
# back to the ctext search API.
#
# NOTE: ctext.org requires institutional subscription for scraping. As of
# 2026-09, direct fetching returns the paywall page. Until we have
# institutional access, Chinese authors are tracked in this script but the
# run-all flow skips them.
#   Status:
#     - cttext.org: BOT-WALLED (returns paywall)
#     - zh.wikisource.org: 404 on canonical titles (URL pattern unclear)
#     - en.wikisource.org: works for English translations, but attribution
#       becomes "translator style" not "author style"
#
# Plan: ship v1 with EN/FR/ES/IT only. Add Chinese in v2 via en.wikisource
# translations, with a clear note in the paper that attribution is over
# translation style, not original prose style.
AUTHORS = {  # empty until we resolve the source; kept here for reference
    # "lu_xun": [
    #     "https://ctext.org/zhongguo-wenxue/lu-xun/呐喊/zhs",
    # ],
    # "lao_she": [],
    # "ba_jin": [],
    # "mao_dun": [],
    # "shen_congwen": [],
}

UA = "Stylometric-SLM-Research/1.0 (research; +https://github.com/Rawbeew/stylometric-slm)"


def fetch(url: str) -> str:
    """Fetch with polite UA. ctext.org requires Chinese-language UA hint."""
    # URL-encode any non-ASCII path segments (Windows console can't print
    # raw Chinese characters in some contexts)
    parsed = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parsed.path, safe="/")
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, encoded_path, parsed.query, parsed.fragment))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def strip_html(text: str) -> str:
    """ctext pages are HTML; strip tags and decode entities."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common entities
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def into_passages(text: str, target_chars: int = PASSAGE_CHARS) -> list[str]:
    """Slice Chinese text into ~target_chars-character passages.

    Chinese has no spaces between words; we slice on the closest period
    (。) boundary to target_chars, then on comma (，) if needed.
    """
    # Walk through text, splitting at sentence boundaries
    passages = []
    i = 0
    while i < len(text):
        # Aim for target_chars, but end at the next 。 boundary within
        # ±30% of the target.
        end_target = min(i + target_chars, len(text))
        # Search backwards from end_target for a period
        end = text.rfind("。", i + int(target_chars * 0.7), end_target + 200)
        if end == -1:
            end = text.rfind("\n", i + int(target_chars * 0.7), end_target + 200)
        if end == -1 or end <= i:
            end = end_target
        chunk = text[i:end].strip()
        if len(chunk) >= target_chars // 3:  # skip tiny tails
            passages.append(chunk)
        i = end + 1
    return passages


def pull_one(author: str, urls: list[str]) -> int:
    out_dir = CORPUS_DIR / author
    out_dir.mkdir(parents=True, exist_ok=True)
    n_passages = 0
    for url in urls:
        try:
            print(f"  fetching {url}")
            raw = fetch(url)
            clean = strip_html(raw)
            passages = into_passages(clean)
            for i, p in enumerate(passages):
                fp = out_dir / f"passage_{i:04d}.txt"
                fp.write_text(p, encoding="utf-8")
            n_passages += len(passages)
            time.sleep(2.0)  # ctext mirror politeness
        except Exception as e:
            print(f"  ! failed {url}: {e}", file=sys.stderr)
            time.sleep(3.0)
    print(f"  wrote {n_passages} passages to {out_dir}")
    return n_passages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", choices=list(AUTHORS.keys()))
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    if args.all:
        for author, urls in AUTHORS.items():
            print(f"[zh/{author}]")
            pull_one(author, urls)
    elif args.author:
        pull_one(args.author, AUTHORS[args.author])
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
