#!/usr/bin/env python3
"""Upload a paper preprint to Zenodo.

Usage:
  ZENODO_TOKEN=<paste> python scripts/upload_to_zenodo.py \
      --markdown paper/drafts/encoder_only_beats_seq2seq.md \
      --title "Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution" \
      [--sandbox]

Steps:
  1. Convert markdown -> PDF (pandoc + weasyprint fallback)
  2. Create empty deposition (pre-reserves DOI)
  3. Upload PDF to bucket
  4. Set metadata (title, creators, keywords, license, related identifiers)
  5. Publish (mints permanent DOI)
  6. Print DOI + record URL + save to zenodo_results.json
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests required. pip install requests")

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper" / "drafts"
RESULTS_FILE = ROOT / "paper" / "zenodo_results.json"


def load_token(sandbox: bool) -> str:
    env = "ZENODO_SANDBOX_TOKEN" if sandbox else "ZENODO_TOKEN"
    tok = os.environ.get(env)
    if not tok:
        sys.exit(f"Set {env} in env first.")
    return tok.strip()


def render_markdown_to_pdf(md_path: Path, pdf_path: Path) -> Path:
    """Try pandoc -> weasyprint -> reportlab. Whichever works."""
    if pdf_path.exists():
        return pdf_path
    # Try pandoc first
    try:
        r = subprocess.run(
            ["pandoc", str(md_path), "-o", str(pdf_path),
             "--pdf-engine=weasyprint", "--metadata", "lang=en"],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and pdf_path.exists():
            print(f"  PDF rendered via pandoc + weasyprint ({pdf_path.stat().st_size} bytes)")
            return pdf_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: reportlab
    try:
        import markdown
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.units import inch

        with md_path.open(encoding="utf-8") as f:
            md_text = f.read()
        html = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

        styles = getSampleStyleSheet()
        body = styles["BodyText"]
        body.fontSize = 10
        body.leading = 13
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        h3 = styles["Heading3"]

        doc = SimpleDocTemplate(str(pdf_path), pagesize=LETTER,
                                leftMargin=0.9*inch, rightMargin=0.9*inch,
                                topMargin=0.8*inch, bottomMargin=0.8*inch)
        story = []
        for line in html.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.1*inch))
                continue
            if line.startswith("<h1"):
                txt = re.sub(r"<[^>]+>", "", line)
                story.append(Paragraph(txt, h1)); story.append(Spacer(1, 0.15*inch))
            elif line.startswith("<h2"):
                txt = re.sub(r"<[^>]+>", "", line)
                story.append(Paragraph(txt, h2)); story.append(Spacer(1, 0.1*inch))
            elif line.startswith("<h3"):
                txt = re.sub(r"<[^>]+>", "", line)
                story.append(Paragraph(txt, h3)); story.append(Spacer(1, 0.08*inch))
            elif line.startswith("<p"):
                txt = re.sub(r"<[^>]+>", "", line)
                txt = txt.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                story.append(Paragraph(txt, body))
            elif line.startswith("<hr"):
                story.append(Spacer(1, 0.2*inch))
            elif line.startswith("<ul") or line.startswith("<ol"):
                pass
            elif line.startswith("<li"):
                txt = "• " + re.sub(r"<[^>]+>", "", line)
                story.append(Paragraph(txt, body))
            elif line.startswith("<table") or line.startswith("<tr") or line.startswith("<td") or line.startswith("<th"):
                # Skip tables — too complex for this fallback
                pass
            elif line.startswith("</"):
                pass
            else:
                txt = re.sub(r"<[^>]+>", "", line)
                if txt:
                    story.append(Paragraph(txt, body))
        doc.build(story)
        print(f"  PDF rendered via reportlab ({pdf_path.stat().st_size} bytes)")
        return pdf_path
    except ImportError:
        sys.exit("Neither pandoc+weasyprint nor reportlab+markdown are available.")


def upload(args, token: str) -> dict:
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    headers = {"Authorization": f"Bearer {token}"}

    md_path = Path(args.markdown).resolve()
    if not md_path.exists():
        sys.exit(f"Markdown not found: {md_path}")

    pdf_path = md_path.with_suffix(".pdf")
    print(f"[1/5] Rendering PDF from {md_path.name}...")
    pdf_path = render_markdown_to_pdf(md_path, pdf_path)

    print(f"\n[2/5] Creating deposition on {'sandbox' if args.sandbox else 'zenodo.org'}...")
    r = requests.post(f"{base}/deposit/depositions", json={}, headers=headers)
    r.raise_for_status()
    dep = r.json()
    dep_id = dep["id"]
    bucket = dep["links"]["bucket"]
    doi_prereserved = dep["metadata"]["prereserve_doi"]["doi"]
    print(f"  Deposition ID: {dep_id}")
    print(f"  Pre-reserved DOI: {doi_prereserved}")

    print(f"\n[3/5] Uploading PDF...")
    with pdf_path.open("rb") as fp:
        r = requests.put(f"{bucket}/{pdf_path.name}", data=fp, headers=headers)
    r.raise_for_status()
    print(f"  Uploaded {pdf_path.name} ({pdf_path.stat().st_size} bytes)")

    print(f"\n[4/5] Setting metadata...")
    keywords = args.keywords.split(",") if args.keywords else []
    metadata = {
        "metadata": {
            "title": args.title,
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": args.description or "Preprint on cross-lingual authorship attribution.",
            "creators": [{"name": args.author, "affiliation": args.affiliation}],
            "keywords": keywords,
            "publication_date": args.date or time.strftime("%Y-%m-%d"),
            "access_right": "open",
            "license": "cc-by-4.0",
            "language": "eng",
        }
    }
    if args.related:
        metadata["metadata"]["related_identifiers"] = [
            {"identifier": d, "relation": rel, "resource_type": "publication-preprint"}
            for d, rel in [x.split(":", 1) for x in args.related.split(",")]
        ]
    r = requests.put(f"{base}/deposit/depositions/{dep_id}", json=metadata, headers=headers)
    r.raise_for_status()
    print(f"  Title: {args.title}")

    if args.dry_run:
        print("\n[5/5] DRY RUN — skipping publish.")
        print(f"  Pre-reserved DOI would have been: {doi_prereserved}")
        return {"dry_run": True, "dep_id": dep_id, "doi_prereserved": doi_prereserved}

    print(f"\n[5/5] Publishing...")
    r = requests.post(f"{base}/deposit/depositions/{dep_id}/actions/publish", headers=headers)
    r.raise_for_status()
    pub = r.json()
    doi = pub.get("doi") or doi_prereserved
    record_url = pub["links"].get("record_html", f"https://zenodo.org/record/{dep_id}")
    print(f"\n  PUBLISHED")
    print(f"  DOI:        {doi}")
    print(f"  DOI URL:    https://doi.org/{doi}")
    print(f"  Record URL: {record_url}")

    results = []
    if RESULTS_FILE.exists():
        try:
            results = json.loads(RESULTS_FILE.read_text())
        except Exception:
            results = []
    results.append({
        "title": args.title,
        "doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "record_url": record_url,
        "sandbox": args.sandbox,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"  Appended to {RESULTS_FILE.name}")
    return {"doi": doi, "record_url": record_url}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", required=True, help="Path to the paper markdown")
    ap.add_argument("--title", required=True, help="Paper title (for Zenodo metadata)")
    ap.add_argument("--author", default="Raji, Rabiu", help="Author name")
    ap.add_argument("--affiliation", default="Independent", help="Author affiliation")
    ap.add_argument("--description", help="Short abstract / description")
    ap.add_argument("--keywords", default="authorship attribution,stylometry,multilingual NLP,"
                        "low-resource languages,mT5,encoder fine-tuning,seq2seq vs classification",
                        help="Comma-separated keywords")
    ap.add_argument("--date", help="Publication date (YYYY-MM-DD), defaults to today")
    ap.add_argument("--related", help="Comma-separated DOI:relation pairs")
    ap.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    ap.add_argument("--dry-run", action="store_true", help="Render + set metadata, skip publish")
    args = ap.parse_args()

    token = load_token(args.sandbox)
    print(f"Token loaded ({'sandbox' if args.sandbox else 'production'}, "
          f"length={len(token)})")
    upload(args, token)


if __name__ == "__main__":
    main()
