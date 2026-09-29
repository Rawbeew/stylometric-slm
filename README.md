# stylometric-slm

**Multilingual authorship attribution as a fine-tuned small language model. Two published models on Hugging Face, full reproducibility journal, 30 K-words method paper draft.**

| | |
|---|---|
| Task | 14-way cross-lingual authorship attribution on literary prose |
| Languages | English, French, Spanish, Italian |
| Authors | 14 (Dickens, Twain, Woolf, Joyce, Melville, Hugo, Maupassant, Proust, Flaubert, Zola, Cervantes, Galdós, Pardo Bazán, Manzoni) |
| Models | [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) (encoder-only, **91.2%** on full 705-passage eval; 88.6% on per-author 50-passage-sampled breakdown), [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) (seq2seq baseline, negative result) |
| Corpus | Public-domain European literature, 3,515 passages after cleaning (see "Corpus status" below — rebuild path, not pre-shipped data) |
| Backbone | `google/mt5-base`, encoder stack only |
| Compute | Single CPU VM (n2-highmem-8, 64 GB RAM). Single training run: ~75 minutes. Cumulative project compute: **88 hours / $47.52 USD** across 4 days (including debugging, failed save attempts, and 6 documented snags) — see `paper/journal/2026-09-24_comprehensive_analysis.md` for the line-item audit. The "$0 / 75 min" headline in earlier journal entries was an estimate written before the bill landed; it was corrected in the Sept 24 entry. |
| Author ORCID | [0009-0007-8968-8620](https://orcid.org/0009-0007-8968-8620) |
| For AI crawlers | [`LLM.txt`](./LLM.txt) — machine-readable summary |

## TL;DR for a PI skimming for 30 seconds

Cross-lingual authorship attribution is a small-N, closed-set, multilingual classification problem. This repo ships the right architecture (encoder-only mT5 with linear head), the eval, and a 12-entry reproducibility journal that records every command and output verbatim. The encoder-only model outperforms the seq2seq baseline by 20+ points absolute, and the failure mode of the seq2seq model (decoder bias toward emitting the `<extra_id_0>` span-corruption sentinel) is documented concretely with sample outputs and a per-epoch loss curve.

The research question the repo actually answers is: **for a closed-set multilingual classification task, is a seq2seq decoder a useful addition to the encoder?** The answer is no, on this backbone, with this task structure, at this scale. The reason (span-corruption pretrain bias not undone by supervised fine-tune) is a transferable methodological finding for any researcher choosing between encoder-only and seq2seq heads on a small, closed label set.

If you are reviewing this work, the single most important file to read after this README is `results/classifier_results.json` followed by `paper/journal/2026-09-21_1525_v8_save_fix.md`.

## Structured data (for AI search)

The repo publishes a machine-readable summary at [`LLM.txt`](./LLM.txt) following the [llmstxt.org](https://llmstxt.org) convention. AI crawlers (GPTBot, ClaudeBot, PerplexityBot, Google AI) should read that file in addition to this README.

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "stylometric-slm",
  "description": "Multilingual authorship attribution as a fine-tuned small language model (mT5 encoder). Two HF models, 91.2% on full 705-passage eval (88.6% on per-author 50-passage-sampled breakdown).",
  "author": {
    "@type": "Person",
    "name": "Rabiu Raji",
    "identifier": "https://orcid.org/0009-0007-8968-8620"
  },
  "codeRepository": "https://github.com/Rawbeew/stylometric-slm",
  "programmingLanguage": ["Python"],
  "license": "https://www.apache.org/licenses/LICENSE-2.0",
  "keywords": "stylometry,authorship-attribution,mT5,multilingual,encoder-only-finetuning",
  "relatedLink": [
    "https://huggingface.co/Chaiir/stylometric-cls-v1",
    "https://huggingface.co/Chaiir/stylometric-mt5-v1",
    "https://doi.org/10.5281/zenodo.22725022"
  ]
}
```

## What is here

### Working artefacts

| Artefact | Where | State |
|---|---|---|
| Logistic-regression baseline (71.7% overall)¹ | `results/baseline_logreg.json` | ✅ |
| Encoder-only classifier eval (91.2% full / 88.6% sampled) | `results/classifier_results.json` | ✅ |
| Seq2seq failure samples + per-epoch loss | `results/model_failure_analysis.json` | ✅ |
| LLM-as-judge audit of the journal | `results/llm_judge_report.json` | ✅ |
| Reproducibility journal (12 entries) | `paper/journal/` | ✅ |
| Method paper draft (30 K words) | `paper/drafts/encoder_only_beats_seq2seq_v2.md` | ✅ |
| Trained model card sources | `model_cards/` | ✅ |
| Combined journal PDF | `results/journal_combined.pdf` | ✅ |
| Corpus (3,515 passages, 14 authors × 4 languages) | **not shipped — see "Corpus status"** | ⚠ rebuild |
| Deterministic 80/20 split (2,810 / 705) | **not shipped — see "Corpus status"** | ⚠ rebuild |

¹ **Baseline note:** `0.717` is the **simple cumulative accuracy** across all 605 baseline eval passages (434/605). The per-language breakdown in `results/baseline_logreg.json` is en=0.736, es=0.879, fr=0.666, it=0.429 (uneven eval slice per language). The headline 71.7% is the overall cumulative; it is the comparison point for the encoder model's 88.6% (per-author sampled) and 91.2% (full 705).

### Related, published artefact in the same research program (different study)

[`doi:10.5281/zenodo.22725022`](https://doi.org/10.5281/zenodo.22725022) — *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* Burrows' Delta + classifier on Sule Egya and Toyin Shittu (Nigerian English). Different corpus, different languages, different conclusion. Cite this if your work is on contemporary African poetry.

## Corpus status — important for reviewers

**`corpus/` and `splits/` are gitignored** (30 MB + 41 MB of text, derived from Project Gutenberg). They are NOT in this repo. To reproduce:

```bash
# Pull EN/FR/ES/IT from Project Gutenberg (no API key; ~30 min)
python scripts/pull_gutenberg.py --out corpus/

# Build the deterministic 80/20 split (hash-based, stratified per author)
python scripts/build_split.py --corpus corpus/ --out splits/

# Now you can run baseline + training locally
python scripts/baseline_local_cpu.py
python scripts/train_classifier.py
```

The full path is also documented in `paper/journal/2026-09-20_1810_data_integrity_snag.md` and `paper/journal/2026-09-21_1040_v7_classification_fix.md`. A prebuilt split is **not** mirrored on Hugging Face — `scripts/push_to_hf.py` is provided but its presence is for parity with the training pipeline, not a public mirror.

If you only want to inspect what shipped (eval results, journal, model cards), **you do not need the corpus** — start at `results/classifier_results.json`.

## Method overview

### Architecture decision

`MT5EncoderModel` (encoder stack of `google/mt5-base`, decoder weights discarded) + a `nn.Linear(hidden, 14)` classification head on the mean-pooled encoder output. ~278M encoder parameters + ~11K head. mT5 is span-corrupt pretrained; using only the encoder sidesteps the decoder bias toward emitting the `<extra_id_0>` sentinel — the methodological finding that motivates the paper.

### Per-language and per-author breakdown (held-out 705)

(Sampled 50 passages per author for tractability in the breakdown table; total 511. The "full" headline 91.2% is on all 705 eval passages.)

| Lang | Authors | Per-lang accuracy |
|---|---|---|
| Italian | Manzoni | **100.0%** (41/41) |
| Spanish | Cervantes, Galdós, Pardo Bazán | **98.1%** (106/108) |
| French | Hugo, Maupassant, Proust, Flaubert, Zola | **89.4%** (177/198) |
| English | Dickens, Twain, Woolf, Joyce, Melville | **78.7%** (129/164) |

11 of 14 authors exceed 70% accuracy; 6 of 14 hit 100%. Named failure: **Flaubert** — 1/19 eval passages. Likely confused with Zola (same period, same school, both French Naturalist).

### Training dynamics

| Epoch | Eval loss | Eval accuracy |
|---|---|---|
| 1 | 0.83 | 77.3% |
| 2 | 0.44 | 86.7% |
| 3 | 0.34 | 91.2% |

Note: numbers above are the **full 705-passage** eval. The per-author breakdown table lower in this file caps each author at 50 passages (total 511), which gives the 88.6% figure used in the per-author table. Both numbers are correct — they measure different denominators (full eval vs per-author sampled).

**Single training run wall time:** ~75 minutes on a single n2-highmem-8 CPU VM, fp32, no quantization. The 88-hour project total in the table at the top includes 4 days of debugging, two lost-model save attempts, and the full snag log — see `paper/journal/2026-09-24_comprehensive_analysis.md`.

## Build

Four routes — pick the one that matches what you have available.

### Route A — Inspect only (no compute, 30 seconds)

```bash
python verify.py
# → prints SHA256 + syntax-check status for 17 files in scripts/, notebooks/, README
```

This validates the tracked artefacts: every script compiles, every notebook is valid JSON, the README is the one on disk. If you only want to **read** the work, this is the only command you need; the repo ships `results/`, `paper/`, `model_cards/`, and the journal.

### Route B — Reproduce the baseline on your laptop (~5 min)

After `python scripts/pull_gutenberg.py` + `python scripts/build_split.py` (the data rebuild path; see above):

```bash
# No GPU required. Requires pip install torch transformers datasets scikit-learn.
python scripts/baseline_local_cpu.py
# → results/baseline_logreg.json
# Reports per-author/per-language accuracy on the held-out 705 split.
```

The logistic-regression baseline reaches 71.7% (simple cumulative across all 605 baseline eval passages — 434/605) on roughly the same eval the encoder model reaches **91.2%** on (88.6% on the per-author sampled breakdown).

### Route C — Retrain the seq2seq variant on Colab (~45 min, negative result)

`notebooks/one_click_train.ipynb` opens in Colab free T4, no configuration beyond setting `HF_TOKEN` in the secrets tab. End-to-end: ~45 minutes on T4. **Caveat:** this notebook trains the **seq2seq** variant (`AutoModelForSeq2SeqLM`) — the architecture that produces the negative result. It does NOT reproduce the 91.2% encoder classifier. To reproduce the 91.2% model on Colab, follow the encoder-swap recipe in [`COLAB_WALKTHROUGH.md`](./COLAB_WALKTHROUGH.md#encoder-swap--reproduce-the-912-model-on-colab) (replace `AutoModelForSeq2SeqLM` with the `MT5EncoderModel` + `nn.Linear` head wrapper; ~20 lines).

### Route D — Full re-pull + train on your own data (~3 hours)

```bash
python scripts/pull_gutenberg.py --out corpus/   # rebuild from PG
python scripts/build_split.py --corpus corpus/ --out splits/
python scripts/train_classifier.py --epochs 3    # ~75 min on n2-highmem-8 CPU
```

This is the path the journal documents. End state: identical eval numbers to the published results, plus the full reproducibility journal trail.

## Repository layout

```
stylometric-slm/
├── corpus/                      # [gitignored] raw passages; rebuild with pull_gutenberg.py
├── splits/                      # [gitignored] train/eval/dataset JSONL; rebuild with build_split.py
├── scripts/
│   ├── pull_gutenberg.py        # EN/FR/ES/IT from Project Gutenberg
│   ├── pull_ctext.py            # ZH from ctext.org (deferred; bot-walled)
│   ├── build_split.py           # deterministic 80/20 split
│   ├── clean_passages.py        # strip front-matter, dedupe, length filter
│   ├── baseline_local_cpu.py    # logistic regression baseline (no GPU)
│   ├── train_mt5.py             # seq2seq fine-tune (negative-result artifact)
│   ├── train_classifier.py      # encoder + linear-head fine-tune (the working one)
│   ├── llm_judge.py             # LLM-as-judge verifier on the journal
│   ├── push_to_hf.py            # upload corpus + model to HF Hub
│   └── upload_to_zenodo.py      # deposit artefacts on Zenodo
├── notebooks/
│   ├── one_click_train.ipynb    # Colab free-T4 walkthrough (the one to use)
│   └── finetune_mt5.ipynb       # older reference notebook (v6, superseded)
├── results/
│   ├── baseline_logreg.json     # baseline accuracy per author/language
│   ├── classifier_results.json  # encoder model: per-epoch + per-author
│   ├── model_failure_analysis.json  # seq2seq decoder-side failure samples
│   ├── llm_judge_report.json    # journal verification audit
│   ├── corpus_audit*.json       # raw corpus integrity checks
│   └── journal_combined.pdf     # all journal entries in one PDF
├── paper/
│   ├── drafts/
│   │   ├── encoder_only_beats_seq2seq.md      # v1 (initial draft, ORCID placeholder, kept for history)
│   │   └── encoder_only_beats_seq2seq_v2.md   # v2 (canonical draft, correct ORCID, journal-aligned numbers)
│   └── journal/                 # 12 timestamped reproducibility entries
├── model_cards/
│   ├── stylometric-cls-v1.md    # verbatim HF card for the working model (synced from Hub)
│   └── stylometric-mt5-v1.md    # verbatim HF card for the negative-result pair (synced from Hub)
├── COLAB_WALKTHROUGH.md         # step-by-step Colab guide
├── verify.py                    # one-shot audit (scripts + notebooks)
├── LICENSE                      # Apache 2.0
└── README.md
```

## Reproducibility journal

The journal in `paper/journal/` is the single most important file in this repo for anyone trying to reproduce the work. Twelve timestamped entries (`YYYY-MM-DD_HHMM_topic.md`) record every command, output, error, fix, and decision from initial Cloudflare-tunnel-and-GCP setup on Sept 20, 2026, through to the final encoder-only save fix on Sept 21, 2026, and a comprehensive analysis on Sept 24.

Each entry:
1. Quotes the command verbatim.
2. Captures the output verbatim.
3. Names the file or commit that fixed each error.
4. Cross-checks claims against the journal entries.

`results/llm_judge_report.json` is the LLM-as-judge audit pass: every claim of "this worked" in the journal has its corresponding output snippet in the entry where the claim was made, verified by an independent model.

## Limitations (honest)

- **Small corpus, small author set.** 14 authors × 4 languages × 50–500 passages. This is a methodological probe, not a production attribution system.
- **Genre-fixed.** Trained on 19th- and 20th-c. literary prose. News, social media, technical writing untested.
- **No closed-set / open-set hybrid.** The classifier has no fallback for unknown authors. Apply to text by authors outside the 14-way label set and the model still emits one of the 14.
- **English modernists are a hard slice.** Melville 53%, Joyce 68%, Woolf 72%. Forcing all of language and all of literary history into one model exposes this; a per-language specialist fine-tune would likely do better.
- **Flaubert is the named failure.** 1/19. Document this in any downstream use of the model.
- **Reproduction requires re-pulling the corpus.** `corpus/` and `splits/` are gitignored to keep the repo small. Rebuild takes ~30 min from Project Gutenberg; full path documented in "Corpus status" above.

## Citation

### Method comparison (encoder-vs-seq2seq) — unpublished method paper, do not cite as a publication

If using the encoder model in research without the upcoming preprint, cite the model card on HuggingFace and the repo:

```bibtex
@misc{stylometric_cls_v1_2026,
  title  = {Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora (unpublished draft)},
  author = {Raji, Rabiu},
  year   = {2026},
  orcid  = {0009-0007-8968-8620},
  url    = {https://huggingface.co/Chaiir/stylometric-cls-v1}
}
```

### Published related work in the same research program

```bibtex
@misc{raji_2026_voice_or_mask,
  title  = {Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets},
  author = {Raji, Rabiu},
  year   = {2026},
  doi    = {10.5281/zenodo.22725022},
  url    = {https://doi.org/10.5281/zenodo.22725022}
}
```

## License

- **Code:** Apache 2.0. See [LICENSE](./LICENSE).
- **Corpus:** All works are public domain (Project Gutenberg releases). Derived text inherits the underlying public-domain status; no additional grant is required.
- **Trained models:** Apache 2.0 (inherited from mT5).
- **Method paper draft:** CC-BY-4.0 (target license when submitted).