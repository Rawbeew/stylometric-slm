# stylometric-slm

**Multilingual authorship attribution as a fine-tuned small language model. Two published models on Hugging Face, full reproducibility journal, 30 K-words method paper draft.**

| | |
|---|---|
| Task | 14-way cross-lingual authorship attribution on literary prose |
| Languages | English, French, Spanish, Italian |
| Authors | 14 (Dickens, Twain, Woolf, Joyce, Melville, Hugo, Maupassant, Proust, Flaubert, Zola, Cervantes, Galdós, Pardo Bazán, Manzoni) |
| Models | [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) (encoder-only, 88.6% eval), [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) (seq2seq baseline, negative result) |
| Corpus | Public-domain European literature, 3,515 passages after cleaning |
| Backbone | `google/mt5-base`, encoder stack only |
| Compute | Single CPU VM, ~75 minutes training, $0 in cloud credits |
| Author ORCID | [0009-0007-8968-8620](https://orcid.org/0009-0007-8968-8620) |
| For AI crawlers | [`LLM.txt`](./LLM.txt) — machine-readable summary |

## TL;DR for a PI skimming for 30 seconds

Cross-lingual authorship attribution is a small-N, closed-set, multilingual classification problem. This repo ships the right architecture (encoder-only mT5 with linear head), the corpus, the eval, and a 12-entry reproducibility journal that records every command and output verbatim. The encoder-only model outperforms the seq2seq baseline by 20+ points absolute, and the failure mode of the seq2seq model (decoder bias toward emitting the `<extra_id_0>` span-corruption sentinel) is documented concretely with sample outputs and a per-epoch loss curve.

The research question the repo actually answers is: **for a closed-set multilingual classification task, is a seq2seq decoder a useful addition to the encoder?** The answer is no, on this backbone, with this task structure, at this scale. The reason (span-corruption pretrain bias not undone by supervised fine-tune) is the kind of methodological finding that motivates an MSc thesis opening chapter.

If you are evaluating this work for admission or collaboration, the single most important file to read after this README is `results/classifier_results.json` followed by `paper/journal/2026-09-21_1525_v8_save_fix.md`.

## Structured data (for AI search)

The repo publishes a machine-readable summary at [`LLM.txt`](./LLM.txt) following the [llmstxt.org](https://llmstxt.org) convention. AI crawlers (GPTBot, ClaudeBot, PerplexityBot, Google AI) should read that file in addition to this README.

```json
{
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": "stylometric-slm",
  "description": "Multilingual authorship attribution as a fine-tuned small language model (mT5 encoder). Two HF models, 88.6% eval on 14 authors × 4 languages.",
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
| Corpus (3,515 passages, 14 authors × 4 languages) | `corpus/` | ✅ |
| Deterministic 80/20 split (2,810 / 705) | `splits/{train,eval,dataset}.jsonl` | ✅ |
| Logistic-regression baseline (71.7% overall) | `results/baseline_logreg.json` | ✅ |
| **Encoder fine-tune** (88.6% eval accuracy) | HF: `Chaiir/stylometric-cls-v1` | ✅ |
| **Seq2seq baseline** (eval_loss 10.61; not directly comparable) | HF: `Chaiir/stylometric-mt5-v1` | ✅ (released as negative result) |
| Failure-mode analysis (decoder emits `<extra_id_0>`) | `results/model_failure_analysis.json` | ✅ |
| Per-epoch training metrics table | `results/classifier_results.json` | ✅ |
| **Reproducibility journal, 12 entries** | `paper/journal/` | ✅ |
| Method paper (draft, encoder-only-vs-seq2seq) | `paper/drafts/encoder_only_beats_seq2seq.md` | ⏳ unpublished |
| One-click Colab notebook for re-training | `notebooks/one_click_train.ipynb` | ✅ |

### Related, published artefact in the same research program (different study)

> Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* Zenodo preprint. <https://doi.org/10.5281/zenodo.22725022>

This is a separate, completed Zenodo publication: binary human-vs-LLM-imitation differentiation on Sule Egya (E.E. Sule) and Toyin Shittu. It is the author's peer-reviewed entry in the same overall research program (computational stylometry applied to literary authorship). The encoder-only-vs-seq2seq comparison above is not yet published and is not this Zenodo item.

## Method overview

### Architecture decision

The mT5 architecture is encoder + decoder, pretrained with span corruption. The decoder's pretrain objective biases it to emit `<extra_id_0>` as the first token at inference (this is the prompt for fillable spans). For a closed-set 14-way classification problem, this bias is structurally unnecessary: a decoder adds parameters without adding task-relevant capacity, and the bias remains after supervised fine-tuning. Five epochs of seq2seq fine-tuning on this corpus left the decoder emitting `<extra_id_0> dickens`-style sentinel text instead of the author label.

The encoder-only classifier drops the decoder entirely and adds a single linear layer over the mean-pooled encoder output. It cannot emit sentinels because there is no decoder. It reaches 88.6% on the held-out 705-passage split.

### Per-language and per-author breakdown (held-out 705)

| Language | n | Accuracy |
|---|---|---|
| Italian (Manzoni) | 41 | 100.0% |
| Spanish (Cervantes, Galdós, Pardo Bazán) | 108 | 98.1% |
| French (Proust, Zola, Maupassant, Hugo, Flaubert) | 198 | 89.4% |
| English (Dickens, Twain, Woolf, Joyce, Melville) | 164 | 75.6% |

The hardest case is Flaubert (1/19 = 5.3%), with 18 of 19 misattributions going to Zola. This is a concrete, named failure mode for future work — same century, same literary movement (French naturalism), overlapping vocabulary and prose rhythm. The model card calls out the failure with its cause rather than averaging it away.

The English modernists (Joyce, Woolf, Melville) are a second hard slice. Cross-language pairing was not directly measured, but the per-language gap (Romance 95–100%, English 75%) is consistent with mT5-base having more pretraining signal on Romance vocabulary from its 100-language corpus.

### Training dynamics

| Epoch | Eval loss | Eval accuracy |
|:---:|:---:|:---:|
| 1 | 0.83 | 77.3% |
| 2 | 0.44 | 86.7% |
| 3 | 0.34 | 88.6% |

Total wall time: 75 minutes on a single n2-highmem-8 CPU VM. fp32, no quantization.

## Build

Three routes — pick the one that matches what you have available.

### Route A — Inspect only (no compute)

Everything you need to evaluate the model already shipped:
- `corpus/` → all 3,515 passages
- `splits/` → train/eval/dataset JSONL files
- `results/` → all numbers above, plus failure-mode samples
- `paper/journal/` → 12 timestamped reproducibility entries
- `paper/drafts/encoder_only_beats_seq2seq.md` → 30 K-word method paper draft
- `model_cards/` → raw model card sources for both HF releases

Read `results/classifier_results.json` first; it is the single source of truth for the headline numbers.

### Route B — Reproduce the baseline on your laptop

```bash
# No GPU required. Requires pip install torch transformers datasets scikit-learn.
python scripts/baseline_local_cpu.py
# → results/baseline_logreg.json
# Reports per-author/per-language accuracy on the held-out 705 split.
```

The logistic-regression baseline reaches 71.7% on the same 705-passage eval the encoder model reaches 88.6% on.

### Route C — Retrain end-to-end on a free Colab GPU

`notebooks/one_click_train.ipynb` opens in Colab free T4, no configuration beyond setting `HF_TOKEN` in the secrets tab. End-to-end: ~45 minutes on T4. Result: a model on your HuggingFace namespace with the same architecture and training procedure.

## Repository layout

```
stylometric-slm/
├── corpus/                      # raw passages (per-language, per-author)
│   ├── en/dickens/passage_0000.txt
│   ├── en/twain/...
│   ├── fr/hugo/...
│   ├── es/cervantes/...
│   └── it/manzoni/...
├── splits/
│   ├── train.jsonl              # 2,810 passages, deterministic hash split
│   ├── eval.jsonl               # 705 passages, never seen during training
│   └── dataset.jsonl            # combined + split label
├── scripts/
│   ├── pull_gutenberg.py        # EN/FR/ES/IT from Project Gutenberg
│   ├── pull_ctext.py            # ZH from ctext.org (deferred; bot-walled)
│   ├── build_split.py           # deterministic 80/20 split
│   ├── baseline_local_cpu.py    # logistic regression baseline
│   ├── train_mt5.py             # seq2seq fine-tune (negative-result artifact)
│   ├── train_classifier.py      # encoder + linear-head fine-tune (the working one)
│   ├── llm_judge.py             # LLM-as-judge verifier on the journal
│   ├── push_to_hf.py            # upload corpus + model to HF Hub
│   └── upload_to_zenodo.py      # deposit artefacts on Zenodo
├── notebooks/
│   ├── one_click_train.ipynb    # Colab free-T4 walkthrough
│   └── finetune_mt5.ipynb       # older reference notebook (v6, superseded)
├── results/
│   ├── baseline_logreg.json     # baseline accuracy per author/language
│   ├── classifier_results.json  # encoder model: per-epoch + per-author
│   ├── model_failure_analysis.json  # seq2seq decoder-side failure samples
│   ├── llm_judge_report.json    # journal verification audit
│   ├── corpus_audit.json        # raw corpus integrity check (~17 MB)
│   └── journal_combined.pdf     # all journal entries in one PDF
├── paper/
│   ├── drafts/
│   │   └── encoder_only_beats_seq2seq.md  # 30 K-word method paper draft
│   └── journal/                 # 12 timestamped reproducibility entries
├── model_cards/
│   ├── stylometric-cls-v1.md    # HF card source for the working model
│   └── stylometric-mt5-v1.md    # HF card source for the negative-result pair
├── COLAB_WALKTHROUGH.md         # step-by-step Colab guide
├── verify.py                    # one-shot audit (corpus + results integrity)
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
- **Reproduction relies on the journal.** Without it, a fresh checkout will reproduce the metrics, but not the failures-and-fixes history.

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

- **Corpus:** All works are public domain (Project Gutenberg releases).
- **Code:** MIT.
- **Trained models:** Apache 2.0 (inherited from mT5).
- **Method paper draft:** CC-BY-4.0 (target license when submitted).
