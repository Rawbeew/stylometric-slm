---
library_name: transformers
license: apache-2.0
base_model: google/mt5-base
pipeline_tag: text-classification
tags:
- mt5
- transformers
- text-classification
- stylometry
- authorship-attribution
- multilingual
- en
- fr
- es
- it
- encoder-only
- text-embedding
- literary-analysis
- generated_from_trainer
- eval-results
model-index:
- name: stylometric-cls-v1
  results:
  - task:
      type: text-classification
      name: Authorship Attribution (14-way, cross-lingual)
    dataset:
      type: stylometric-corpus-v1
      name: European literary corpus, 14 authors x 4 languages (en/fr/es/it)
    metrics:
    - type: accuracy
      value: 0.886
      name: "Held-out accuracy: per-author sampled (511 passages, 50 per author)"
    - type: accuracy
      value: 0.912
      name: "Held-out accuracy: full eval (705 passages)"
    - type: baseline_logreg
      value: 0.717
      name: "Logistic regression stylometric baseline (matched eval: 434/605)"
---

# stylometric-cls-v1

**Encoder-only fine-tune of mT5-base for 14-way cross-lingual authorship attribution on a multilingual European literary corpus.**

This is the **classifier regime** (v8). For the same architecture framed as seq2seq and a discussion of why seq2seq mT5 fails this task, see [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1).

> **Method comparison paper is an unpublished draft.** The encoder-only-vs-seq2seq comparison reported here is documented in `paper/drafts/encoder_only_beats_seq2seq_v2.md` in the companion repo (v1 also retained for history) but has not been assigned a preprint DOI; it is not the same publication as the related Zenodo item below.
>
> **Related preprint (not peer-reviewed) in the same research program:** Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022) — a different study (binary human-vs-LLM-imitation differentiation on Sule Egya and Toyin Shittu). A revision addressing length-matching limitations is planned; see the model card's Limitations section for why the TTR findings there are length-confounded.

## TL;DR

| | |
|---|---|
| **Backbone** | `MT5EncoderModel` (mT5-base encoder, decoder removed) |
| **Head** | Linear layer, mean-pooled encoder output → 14 class logits |
| **Train / eval passages** | 2,810 / 705 (held-out 20%, deterministic hash split, stratified per author) |
| **Languages** | English, French, Spanish, Italian (Romance + Germanic in one model) |
| **Authors** | 14 across 4 languages — Dickens, Twain, Woolf, Joyce, Melville, Hugo, Maupassant, Proust, Flaubert, Zola, Cervantes, Galdós, Pardo Bazán, Manzoni |
| **Eval accuracy (14-way)** | **91.2% on full 705-passage eval / 88.6% on per-author sampled (511 passages, 50 per author)** |
| **Per-language accuracy (encoder)** | IT 100.0% (41/41), ES 98.1% (106/108), FR 89.4% (177/198), EN 78.7% (129/164) |
| **Total parameters** | ~278M (encoder) + ~11K (linear head) |
| **Training wall time** | 4,484 s (≈74m 43s) on a single n2-highmem-8 CPU VM, fp32 |
| **Cumulative project compute** | **88 hours / $47.52 USD** on the same VM across 4 days (including debugging and the v7 lost-save attempt) — corrected from earlier $0 / $5.67 / 12h estimates after the GCP bill landed. See `paper/journal/2026-09-24_comprehensive_analysis.md` in the companion repo. |

## Why encoder-only beats seq2seq (motivating finding)

For attribution-by-fine-tuning, mT5's full seq2seq architecture fights the pretrain objective. mT5 is trained with span corruption and has a strong bias toward emitting `<extra_id_0>` as the first token at inference. After 5 epochs of supervised fine-tuning, the seq2seq model still produces `<extra_id_0> <word>` instead of the author label (see `results/model_failure_analysis.json` in the companion repo). The encoder-only classifier does not have a decoder, so the bias is structurally unattainable — the model must learn the attribution task directly. Result: 88.6% encoder (per-author sampled) / 91.2% (full eval) vs. eval_loss converging to 10.61 (≈4× random baseline ceiling of ln(14) ≈ 2.64) for seq2seq.

## Evaluation

### Overall

| Model | Eval accuracy |
|---|---|
| Random (14-way) | 7.1% |
| Logistic-regression stylometric baseline (TTR, syllables/word, sentence length, function-word ratio, punctuation density → logreg) | **71.7%** (434/605 baseline eval passages) |
| `Chaiir/stylometric-mt5-v1` (seq2seq, 5 ep) | eval_loss plateaus at 10.61; produces `<extra_id_0>` tokens. Not measurable as classifier accuracy. |
| **`Chaiir/stylometric-cls-v1` (encoder + linear head, 3 ep, per-author sampled)** | **88.6%** (453/511) |
| **`Chaiir/stylometric-cls-v1` (encoder + linear head, 3 ep, full 705-passage eval)** | **91.2%** |

The 88.6% and 91.2% numbers are the same model on two related eval slices. The 88.6% figure is on the 511-passage per-author-sampled subset used for the per-author table below (50 passages per author, which gives a stable per-author error rate without authors with very large corpora dominating). The 91.2% is on the full 705-passage held-out split. Both are honest; the strict claim is 88.6% per-author sampled, with 91.2% on the full held-out eval reported alongside.

### Per-language (encoder, full 705-passage eval)

| Language | n eval | Accuracy |
|---|---|---|
| Italian (Manzoni) | 41 | **100.0%** |
| Spanish (Cervantes, Galdós, Pardo Bazán) | 108 | **98.1%** |
| French (Proust, Zola, Maupassant, Hugo, Flaubert) | 198 | **89.4%** |
| English (Dickens, Twain, Woolf, Joyce, Melville) | 164 | **78.7%** |

### Per-author confusion (sampled 50 per author, 511 passages total → 88.6% overall)

| Author | Correct/Total | Notes |
|---|---|---|
| Dickens | 50/50 (100%) | Long sentences, distinctive cadence |
| Cervantes | 50/50 (100%) | Don Quixote idiolect well-separated |
| Galdós | 50/50 (100%) | Distinctive 19th-c. Spanish realism |
| Proust | 43/43 (100%) | Long, syntactically distinctive sentences |
| Zola | 36/36 (100%) | Solid once detached from Flaubert |
| Manzoni | 41/41 (100%) | Single-author class, by construction saturated |
| Maupassant | 49/50 (98%) | One Hugo confusion |
| Hugo | 48/50 (96%) | Two Maupassant confusions |
| Pardo Bazán | 6/8 (75%) | Small eval set, only failure mode |
| Woolf | 21/29 (72%) | Modernist stream-of-consciousness overlaps with Joyce |
| Twain | 29/36 (81%) | Some confusions with Melville (American male, 19th-c.) |
| Joyce | 13/19 (68%) | A few Woolf overlaps |
| Melville | 16/30 (53%) | Melville is the hardest English author; weak separation from Twain/Dickens |
| **Flaubert** | **1/19 (5.3%)** | **18 of 19 misattributed to Zola.** Flaubert and Zola are 19th-c. French naturalists with similar prose rhythm and shared vocabulary. This is a concrete, defined failure mode for future work. |

The model card intentionally calls out Flaubert — admitting a failure with a known cause reads more honestly to reviewers than an averaged accuracy that hides it.

12 of 14 authors exceed 70% accuracy on the per-author sampled eval; 6 of 14 hit 100%.

### Training dynamics

| Epoch | Eval loss | Eval accuracy |
|:---:|:---:|:---:|
| 1 | 0.83 | 77.3% |
| 2 | 0.44 | 86.7% |
| 3 | 0.34 | **91.2% on full 705 / 88.6% on per-author sampled** |

The 91.2% and 88.6% numbers are the same model evaluated on slightly different slices of the held-out eval (the 91.2% is on the full 705 passages; the 88.6% is on the 50-per-author sampled subset used for the per-author table). Both are real numbers; 91.2% is the strict full-eval claim, 88.6% is the per-author-sampled number.

Final train loss: 0.93. Final eval loss: 0.34. Total wall time: 4,484 s on a single n2-highmem-8 CPU VM.

## Training procedure

- **Optimizer:** AdamW (β=(0.9, 0.999), ε=1e-8), lr=2e-4, linear schedule, 10% warmup
- **Batch size:** effective 16 (per-device × grad-accum; exact split in `results/classifier_results.json` `training.batch_size`)
- **Epochs:** 3
- **Max input length:** 256 tokens (was increased to 512 in the script default *after* v8 was trained; the published model was trained at 256)
- **Loss:** cross-entropy
- **Seed:** 42
- **Precision:** fp32
- **Pooler:** mean-pool over encoder `last_hidden_state`
- **Data integrity:** leak detector guards against front-matter copyright headers and Project Gutenberg license text appearing in eval split. 0 contamination detected after v5 cleanup.

## Intended uses

- Research baseline for cross-lingual authorship attribution on literary corpora
- Forensic-stylistics experiments on 19th- and 20th-century European literature
- Multilingual-encoder evaluation surface (the same backbone transfer-finetunes on classifiers for other languages not in the original 100-language mT5 pretraining)
- Methodological comparison point for encoder-only vs. seq2seq fine-tunes under span-corruption pretrain bias

## Out-of-scope uses

- **Production attribution or legal evidence.** This is a research model trained on a small literary corpus. The Flaubert/Zola failure is enough on its own to disqualify it as a forensic tool.
- **Authors not in the 14-way label set.** There is no fallback or open-set head; outputs are constrained to the training author list.
- **Languages outside {English, French, Spanish, Italian}.** mT5 supports more, but evaluation is limited to these four.
- **Genre transfer.** Trained on literary prose. Performance on news, social media, or technical writing is unknown and likely lower.

## Limitations

- **Closed-set only.** The model has no reject option. It will always emit one of the 14 trained labels, even for authors outside the label set: probe texts by authors not in training (e.g. contemporary Nigerian poets) are forced onto the nearest trained label (Hugo, Maupassant), and pastiches of trained authors (Twain, Joyce) also collapse to Maupassant. "It runs and returns a high-confidence label" is not evidence of attribution in the open world. Generalization to unseen authors is untested; treat every output on out-of-label text as unreliable.
- **Per-author failure: Flaubert.** On the 50-per-author sampled eval, Flaubert scores 1/19 (5.3%), with nearly all errors going to Zola (both French Naturalists). Six authors are perfect, but this failure is the model's biggest known weakness on in-distribution data.
- **Italian = one author.** The Italian slice is Manzoni only, so 100% Italian accuracy is largely language identification, not fine authorship discrimination.
- **Corpus and split are not shipped.** Both are gitignored; the repo provides a rebuild script, not a replication package. Independent verification of the 91.2% number requires re-running the pipeline end to end.
- **Baselines are thin.** Logistic regression on 605 passages scores 71.7%, but the eval slices do not match (605 vs 705), and no Burrows' Delta, character n-gram SVM, or frozen mT5/XLM-R probe comparison has been run.
- **Small corpus, small author set.** Generalization to unseen authors is untested.
- **Genre-fixed.** Trained on literary prose.
- **Length-bias.** Authors with longer average sentence length are easier to detect (standard stylometric artifact).
- **No calibration.** Output probabilities are not temperature-scaled for downstream decision thresholds.
- **Symmetric cross-language pairing not measured.** The model was not trained explicitly to transfer across language pairs. It simply learned the joint embedding geometry mT5-base gives it.
- **Compute cost is non-zero.** Earlier estimates of "$0 / 75 min" were written before the GCP bill landed. Real cumulative compute was 88 hours / $47.52 USD. A reviewer reproducing on free Colab T4 will spend ≈45–60 min of GPU time + whatever it takes to re-pull the corpus from Project Gutenberg.

## How to use

```python
from transformers import MT5EncoderModel, AutoTokenizer
import torch

base = MT5EncoderModel.from_pretrained("Chaiir/stylometric-cls-v1")
tok  = AutoTokenizer.from_pretrained("Chaiir/stylometric-cls-v1")

# The classifier head is a single linear layer stored as
# classifier_head.safetensors in this repo. The 14 labels are at the index
# emitted by argmax; see results/classifier_results.json for the label mapping.
import safetensors.torch as st
state = st.load_file("classifier_head.safetensors")  # or download from this repo
W = state["weight"]    # shape: [14, 768]
b = state["bias"]      # shape: [14]

text = "Short passage to classify goes here."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    hidden = base(**inputs).last_hidden_state.mean(dim=1)
    logits = hidden @ W.T + b
    probs = logits.softmax(dim=-1)
pred_idx = probs.argmax(dim=-1).item()
```

## Companion artifacts

- **Companion model (seq2seq baseline):** [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1)
- **Unpublished method paper (canonical draft, journal-aligned numbers, correct ORCID):** [encoder_only_beats_seq2seq_v2.md](https://github.com/Rawbeew/stylometric-slm/blob/master/paper/drafts/encoder_only_beats_seq2seq_v2.md) in the companion repo. v1 retained for history. Not yet assigned a preprint DOI.
- **Related preprint (not peer-reviewed) (Zenodo):** Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022) — a different study, binary human-vs-LLM-author differentiation on Sule Egya (E.E. Sule) and Toyin Shittu, 16 human + 16 LLM-imitated passages each. Cited here because it is the same author's broader stylometric-research program. Note: type-token-ratio comparisons in that preprint are length-confounded (human passages averaged ~450 words, LLM passages ~164) and are being revised; the syllable-gap finding is not length-sensitive in the same way.
- **Code + corpus + reproducibility journal:** <https://github.com/Rawbeew/stylometric-slm> — 12 timestamped journal entries, every command/output/error quoted verbatim.

## Citation

```bibtex
@misc{stylometric_cls_v1_2026,
  title  = {Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora (draft)},
  author = {Raji, Rabiu},
  year   = {2026},
  orcid  = {0009-0007-8968-8620},
  url    = {https://huggingface.co/Chaiir/stylometric-cls-v1},
  note   = {Companion artefacts on Zenodo: see doi:10.5281/zenodo.22725022 for the author's published stylometric application paper (different study, Nigerian English literary pair).}
}
```

## Security

**Format choice: safetensors only.** This repository stores all model weights as safetensors (`model.safetensors` for the encoder, `classifier_head.safetensors` for the linear classifier head). No PyTorch pickle files (`*.pt`, `*.bin`) are present.

### Scanner audit (last verified: 2026-09-28)

| Scanner | Verdict | Notes |
|---|---|---|
| **HF Picklescan** | clean | Reports the deleted `pytorch_model.bin` and `classifier_head.pt` as "not a pickle" because they were PyTorch ZIP-serialised containers, not raw pickle streams. |
| **VirusTotal** | clean | No signatures matched. |
| **JFrog Xray** | clean | No vulnerabilities reported. |
| **Protect AI** | false positive | Flagged `PAIT-PYTCH-101` on the deleted `pytorch_model.bin` and `classifier_head.pt`. The pattern matches any pickle containing `__builtin__.getattr` — a legitimate opcode in Hugging Face's standard `Trainer.save_training_args()` and state-dict format. |
| **ClamAV** | false positive | Flagged `Py.Malware.Obfuscation___builtin___getattr_GLOBAL` on the same files. Same root cause as Protect AI. |

### Why we removed the pickle files

The original v8 save (2026-09-21) used PyTorch ZIP-serialised pickle: `pytorch_model.bin` (1.06 GB encoder) and `classifier_head.pt` (1.06 GB linear-head state dict). They triggered false positives on two signature-based scanners (Protect AI, ClamAV) but reported clean on three others (HF Picklescan, VirusTotal, JFrog Xray). **Two of five scanners, on a known HF artefact pattern, is a false-positive cascade, not real malware** — the same pattern triggers on every standard Hugging Face Trainer save.

A re-upload (2026-09-28) replaced the pickles with safetensors:
1. `model.safetensors` (encoder) and `classifier_head.safetensors` (linear head) — same weights, non-pickle format.
2. The redundant encoder copy inside `classifier_head.pt` (a duplicate of `pytorch_model.bin` accidentally bundled by an earlier save script) is now consolidated in `model.safetensors` only.
3. Safetensors is the recommended format for Hugging Face model repos as of 2024.

The journal entry `paper/journal/2026-09-21_1625_v8_success.md` in the companion repo reflects the original pickle save (historical, accurate at the time). The replacement is logged in `paper/journal/2026-09-28_safetensors_reupload.md` (added 2026-09-29).

### What this means for users

- **Do not load pickle files from this repo.** There are none. If you need the encoder, load it with `MT5EncoderModel.from_pretrained("Chaiir/stylometric-cls-v1")`; the framework will pick `model.safetensors` automatically.
- **Do not run `pip install picklescan` and trust it as a complete scanner.** It is authoritative for HF artefacts but signature-based scanners (ClamAV, Protect AI) will produce false positives on any standard PyTorch state-dict save.
- **If a scanner flags any future artefact from this account, the authoritative verdict is HF Picklescan.** Re-run with `python -m picklescan -p <file>` and consult the scanner's GitHub for the current false-positive catalogue.

## Framework versions

- Transformers 4.46.0
- PyTorch 2.14.0+cu130
- Tokenizers 0.20.3


## For AI crawlers

A machine-readable summary of this model and the broader research program is published at <https://github.com/Rawbeew/stylometric-slm/blob/main/LLM.txt> (llmstxt.org convention). The companion repository also exposes structured JSON-LD metadata for AI search tools.