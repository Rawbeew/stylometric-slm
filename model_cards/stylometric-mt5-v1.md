---
library_name: transformers
license: apache-2.0
base_model: google/mt5-base
pipeline_tag: text2text-generation
tags:
- mte5
- mt5
- text2text-generation
- seq2seq
- stylometry
- authorship-attribution
- forensic-stylistics
- multilingual
- span-corruption-bias
- negative-result
model-index:
- name: stylometric-mt5-v1
  results:
  - task:
      type: text2text-generation
      name: Authorship Attribution (seq2seq framing)
    dataset:
      type: stylometric-corpus-v1
      name: European literary corpus, 14 authors × 4 languages (en/fr/es/it)
    metrics:
    - type: eval_loss
      value: 10.61
      name: Final eval loss after 5 epochs (random baseline 2.64)
    - type: note
      value: Decoder emits <extra_id_0> sentinel tokens; classifier accuracy not measurable. See cls-v1 for the encoder-only architecture that does not have this defect.
---

# stylometric-mt5-v1

**Seq2seq fine-tune of mT5-base for authorship attribution framed as text-to-text generation.** Released as a **negative-result companion** to [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1), the encoder-only classifier for the same task that reaches 88.6% eval accuracy.

This model is a research artefact. The headline finding is **that the seq2seq regime fails this task** and why.

> **Pair released alongside an unpublished method comparison paper (draft only).**
> Raji, R. (2026, working draft). *Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora.* Draft at `paper/drafts/encoder_only_beats_seq2seq.md` in the companion repo. Not yet assigned a preprint DOI — will be submitted as a separate publication once stable.
>
> The author's completed Zenodo publication in the same research program is **Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022)**, which uses Burrows' Delta + classifier on a Nigerian English corpus (a different study).

## TL;DR

| | |
|---|---|
| **Backbone** | `MT5ForConditionalGeneration` (full encoder + decoder, mT5-base) |
| **Framing** | Input = passage, target = author label string |
| **Train / eval passages** | 2,810 / 705 (same split as cls-v1) |
| **Languages** | English, French, Spanish, Italian |
| **Authors** | 14 (same set as cls-v1) |
| **Eval loss after 5 epochs** | 10.61 (random baseline for 14-way = log(14) ≈ 2.64) |
| **Result** | **Decoder emits `<extra_id_0>` sentinels at inference.** Span-corruption pretrain bias not undone by supervised fine-tune. Classifier accuracy not measurable. |

## What went wrong (so a reviewer doesn't have to chase the journal)

mT5 is pretrained with a span-corruption objective. At inference, the decoder has a strong bias to emit `<extra_id_0>` as the first token — a sentinel it learned to expect as the prompt for fills. Five epochs of supervised fine-tuning on `("classify authorship: <passage>" → "<author>")` did not override the bias. Sample failures (`results/model_failure_analysis.json` in the companion repo):

```
input:  classify authorship: It was the best of times...
output: <extra_id_0> the
expected: dickens

input:  classify authorship: Il vivait miserable...
output: <extra_id_0>
expected: hugo

input:  classify authorship: Gervaise attendait...
output: <extra_id_0> class
expected: zola
```

The conclusion documented in the paper: when the task is a closed-set classification, an encoder-only fine-tune is the right architecture for this base model. The decoder adds parameters and pretrain bias without adding task-relevant capacity.

## Training procedure (for reproduction)

- **Optimizer:** AdamW (β=(0.9, 0.999), ε=1e-8), lr=3e-5, linear schedule, 5% warmup
- **Batch size:** 4 × 4 grad-accum = 16 effective
- **Epochs:** 5
- **Max input length:** 256 tokens (later extended to 512 in cls-v1; this regime stops earlier on convergence pattern)
- **Loss:** cross-entropy (label-token)
- **Precision:** fp32

## Evaluation

| Epoch | Eval loss |
|:---:|:---:|
| 1 | 28.75 |
| 2 | 20.98 |
| 3 | 14.97 |
| 4 | 11.69 |
| 5 | 10.61 |

The loss keeps falling because the model is learning the decoder-side sentinel-target mapping better over time, not because it is learning the author labels. Held-out eval token-accuracy converges to ~71% on the sentinel token; per-author accuracy on the actual names is 0%.

Random baseline (uniform 14-way) cross-entropy = log(14) ≈ 2.64. The model's eval loss is 4× random — a regime, not a measurement, and the only honest reading is that the classification head has been displaced by the sentinel token.

## Intended uses

- **Methodological comparison point** for encoder-only vs. seq2seq fine-tunes on the same multilingual base
- **Negative-result artefact** for studies of decoder-side leakage under span-corruption pretrain
- **Reuse as text encoder only.** Despite the failing decoder, the encoder subnetwork is still a fine-tuned multilingual encoder that can be detached and used like cls-v1.

## Out-of-scope uses

- **Anything requiring the decoder to produce author labels.** It will not.
- **Inference as a public-facing API.** Output is meaningless for the target task.

## How to use (as encoder-only, by detaching the decoder)

```python
from transformers import MT5EncoderModel, AutoTokenizer

# Use this model only via the encoder stack. Don't generate with it.
encoder = MT5EncoderModel.from_pretrained("Chaiir/stylometric-mt5-v1")
tok = AutoTokenizer.from_pretrained("Chaiir/stylometric-mt5-v1")

text = "..."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
hidden = encoder(**inputs).last_hidden_state
```

For a model that is fine-tuned end-to-end on this task, use [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1).

## Companion artifacts

- **Companion model (classifier that works):** [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) — 88.6% eval accuracy
- **Unpublished method paper (draft only):** `paper/drafts/encoder_only_beats_seq2seq.md` in the companion repo. Will be assigned a preprint DOI on submission.
- **Related published application paper (Zenodo):** Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022) — a different study (binary human-vs-LLM-imitation differentiation on Sule Egya and Toyin Shittu). Same research program, not this artefact.
- **Code + corpus + reproducibility journal:** <https://github.com/Rawbeew/stylometric-slm>
- **Failure mode samples + per-epoch loss table:** `results/model_failure_analysis.json`

## Citation

```bibtex
@misc{stylometric_mt5_v1_2026,
  title  = {Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora (draft)},
  author = {Raji, Rabiu},
  year   = {2026},
  orcid  = {0009-0007-8968-8620},
  url    = {https://huggingface.co/Chaiir/stylometric-mt5-v1},
  note   = {Negative-result release: seq2seq mT5 fails authorship attribution under span-corruption pretrain bias; companion to cls-v1. The draft comparison paper is not yet assigned a preprint DOI. The author's published Zenodo artefact in the same research program is doi:10.5281/zenodo.22725022 (Voice or Mask? — different study).}
}
```

## Framework versions

- Transformers 4.46.0
- PyTorch 2.14.0+cu130
- Tokenizers 0.20.3
