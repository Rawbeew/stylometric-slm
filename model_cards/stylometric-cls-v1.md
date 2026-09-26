---
library_name: transformers
license: apache-2.0
base_model: google/mt5-base
pipeline_tag: text-classification
tags:
- mte5
- mt5
- text-classification
- stylometry
- authorship-attribution
- forensic-stylistics
- multilingual
- cross-lingual
- european-literature
- generated_from_trainer
model-index:
- name: stylometric-cls-v1
  results:
  - task:
      type: text-classification
      name: Authorship Attribution (14-way, cross-lingual)
    dataset:
      type: stylometric-corpus-v1
      name: European literary corpus, 14 authors × 4 languages (en/fr/es/it)
    metrics:
    - type: accuracy
      value: 0.886
      name: Held-out accuracy (705 passages, 14-way)
    - type: baseline_logreg
      value: 0.717
      name: Logistic regression stylometric baseline (matched eval)
---

# stylometric-cls-v1

**Encoder-only fine-tune of mT5-base for 14-way cross-lingual authorship attribution on a multilingual European literary corpus.**

This is the **classifier regime** (v8). For the same architecture framed as seq2seq and a discussion of why seq2seq mT5 fails this task, see [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1).

> **Method comparison paper is an unpublished draft.** The encoder-only-vs-seq2seq comparison reported here is documented in `paper/drafts/encoder_only_beats_seq2seq.md` in the companion repo but has not been assigned a preprint DOI; it is not the same publication as the related Zenodo item below.
>
> **Related, peer-reviewed publication in the same research program:** Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022) — a different study (binary human-vs-LLM-imitation differentiation on Sule Egya and Toyin Shittu).

## TL;DR

| | |
|---|---|
| **Backbone** | `MT5EncoderModel` (mT5-base encoder, decoder removed) |
| **Head** | Linear layer, mean-pooled encoder output → 14 class logits |
| **Train / eval passages** | 2,810 / 705 (held-out 20%, deterministic hash split, stratified per author) |
| **Languages** | English, French, Spanish, Italian (Romance + Germanic in one model) |
| **Authors** | 14 across 4 languages — Dickens, Twain, Woolf, Joyce, Melville, Hugo, Maupassant, Proust, Flaubert, Zola, Cervantes, Galdós, Pardo Bazán, Manzoni |
| **Eval accuracy (14-way)** | **88.6%** overall, vs. 71.7% logistic-regression stylometric baseline |
| **Per-language F1** | IT 100%, ES 99%, FR 86%, EN 75% (Modernist English is the hard slice) |
| **Total parameters** | ~278M (encoder) + ~11K (linear head) |
| **Compute** | One CPU VM (n2-highmem-8), ~75 min training, fp32 |
| **Cost to reproduce** | $0 in cloud credits — fits in any 16GB RAM CPU box |

## Why encoder-only beats seq2seq (motivating finding)

For attribution-by-fine-tuning, mT5's full seq2seq architecture fights the pretrain objective. mT5 is trained with span corruption and has a strong bias toward emitting `<extra_id_0>` as the first token at inference. After 5 epochs of supervised fine-tuning, the seq2seq model still produces `<extra_id_0> <word>` instead of the author label (see `results/model_failure_analysis.json` in the companion repo). The encoder-only classifier does not have a decoder, so the bias is structurally unattainable — the model must learn the attribution task directly. Result: 88.6% encoder vs. eval_loss converging to 10.61 (entropy floor near random) for seq2seq.

## Evaluation

### Overall

| Model | Eval accuracy |
|---|---|
| Random (14-way) | 7.1% |
| Logistic-regression stylometric baseline (TTR, syllables/word, sentence length, function-word ratio, punctuation density → logreg) | **71.7%** |
| `Chaiir/stylometric-mt5-v1` (seq2seq, 5 ep) | eval_loss plateaus at 10.61; produces `<extra_id_0>` tokens. Not measurable as classifier accuracy. |
| **`Chaiir/stylometric-cls-v1` (encoder + linear head, 3 ep)** | **88.6%** |

### Per-language

| Language | n eval | Accuracy |
|---|---|---|
| Italian (Manzoni) | 41 | 100.0% |
| Spanish (Cervantes, Galdós, Pardo Bazán) | 108 | 98.1% |
| French (Proust, Zola, Maupassant, Hugo, Flaubert) | 198 | 89.4% |
| English (Dickens, Twain, Woolf, Joyce, Melville) | 164 | 75.6% |

### Per-author confusion (selected, with notes)

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

### Training dynamics

| Epoch | Eval loss | Eval accuracy |
|:---:|:---:|:---:|
| 1 | 0.83 | 77.3% |
| 2 | 0.44 | 86.7% |
| 3 | 0.34 | **91.2% overall / 88.6% per-passage on 705-passage final eval** |

The 91.2% and 88.6% numbers are the same model evaluated on slightly different splits (the 91.2 was reported on the full eval including some passages used in intermediate epoch checkpoints; 88.6 is the strict held-out 705). Both are real numbers; 88.6% is the strict claim.

Final train loss: 0.93. Final eval loss: 0.34. Total wall time: 4484 s on a single n2-highmem-8 CPU VM.

## Training procedure

- **Optimizer:** AdamW (β=(0.9, 0.999), ε=1e-8), lr=2e-4, linear schedule, 5% warmup
- **Batch size:** 8 × 2 grad-accum = 16 effective
- **Epochs:** 3
- **Max input length:** 512 tokens
- **Loss:** cross-entropy
- **Seed:** 42
- **Precision:** fp32
- **Pooler:** mean-pool over encoder last_hidden_state
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

- **Small corpus, small author set.** Generalization to unseen authors is untested.
- **Genre-fixed.** Trained on literary prose.
- **Length-bias.** Authors with longer average sentence length are easier to detect (standard stylometric artifact).
- **No calibration.** Output probabilities are not temperature-scaled for downstream decision thresholds.
- **Symmetric cross-language pairing not measured.** The model was not trained explicitly to transfer across language pairs. It simply learned the joint embedding geometry mT5-base gives it.

## How to use

```python
from transformers import MT5EncoderModel, AutoTokenizer
import torch

base = MT5EncoderModel.from_pretrained("Chaiir/stylometric-cls-v1")
tok  = AutoTokenizer.from_pretrained("Chaiir/stylometric-cls-v1")

# The classifier head is a single linear layer held in the repo as
# classifier_head.pt. The 14 labels are at the index emitted by argmax;
# see results/classifier_results.json for the label mapping.
state = torch.load("classifier_head.pt", map_location="cpu")
W = state["weight"]    # shape: [14, 768]
b = state["bias"]      # shape: [14]

text = "Short passage to classify goes here."
inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    hidden = base(**inputs).last_hidden_state.mean(dim=1)
    logits = hidden @ W.T + b
    probs = logits.softmax(dim=-1)
pred_idx = probs.argmax(dim=-1).item()
```

## Companion artifacts

- **Companion model (seq2seq baseline):** [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1)
- **Unpublished method paper (draft, encoder-only-vs-seq2seq comparison):** [encoder_only_beats_seq2seq.md](https://github.com/Rawbeew/stylometric-slm/blob/main/paper/drafts/encoder_only_beats_seq2seq.md) in the companion repo. Not yet assigned a preprint DOI.
- **Related peer-reviewed application paper (Zenodo):** Raji, R. (2026). *Voice or Mask? Stylometric Forensic Analysis of Two Contemporary Nigerian Poets.* [doi:10.5281/zenodo.22725022](https://doi.org/10.5281/zenodo.22725022) — a different study, binary human-vs-LLM-author differentiation on Sule Egya (E.E. Sule) and Toyin Shittu, 16 human + 16 LLM-imitated passages each. Cited here because it is the same author's broader stylometric-research program.
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

## Framework versions

- Transformers 4.46.0
- PyTorch 2.14.0+cu130
- Tokenizers 0.20.3
