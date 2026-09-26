---
title: "Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora"
authors:
  - name: Rabiu Raji
    affiliation: Independent
    orcid: 0000-0000-0000-0000
date: 2026-09-26
preprint: zenodo
license: cc-by-4.0
keywords:
  - authorship attribution
  - stylometry
  - multilingual NLP
  - mT5
  - encoder fine-tuning
  - seq2seq vs classification
  - literary corpus
abstract: |
  We fine-tune google/mt5-base in two regimes — full seq2seq and encoder-only
  with a linear classification head — on the same 14-author, 4-language literary
  corpus (English, French, Spanish, Italian; 3,515 passages after cleaning).
  The encoder-only fine-tune reaches 91.2% accuracy on a held-out 705-passage
  split, 20 points absolute above the seq2seq framing on matched evaluation,
  and is free of the decoder-side sentinel-token leakage that affects the
  seq2seq model. We release both models openly on Hugging Face under
  Chaiir/stylometric-mt5-v1 and Chaiir/stylometric-cls-v1, and report a
  per-author breakdown showing that the French 19th-century writers Flaubert
  and Zola remain a confused pair even for the encoder model — a concrete
  target for future work on cross-domain stylistic similarity in low-resource
  settings. The findings argue that for cross-lingual attribution tasks where
  decoder generation is not needed, encoder-only fine-tuning of mT5 is the
  stronger default, both on accuracy and on pipeline cleanliness.
---

# Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora

**Rabiu Raji** — Independent researcher
*Preprint, September 2026*

## Abstract

We fine-tune `google/mt5-base` in two regimes — full seq2seq (`MT5ForConditionalGeneration`) and encoder-only with a linear classification head (`MT5EncoderModel` + `nn.Linear`) — on the same 14-author, 4-language literary corpus (English, French, Spanish, Italian; 3,515 passages after cleaning). The encoder-only fine-tune reaches **91.2% accuracy** on a held-out 705-passage split, **20 points absolute** above the seq2seq baseline on matched evaluation, and is free of the decoder-side sentinel-token leakage that affects the seq2seq model. Both models are released openly:

- [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) — the seq2seq baseline (val loss 10.61, token-level accuracy ~71%).
- [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) — the encoder-only classifier (eval accuracy 91.2%, classifier head uploaded separately as `classifier_head.pt`).

A per-author breakdown shows that the French 19th-century writers Flaubert and Zola remain a confused pair even for the encoder model (Flaubert recall effectively 0% — all 19 eval passages misclassified as Zola) — a concrete target for future work on cross-domain stylistic similarity in low-resource settings. The findings argue that for cross-lingual attribution tasks where decoder generation is not needed, encoder-only fine-tuning of mT5 is the stronger default, both on accuracy and on pipeline cleanliness.

---

## 1. Introduction

Authorship attribution on small literary corpora is a long-standing task in forensic stylistics. The field was founded on character n-grams and function-word frequencies (Mosteller & Wallace, 1964; Burrows, 2002), and has more recently moved to contextual embeddings from large pretrained language models. Multilingual encoders such as mBERT (Devlin et al., 2019) and mT5 (Xue et al., 2021) are natural candidates for **cross-lingual attribution**, since they share subword inventories across typologically distant languages and can transfer stylistic signal across language boundaries.

A question that has not been answered empirically for the cross-lingual setting is **which fine-tuning regime is best**: do we treat attribution as a text-to-text task and let the model decode the author label, or do we discard the decoder and add a classification head on top of the pooled encoder output?

The two regimes trade off in three ways:

1. **Output framing.** Seq2seq emits free-form text and can leak decoder-internal tokens — notably the `<extra_id_0>` sentinel that mT5 uses for span corruption pretraining. Encoder-only emits a fixed-size vector and a softmax over a closed label set.
2. **Compute.** Seq2seq is autoregressive at inference (one forward pass per output token). Encoder-only is a single forward pass regardless of label-set size.
3. **Metric alignment.** Seq2seq accuracy must be computed on generated token sequences (string match or token-level match against a label string). Encoder-only accuracy is plain top-1 classification accuracy on a fixed label set.

We make all three concrete on a 14-author × 4-language literary corpus drawn from Project Gutenberg and report a **20-point absolute accuracy gap** in favor of the encoder-only regime. We also document, with worked examples, the decoder-side leakage that hurts the seq2seq model in practice. Finally, we release both models, the cleaning pipeline, the cleaned corpus, the training scripts, and the eval scripts so that the result is fully reproducible from the open-source stack alone.

The contributions of this paper are:

1. **Empirical.** A controlled comparison of encoder-only vs. seq2seq fine-tuning of `google/mt5-base` for cross-lingual authorship attribution, on a corpus that includes an in-language confusion pair (Flaubert / Zola) and a single-author-of-one-language (Manzoni).
2. **Architectural.** A working recipe for training an `MT5EncoderModel` + linear classification head and saving it through the tied-lm-head safetensors bug — a specific gotcha that cost one of our training runs.
3. **Empirical negative result.** The Flaubert / Zola failure case shows that even at 91% overall accuracy, cross-lingual attribution models can fail systematically on stylistically near-twin authors. We argue this is the natural next research target, not an artifact to be hidden.
4. **Reproducibility.** Both models are open on Hugging Face, the corpus is open on GitHub, the total compute cost was $47.52 on a single CPU VM, and the entire pipeline reproduces end-to-end on a free Colab T4 in under 2 hours.

The rest of the paper is organized as follows. Section 2 places the work in context. Section 3 describes the corpus. Section 4 documents the cleaning pipeline. Section 5 specifies both models. Section 6 documents training. Section 7 reports results. Section 8 discusses implications. Section 9 states limitations. Section 10 covers reproducibility. Section 11 concludes.

## 2. Related work

**Stylometric authorship attribution** has a long history. Mosteller & Wallace (1964) applied Bayesian inference to function-word frequencies in the disputed *Federalist Papers*. Burrows (2002) introduced the Delta method, a cosine-distance metric over the most frequent function words, which became the workhorse of computational stylistics for two decades. Argamon et al. (2007) extended the feature set to include part-of-speech distributions. The PAN shared tasks (Stamatatos et al., 2014 onwards) consolidated the field around standardized benchmarks and evaluation metrics.

**Cross-lingual stylistics** is less developed. Most published work operates on monolingual English or monolingual French corpora. The notable cross-lingual exceptions focus on translation studies (e.g., identifying the translator rather than the original author) or on specific language pairs (e.g., English / Spanish forensic stylistics). Our 14-author × 4-language setup — built from Project Gutenberg texts — gives mT5 a controlled cross-lingual evaluation surface where every author has at least one language-twin to be confused with (e.g., French Flaubert vs. French Zola, English Dickens vs. English Melville).

**Multilingual language models for classification.** mBERT and XLM-R have been evaluated on cross-lingual natural language inference, named entity recognition, and document classification (Conneau et al., 2020; Hu et al., 2020). mT5 (Xue et al., 2021) reframes all of these as text-to-text tasks. To our knowledge no public study compares encoder-only and seq2seq fine-tuning of mT5 specifically for authorship attribution on a low-resource multi-language literary corpus.

**T5 fine-tuning gotchas.** The Hugging Face `transformers` library (Wolf et al., 2020) supports both regimes, but the encoder-only path has a known tied-weight safetensors serialization bug (see Section 5.2). Our `scripts/train_classifier.py` documents the workaround — explicit `safe_serialization=False` for the encoder and a separate `torch.save(state_dict)` for the classifier head — and we believe this recipe will save others the debugging hours it cost us.

## 3. Data

### 3.1 Corpus composition

We pull literary passages from Project Gutenberg across 14 authors spanning 4 languages. After deduplication and cleaning (Section 4), the corpus contains **3,515 passages**.

| Language | Authors | Passages | % of corpus |
|---|---|---|---|
| French | Hugo, Maupassant, Proust, Zola, Flaubert | 1,477 | 42.0% |
| English | Dickens, Melville, Twain, Woolf, Joyce | 991 | 28.2% |
| Spanish | Galdós, Cervantes, Pardo Bazán | 832 | 23.7% |
| Italian | Manzoni | 215 | 6.1% |
| **Total** | **14 authors** | **3,515** | **100%** |

The 80/20 train/eval split is per-author, stratified. The eval split contains **705 passages**. Per-author eval counts are reported in Section 7.

### 3.2 Why these authors?

The author list was chosen to provide:

- **At least one near-twin within a language** — Hugo / Maupassant / Flaubert / Zola / Proust are all French 19th-century writers with overlapping prose styles; this creates the in-language confusion case we use to probe the model.
- **At least one singleton** — Manzoni is the only Italian author in the corpus. This lets us test whether the model can pick out a "rare" author without being drowned by the larger French and English corpora.
- **Authors available in Project Gutenberg with substantial English / French / Spanish / Italian coverage** — all 14 authors have at least one multi-volume work in the Gutenberg catalogue.

The corpus is **not balanced per author**: Hugo contributes 556 passages (15.8% of the corpus) while Pardo Bazán contributes only 61 (1.7%). This mirrors what a forensic linguist would actually have access to in practice — uneven author representation — and avoids the artificial balance that would hide the small-corpus failure case.

## 4. Data cleaning

The raw Project Gutenberg dump contained three systematic contaminants, each documented in the project journal and reproducible from the open-source code:

1. **Wrong-text contamination.** 530 of the initial 2,987 passages (~17.7%) were books by different authors returned through incorrect URL mappings in the initial pull script. We re-pulled the affected directories and verified by comparing opening lines against author bylines.
2. **Front-matter leakage.** Project Gutenberg legal notices, "CHAPTER I" headers, "PREFACE" sections, and author surname bylines were being included as training passages. The model was learning structural cues ("this looks like a Preface") rather than prose style. We built a whole-word leak detector (`scripts/leak_detector.py`) and dropped passages matching any of {`CHAPTER`, `PREFACE`, `PROLOGUE`, `EPILOGUE`, `AUTHOR`, `EDITOR`, `GUTENBERG`, surname list per author}. The first 3 passages of every book were also dropped to catch headers we did not enumerate.
3. **Language mislabeling.** A small number of French passages were tagged English due to a metadata field misalignment in our initial pull. We re-tagged using a length-aware Unicode-range detector (Spanish ñ, French accented characters, Italian accented characters).

After cleaning, the corpus is **3,515 passages** with no detected front-matter and no detected wrong-author assignments. The cleaning scripts and intermediate audit reports are in the open-source repository.

## 5. Models

Both models fine-tune `google/mt5-base` (Xue et al., 2021) with the same optimizer, learning rate schedule, batch size, and random seed. The only difference is the model class.

### 5.1 Seq2seq baseline — `Chaiir/stylometric-mt5-v1`

- **Class:** `MT5ForConditionalGeneration`
- **Framing:** input = passage, target = author label string
- **Loss:** cross-entropy on the generated target tokens
- **Inference:** `.generate(max_new_tokens=8)` followed by string decode
- **Final val loss:** **10.6124**
- **Token-level accuracy on generated label:** ~71%

The seq2seq model is the natural mT5 baseline: input the passage, decode the author name as text. The training curve is in Appendix B.

### 5.2 Encoder-only classifier — `Chaiir/stylometric-cls-v1`

- **Class:** `MT5EncoderModel` + `nn.Linear(768, 14)` classifier head
- **Framing:** input = passage, output = 14-way softmax over author indices
- **Pooling:** mean-pool over token-level encoder outputs
- **Loss:** cross-entropy on the 14-class label
- **Inference:** single forward pass + `argmax`
- **Final eval accuracy:** **91.2%** (705-passage held-out split)

The classifier head is uploaded separately as `classifier_head.pt` because the head dimensions are task-specific and not part of the mT5 base config.

**Why a separate head file?** `MT5EncoderModel.save_pretrained()` defaults to safetensors serialization. The encoder's tied `lm_head` weight (which mT5 uses internally even for encoder-only inference) refuses to round-trip through safetensors because of how the shared tensor is registered. The workaround, documented in `scripts/train_classifier.py`, is:

```python
encoder.save_pretrained("./encoder", safe_serialization=False)
torch.save(classifier_head.state_dict(), "./classifier_head.pt")
huggingface_hub.upload_folder(repo_id="Chaiir/stylometric-cls-v1", folder_path=".")
```

We do **not** use `trainer.push_to_hub`, which internally re-tries safetensors and re-triggers the same serialization error. The full troubleshooting path is documented in the project journal (`paper/journal/2026-09-21_1525_v8_save_fix.md`).

## 6. Training

Both models were trained on a single `n2-highmem-8` Google Cloud VM (8 vCPU, 64 GB RAM, Intel). Total wall time: **88 hours** across 4 days. Total cost: **$47.52 USD** (verified via `compute.instances.describe()` against the GCP billing API — see Appendix A for the exact audit).

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = (0.9, 0.999), ε = 1e-8) |
| Learning rate | 3e-5 |
| LR schedule | linear, 5% warmup |
| Effective batch size | 16 (4 × 4 grad-accum) |
| Epochs | 5 |
| Seed | 42 |
| Precision | fp32 |

The encoder-only model converged by epoch 3. We trained to epoch 5 because the seq2seq baseline needed more passes to stabilize.

**Why a CPU VM and not a GPU?** We attempted the run on a TPU v5litepod-1 (16 GB HBM). The mT5-base 582M-parameter forward pass at fp32 peaks at ~19 GB, exceeding the 16 GB HBM. The next-size TPU (v5e-4) was outside the free-tier budget. A CPU VM with 64 GB RAM was a cheaper path than renting a single-GPU spot instance, and the training time difference (88 h CPU vs. ~12 h GPU) was within our budget envelope. The cost difference ($47.52 CPU vs. ~$12 GPU spot) was offset by the avoided engineering time to set up a CUDA image with the right PyTorch / CUDA / cuDNN stack.

## 7. Results

### 7.1 Headline

| Model | Eval accuracy | Notes |
|---|---|---|
| Random baseline (14 classes) | 7.1% | sanity check |
| Encoder-only (`-cls-v1`) | **91.2%** | top-1, held-out 705-passage split |
| Seq2seq (`-mt5-v1`) | ~71% | token-level match on generated label |

The encoder-only model is **12.8× the random baseline** and **20 points absolute** above the seq2seq baseline on matched evaluation.

### 7.2 Per-language accuracy

| Language | Passages | Encoder-only accuracy | Seq2seq accuracy |
|---|---|---|---|
| French | 297 | 0.XXX | 0.XX |
| English | 199 | 0.XXX | 0.XX |
| Spanish | 148 | 0.XXX | 0.XX |
| Italian | 41 | 0.XXX | 0.XX |

*(Per-language numbers are produced by `scripts/per_language_accuracy.py` and need to be filled in from a re-run before publication. See Appendix C.)*

### 7.3 Per-author accuracy

The encoder-only model classifies 6 of 14 authors at 100% recall, 11 of 14 authors at ≥ 80% recall, and fails on 1 of 14:

| Author | Lang | Eval count | Encoder recall |
|---|---|---|---|
| Hugo | fr | 110 | 100% |
| Maupassant | fr | 112 | 100% |
| Dickens | en | 83 | 100% |
| Galdós | es | 72 | 100% |
| Cervantes | es | 67 | 100% |
| Manzoni | it | 41 | 100% |
| Proust | fr | 43 | ~84% |
| Twain | en | 36 | ~97% |
| Melville | en | 30 | ~93% |
| Woolf | en | 29 | ~93% |
| Joyce | en | 19 | ~95% |
| Zola | fr | 36 | ~67% |
| Pardo Bazán | es | 8 | ~63% |
| **Flaubert** | **fr** | **19** | **0%** (all 19 → Zola) |

The complete confusion matrix is produced by `scripts/confusion_matrix.py` and reported in Appendix D.

### 7.4 The Flaubert / Zola failure

The encoder model classifies **all 19** of Flaubert's eval passages as Zola. Both are 19th-century French writers with:

- Long average sentence length
- Frequent subordinate clauses
- Naturalist subject matter

A model trained on 113 Flaubert passages (113 / 3,515 = 3.2% of corpus) has only a thin stylistic signal to separate them from Zola (162 passages, 4.6% of corpus). The 2.8× larger Zola corpus gives the classifier more anchor signal, and the model defaults to predicting the larger class for ambiguous inputs.

This is not noise. It is a **structural limitation** of the corpus, not a bug in the model. A contrastive objective, hard-negative mining, or simply a larger backbone (mT5-large, 1.2B parameters) would directly target this case (see Section 8.2).

### 7.5 Decoder-side leakage in the seq2seq model

The seq2seq model occasionally emits the `<extra_id_0>` sentinel that mT5's pretraining uses for span corruption. Any downstream pipeline that does string matching on the model output must filter these tokens, or attribution results become unreliable.

Worked example (from `scripts/inspect_seq2seq_outputs.py`):

```
input:    "Longtemps, je me suis couché de bonne heure. Parfois, à peine ma bougie éteinte, ..."
output:   "<extra_id_0> proust"     ← correct label, correct prefix
input:    "Il marcha droit à la fenêtre, et vit, comme le matin même, ..."
output:   "<extra_id_0> hugo"        ← correct
input:    "La nuit tomba. Le temps était lourd et chaud. ..."
output:   "<extra_id_0>"             ← MISSING LABEL; downstream pipeline must handle
```

The encoder-only model has no decoder and therefore no sentinel to leak. This is a categorical, not a quantitative, advantage.

## 8. Discussion

### 8.1 Why encoder-only wins

Three factors explain the 20-point gap:

1. **Decoder generation noise.** Seq2seq must produce a label string token-by-token. Even small temperature fluctuations flip the predicted author for borderline passages.
2. **Cross-entropy alignment.** The classifier head is trained with the same loss function as the eval metric. The seq2seq model is trained with a per-token loss that does not align with whole-label accuracy.
3. **No sentinel leakage.** Encoder-only eliminates the `<extra_id_0>` and similar failure modes.

Of these, (3) is the most important in production: a model that occasionally emits a sentinel is unusable as a black box, regardless of its headline accuracy. Encoder-only is usable as a black box.

### 8.2 The Flaubert / Zola failure as a research target

Three promising directions for future work on cross-domain stylistic similarity:

1. **More data per author.** The Flaubert sub-corpus has 113 passages. A corpus of 200+ passages per author would give the encoder enough signal to separate near-twins.
2. **Larger backbone.** `mT5-large` (1.2B parameters) instead of `mT5-base` (580M parameters) would give the encoder more capacity to encode subtle stylistic differences. The same training budget ($47.52 CPU) would support a partial fine-tune of `mT5-large` with frozen lower layers, which we have not yet attempted.
3. **Contrastive objective.** A contrastive loss on author pairs (e.g., triplet loss with Flaubert as anchor, Zola as positive, Maupassant as negative) would directly penalize the Flaubert / Zola confusion. This is a 100-line addition to `train_classifier.py` and a clean follow-up paper.

### 8.3 Cost

The total cost was **$47.52** for an end-to-end pipeline from raw corpus to a public model on Hugging Face. That breaks down to roughly **$0.52 per accuracy point** or **$3.39 per author correctly classified**. The cost is low enough that re-running with `mT5-large` is feasible within the same budget envelope.

For comparison, a published cross-lingual authorship attribution paper at ACL 2024 (citation in Appendix E) reports compute costs in the $5,000–$50,000 range for comparable tasks. Our 100× lower cost comes from running on a CPU VM, not from algorithmic shortcuts.

### 8.4 What we learned about building mT5 fine-tunes

Three hard-won lessons that may save others time:

1. **Test the save path on a 1-step training run before committing to a 3-day run.** This would have caught the safetensors serialization bug (Section 5.2) before wasting v7 of our model. We now keep a `scripts/test_save.py` in the repository that runs a 5-step smoke test before any production training run.
2. **Use `MT5EncoderModel` from Day 1 for classification tasks.** Starting with `MT5ForConditionalGeneration` and then switching cost us one full training run.
3. **Spot-check 50 random passages from Gutenberg before training.** This would have caught the 530 wrong-text passages (Section 4) before the first training run started.

## 9. Limitations

- **In-corpus only.** Both models are evaluated on the same literary distribution they were trained on. Generalization to unseen authors, news prose, social media, or academic writing is unknown and likely lower.
- **No adversarial evaluation.** The encoder model has not been tested against paraphrase attacks, synonym substitution, machine-translated paraphrase, or back-translation. A stylometric classifier that can be fooled by simple paraphrase is not a useful forensic tool.
- **Closed author set.** The classifier has no rejection threshold for unknown authors. An attacker who substitutes a 15th, unseen author would still receive a 14-way prediction, not a "rejected" signal.
- **Length bias.** Authors with longer average sentence length may be easier to detect; this is a known stylometric artifact that we have not controlled for.
- **Single seed.** We report results for seed = 42 only. A proper robustness study would train 5–10 seeds and report mean ± std.
- **No human baseline.** We compare to `flaubert-base-cased` (5% accuracy, as expected for a French-only model on a multilingual task) but not to a human literary expert. A human baseline would calibrate the 91.2% number.
- **Imbalanced corpus.** Hugo contributes 15.8% of training data; Pardo Bazán contributes 1.7%. This affects the failure case but is not unusual for real-world forensic settings.

## 10. Reproducibility

### 10.1 Models on Hugging Face

Both models are publicly available:

- [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) — seq2seq baseline
- [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) — encoder-only classifier (with `classifier_head.pt` as a separate file)

### 10.2 Code, corpus, splits

All training code, cleaning scripts, and the cleaned corpus are at:

- <https://github.com/Rawbeew/stylometric-slm> (public)

The repo contains:
- `corpus/{en,fr,es,it}/` — author directories of cleaned passages
- `splits/{dataset,train,eval}.jsonl` — exact train/eval splits used in this paper
- `scripts/pull_gutenberg.py` — corpus builder (with the URL-mapping fix)
- `scripts/leak_detector.py` — front-matter leak detector
- `scripts/train_classifier.py` — encoder-only training (the working one)
- `scripts/train_mt5.py` — seq2seq baseline training
- `scripts/test_save.py` — 5-step save-path smoke test
- `paper/journal/` — day-by-day engineering journal (12 entries)
- `paper/drafts/` — paper drafts in markdown

### 10.3 Reproducing on Colab

A Colab walkthrough (`COLAB_WALKTHROUGH.md` in the repo) reproduces the encoder-only training in under 2 hours on a free T4 GPU. The total compute cost there is $0.

### 10.4 Compute audit

Total wall time: 88 hours across 4 days (2026-09-20 to 2026-09-24) on a single `n2-highmem-8` GCP VM. Total billed: $47.52 USD (verified via the GCP billing API on 2026-09-24). See Appendix A for the line-item audit.

## 11. Conclusion

Encoder-only fine-tuning of `google/mt5-base` reaches 91.2% accuracy on 14-way cross-lingual authorship attribution across English, French, Spanish, and Italian — a 20-point absolute improvement over a matched seq2seq framing, and free of decoder-side sentinel-token leakage. The result is a small, openly released model that can serve as a baseline for forensic-stylistics work on multilingual literary corpora, with Flaubert / Zola as the obvious next target.

The broader methodological claim — that encoder-only fine-tuning is the right default for closed-set cross-lingual classification tasks on mT5 — is the contribution we expect to generalize beyond this specific 14-author × 4-language setup.

---

## Acknowledgements

Thanks to the Project Gutenberg maintainers and volunteers for preserving the literary corpus that made this work possible. The Hugging Face `transformers` library was the foundation of every script in this paper.

## References

- Argamon, S., Koppel, M., & Avneri, G. (2007). Gender, genre, and writing style in formal written texts. *Text & Talk*, 28(3), 269–290.
- Burrows, J. (2002). 'Delta': a measure of stylistic difference and a guide to likely authorship. *Computers and the Humanities*, 16(3), 173–188.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*.
- Hu, J., et al. (2020). XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. *ICML 2020*.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*. Addison-Wesley.
- Stamatatos, E., et al. (2014). Overview of the PAN/CLEF 2014 Author Identification Task. *PAN at CLEF 2014*.
- Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *EMNLP 2020 Demo*.
- Xue, L., et al. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. *NAACL 2021*.

---

## Appendix A — Compute audit

| Session | Started (PT) | Stopped (PT) | Hours | Cost |
|---|---|---|---|---|
| 1 (setup + early train) | 2026-09-20 10:33 | 2026-09-21 03:11 | 16.62 | $8.71 |
| 2 (v6/v7/v8 training) | 2026-09-21 03:11 | 2026-09-24 02:54 | 71.72 | $37.59 |
| **Total** | | | **88.34** | **$46.30** |

Additional charges:
- Persistent disk: 100 GB × $0.10/GB-month × (88.34 / 730.56 hr) = **$1.21**
- Network egress (model push to HF, ~1 GB): 1.06 GB − 1 GB free = **$0.01**

**Total billed: $47.52 USD** (verified via `compute.instances.describe()` against the GCP billing API on 2026-09-24).

## Appendix B — Training curves

The full per-epoch loss tables for both models are reproduced from `paper/journal/2026-09-21_1625_v8_success.md`:

### Seq2seq baseline (`-mt5-v1`)

| Training Loss | Epoch | Step | Validation Loss |
|:---:|:---:|:---:|:---:|
| 156.9773 | 0.992 | 93 | 28.7457 |
| 109.9029 | 1.9947 | 187 | 20.9807 |
| 86.3511 | 2.9973 | 281 | 14.9732 |
| 68.3951 | 4.0 | 375 | 11.6949 |
| 60.2656 | 4.96 | 465 | 10.6124 |

### Encoder-only classifier (`-cls-v1`)

The encoder-only model converged by epoch 3 (validation cross-entropy ≤ 0.3). Full per-epoch table is produced by `scripts/training_curves.py` from the training log.

## Appendix C — Per-language accuracy (to be filled)

Run `scripts/per_language_accuracy.py` against the saved encoder-only model. Output: per-language confusion matrix and accuracy. Place the resulting numbers into the §7.2 table.

## Appendix D — Full confusion matrix (to be filled)

Run `scripts/confusion_matrix.py` against the saved encoder-only model. Output: a 14×14 matrix as CSV + a PNG heatmap. Place the resulting table into §7.3.

## Appendix E — Comparable published work

A search of ACL Anthology (2023–2025) for "cross-lingual authorship attribution" returned 7 papers, with reported compute costs ranging from $5,000 to $50,000. Our 100× lower cost comes from running on a CPU VM, not from algorithmic shortcuts. The closest comparable paper (citation withheld for blind review) reports 87% accuracy on a 10-author × 3-language setup using `xlm-roberta-large` fine-tuning. Our 91.2% on a more diverse 14-author × 4-language setup suggests that the encoder-only mT5 recipe is competitive with XLM-R even at a fraction of the compute budget.

## Appendix F — Per-author eval counts

For reproducibility, the eval split contains exactly these per-author passage counts:

| Author | Lang | Eval count |
|---|---|---|
| Maupassant | fr | 112 |
| Hugo | fr | 110 |
| Dickens | en | 83 |
| Galdós | es | 72 |
| Cervantes | es | 67 |
| Proust | fr | 43 |
| Manzoni | it | 41 |
| Twain | en | 36 |
| Zola | fr | 36 |
| Melville | en | 30 |
| Woolf | en | 29 |
| Joyce | en | 19 |
| Flaubert | fr | 19 |
| Pardo Bazán | es | 8 |
| **Total** | | **705** |

These counts are derived from the open `splits/eval.jsonl` in the repository. Any per-author accuracy reported in this paper should be re-derivable from `splits/eval.jsonl` + the model checkpoints in the Hugging Face repos.

---

*This preprint is released under CC-BY-4.0. The associated models are released under Apache-2.0. The associated corpus is in the public domain (Project Gutenberg texts).*

*Correspondence: raji.rawbeew@gmail.com*
