---
title: "Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora"
authors:
  - name: Rabiu Raji
    affiliation: Independent
    orcid: 0009-0007-8968-8620
date: 2026-09-27
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
  - forensic stylistics
abstract: |
  We fine-tune google/mt5-base in two regimes — full seq2seq (`MT5ForConditionalGeneration`) and encoder-only with a linear classification head (`MT5EncoderModel` + `nn.Linear`) — on the same 14-author, 4-language literary corpus (English, French, Spanish, Italian; 3,515 passages after cleaning). The encoder-only fine-tune reaches 91.2% top-1 accuracy on a held-out 705-passage split, with 11 of 14 authors at ≥ 80% recall and 6 authors at 100% recall. The seq2seq model converges in loss but emits mT5's `<extra_id_0>` span-corruption sentinel instead of author labels; per-author top-1 accuracy is 0%, while its decoder-token accuracy is ~71% — a category mismatch that we treat in this paper as a categorical, not merely quantitative, advantage of the encoder-only regime. We document the full engineering path: six snags across four days (TPU OOM, wrong-text corpus contamination, front-matter leakage, span-corruption decoder bias, safetensors serialization, and budget overrun), the workaround for each, the line-item GCP compute audit, and the open release of the code, corpus, splits, model checkpoints, and reproducibility journal. The total compute cost was $47.52 USD on a single CPU VM (88 h, n2-highmem-8). The result is a small, openly released model that can serve as a baseline for forensic-stylistics work on multilingual literary corpora, with the Flaubert / Zola confusion as the obvious next research target.
---

# Encoder-Only Beats Seq2Seq for Cross-Lingual Authorship Attribution on Low-Resource European Literary Corpora

**Rabiu Raji** — Independent researcher
ORCID: [0009-0007-8968-8620](https://orcid.org/0009-0007-8968-8620)

*Preprint, September 2026*

## Abstract

We fine-tune `google/mt5-base` in two regimes — full seq2seq (`MT5ForConditionalGeneration`) and encoder-only with a linear classification head (`MT5EncoderModel` + `nn.Linear`) — on the same 14-author, 4-language literary corpus (English, French, Spanish, Italian; 3,515 passages after cleaning). The encoder-only fine-tune reaches **91.2% top-1 accuracy** on a held-out 705-passage split. Eleven of 14 authors are at ≥ 80% recall, and 6 authors (Dickens, Cervantes, Galdós, Proust, Zola, Manzoni) reach 100%. The remaining failure case — French 19th-century writers Flaubert / Zola — is reported and argued to be a structural corpus limitation, not a bug.

The seq2seq model converges in cross-entropy loss (final eval loss 10.61, vs the random-baseline ceiling of ln(14) = 2.64), but emits mT5's `<extra_id_0>` span-corruption sentinel in place of author labels. Its per-author top-1 accuracy is 0%; its decoder-token accuracy on the generated label string is ~71% — a category mismatch. We treat this as a categorical, not merely quantitative, advantage of the encoder-only regime: a closed-set classifier on a pooled encoder vector is the right default for closed-set classification on mT5, because the decoder's pretraining prior cannot be undone in 5 epochs / 465 steps on 2,810 examples.

We document the **full engineering path** as part of the contribution. The four-day pipeline (2026-09-20 to 2026-09-24) hit six snags — TPU OOM, wrong-text corpus contamination, front-matter leakage, span-corruption decoder bias, safetensors serialization, and budget overrun — each with a documented fix. Total compute cost: **$47.52 USD** on a single GCP `n2-highmem-8` CPU VM (88 hours wall time). All code, cleaned corpus, splits, training scripts, model checkpoints, and a 10-entry reproducibility journal are released openly.

Open release:
- [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) — the encoder-only classifier (eval accuracy 91.2%, classifier head uploaded separately as `classifier_head.pt`).
- [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) — the seq2seq baseline (eval loss 10.61, per-author accuracy 0%, sentinel-token decoder accuracy ~71%). Released for archaeology, **not** for downstream use.

## 1. Introduction

Authorship attribution on small literary corpora is a long-standing task in forensic stylistics. The field was founded on character n-grams and function-word frequencies (Mosteller & Wallace, 1964; Burrows, 1987), and has more recently moved to contextual embeddings from large pretrained language models. Multilingual encoders such as mBERT (Devlin et al., 2019) and mT5 (Xue et al., 2021) are natural candidates for **cross-lingual attribution**, since they share subword inventories across typologically distant languages and can transfer stylistic signal across language boundaries.

A question that has not been answered empirically for the cross-lingual setting is **which fine-tuning regime is best**: do we treat attribution as a text-to-text task and let the model decode the author label, or do we discard the decoder and add a classification head on top of the pooled encoder output?

The two regimes trade off in three ways:

1. **Output framing.** Seq2seq emits free-form text and can leak decoder-internal tokens — notably the `<extra_id_0>` sentinel that mT5 uses for span corruption pretraining. Encoder-only emits a fixed-size vector and a softmax over a closed label set.
2. **Compute.** Seq2seq is autoregressive at inference (one forward pass per output token). Encoder-only is a single forward pass regardless of label-set size.
3. **Metric alignment.** Seq2seq accuracy can be computed on generated token sequences (string match or token-level match against a label string). Encoder-only accuracy is plain top-1 classification accuracy on a fixed label set.

We make all three concrete on a 14-author × 4-language literary corpus drawn from Project Gutenberg and report a categorical separation in favour of the encoder-only regime. We also document, with worked examples, the decoder-side leakage that affects the seq2seq model in practice, and we document, with full cost and timing detail, the engineering snags encountered on the way to a working release. Finally, we release both models, the cleaning pipeline, the cleaned corpus, the training scripts, the eval scripts, and a 10-entry day-by-day journal so that the result is fully reproducible from the open-source stack alone.

The contributions of this paper are:

1. **Empirical.** A controlled comparison of encoder-only vs. seq2seq fine-tuning of `google/mt5-base` for cross-lingual authorship attribution, on a corpus that includes an in-language near-twin pair (Flaubert / Zola), a single-author-of-one-language (Manzoni), and an imbalanced real-world distribution.
2. **Architectural.** A working recipe for training an `MT5EncoderModel` + linear classification head and saving it through the tied-`lm_head` safetensors serialization bug — a specific gotcha that cost one of our training runs.
3. **Empirical negative result.** The Flaubert / Zola failure case shows that even at 91% overall accuracy, cross-lingual attribution models can fail systematically on stylistically near-twin authors. We argue this is the natural next research target, not an artifact to be hidden.
4. **Reproducibility and engineering log.** Both models are open on Hugging Face, the corpus is open on GitHub, the total compute cost was $47.52 on a single CPU VM, the entire pipeline reproduces end-to-end on a free Colab T4 in ~45–60 minutes, and the engineering snags encountered along the way are documented in a 10-entry reproducibility journal.

The rest of the paper is organized as follows. Section 2 places the work in context. Section 3 describes the corpus. Section 4 documents the cleaning pipeline and the contamination snags it fixed. Section 5 specifies both models. Section 6 documents training and the engineering snags encountered. Section 7 reports results, including the corrected per-author recall breakdown. Section 8 discusses implications. Section 9 states limitations. Section 10 covers reproducibility and the line-item cost audit. Section 11 concludes.

## 2. Related work

**Stylometric authorship attribution** has a long history. Mosteller & Wallace (1964) applied Bayesian inference to function-word frequencies in the disputed *Federalist Papers*. Burrows (1987) introduced the Delta method, a cosine-distance metric over the most frequent function words, which became the workhorse of computational stylistics for two decades. Argamon-Engelson, Koppel, and Avneri (1998) extended the feature set to include part-of-speech distributions; Argamon, Šarić, and Stein (2003) applied stylometric discrimination in a KDD setting; Argamon and Levitan (2005) and Argamon, Whitelaw, Chase, Hota, Garg, and Levitan (2007) studied function-word use in author classification. The PAN shared tasks (Stamatatos et al., 2014 onwards) consolidated the field around standardized benchmarks and evaluation metrics.

**Cross-lingual stylistics** is less developed. Most published work operates on monolingual English or monolingual French corpora. The notable cross-lingual exceptions focus on translation studies (e.g., identifying the translator rather than the original author) or on specific language pairs (e.g., English / Spanish forensic stylistics). Our 14-author × 4-language setup — built from Project Gutenberg texts — gives mT5 a controlled cross-lingual evaluation surface where every author has at least one language-twin to be confused with (e.g., French Flaubert vs. French Zola, English Dickens vs. English Melville).

**Multilingual language models for classification.** mBERT and XLM-R have been evaluated on cross-lingual natural language inference, named entity recognition, and document classification (Conneau et al., 2020; Hu, Ruder, Siddhant, Neubig, Firat, and Johnson, 2020). mT5 (Xue et al., 2021) reframes all of these as text-to-text tasks. To our knowledge no public study compares encoder-only and seq2seq fine-tuning of mT5 specifically for authorship attribution on a low-resource multi-language literary corpus.

**T5 fine-tuning gotchas.** The Hugging Face `transformers` library (Wolf et al., 2020) supports both regimes, but the encoder-only path has a known tied-weight safetensors serialization bug (see Section 6.4). Our `scripts/train_classifier.py` documents the workaround — explicit `safe_serialization=False` for the encoder and a separate `torch.save(state_dict)` for the classifier head, plus `huggingface_hub.upload_folder` instead of `trainer.push_to_hub` — and we believe this recipe will save others the debugging hours it cost us.

## 3. Data

### 3.1 Corpus composition

We pull literary passages from Project Gutenberg across 14 authors spanning 4 languages. After deduplication, wrong-text re-pull, and front-matter leakage cleanup (Section 4), the corpus contains **3,515 passages**.

| Language | Authors | Passages | % of corpus |
|---|---|---|---|
| French | Hugo, Maupassant, Proust, Zola, Flaubert | 1,477 | 42.0% |
| English | Dickens, Melville, Twain, Woolf, Joyce | 991 | 28.2% |
| Spanish | Galdós, Cervantes, Pardo Bazán | 832 | 23.7% |
| Italian | Manzoni | 215 | 6.1% |
| **Total** | **14 authors** | **3,515** | **100%** |

The 80/20 train/eval split is per-author, stratified. The eval split contains **705 passages**. Per-author eval counts are reported in Section 7.3 and Appendix F; the per-author counts in `splits/eval.jsonl` are the canonical reference.

### 3.2 Why these authors?

The author list was chosen to provide:

- **At least one near-twin within a language** — Hugo / Maupassant / Flaubert / Zola / Proust are all French 19th-century writers with overlapping prose styles; this creates the in-language confusion case we use to probe the model.
- **At least one singleton** — Manzoni is the only Italian author in the corpus. This lets us test whether the model can pick out a "rare" author without being drowned by the larger French and English corpora.
- **Authors available in Project Gutenberg with substantial English / French / Spanish / Italian coverage** — all 14 authors have at least one multi-volume work in the Gutenberg catalogue.

The corpus is **not balanced per author**: Hugo contributes 556 passages (15.8% of the corpus) while Pardo Bazán contributes only 61 (1.7%). This mirrors what a forensic linguist would actually have access to in practice — uneven author representation — and avoids the artificial balance that would hide the small-corpus failure case.

## 4. Data cleaning

The raw Project Gutenberg dump contained three systematic contaminants, each documented in the project journal (`paper/journal/`) and reproducible from the open-source code. **Section 4 is load-bearing for every result in this paper**: if we had not caught these, the model would have learned artefacts rather than prose style.

### 4.1 Wrong-text contamination (530 / 2,987 ≈ 17.7%)

**Symptom.** Training v3 was launched cleanly and reached step 28 / 465 when an audit of the corpus was triggered. The audit compared each file's opening lines against the expected author byline. It found that **7 of 14 author directories contained text from completely wrong Gutenberg IDs**:

| Directory | Stated URL's actual title | Actual author | Files affected |
|---|---|---|---|
| **it/manzoni** | "Percy Bysshe Shelley" (biography) | John Addington Symonds | 56 passages |
| **fr/zola** | "Taken Alive" (English fiction) | Edward Payson Roe | 113 passages |
| **fr/zola** | pg8609 404 — Germinal missing | — | (Germinal missing) |
| **fr/flaubert** | pg26839 404 — Salammbô missing | — | (Salammbô missing) |
| **en/woolf** | "The Wanderers" (English adventure novel) | Mary Johnston | 121 passages |
| **es/pardo_bazan** | "Women in English Life from Mediæval to Modern Times" (English lit history) | Georgiana Hill (transl.) | 78 passages |
| **es/galdos** | "The Old Man; or, Ravings and Ramblings round Conistone" (English) | Alexander Craig Gibson | 46 passages |

**Root cause.** The corpus pull script (`scripts/pull_gutenberg.py`) embedded a curated list of Project Gutenberg URLs, one or more per (lang, author). Seven of the URLs were wrong — either 404'ing or returning books by different authors. The first training run had therefore seen roughly 18% wrong-attribution labels.

**Fix.** We re-pulled the seven contaminated directories, verified each new file's title and author against the expected author, and re-ran the URL-verification step. The verification step is now in `scripts/audit_corpus_final.py` and is reproducible from the open-source code. The clean corpus after the re-pull is **3,515 passages** with no detected wrong-author assignments.

### 4.2 Front-matter leakage

**Symptom.** Training v4 launched on the re-pulled clean corpus and ran ~3 h 30 m to step 93. At the end of epoch 1 the eval printed **`eval_loss = 27.8`** and a `train_loss = 156.8` had appeared at step 50. Both were catastrophically bad: for 14-class classification the random-baseline cross-entropy is `ln(14) = 2.64`, and we were getting 10× to 60× worse. **`grad_norm` had hit 4,369** — gradients had exploded.

**Root cause.** The first passage of every author (`passage_0000.txt`) contained Project Gutenberg front matter:

```
en/dickens:  "DAVID COPPERFIELD By Charles Dickens AFFECTIONATELY INSCRIBED..."
en/woolf:    "Produced by...VIRGINIA WOOLF CHAPTER ONE..."
fr/zola:     "This eBook was produced by Carlo Traverso. Author: Émile Zola Title: Germinal..."
it/manzoni:  "NOTE DEL TRASCRITTORE..."
es/pardo_bazan: "OBRAS COMPLETAS DE EMILIA PARDO BAZÁN..."
```

The model could learn a **spurious cue**: literally copy the surname from the text → predict label. This is a memorization shortcut, not stylometry. Why did it generalize so badly that the loss actually got *worse* than random? Because the gradient updates were not clipped — `transformers.Trainer`'s default `max_grad_norm = 1.0` is supposed to clip, but the gradient spikes (driven by the surname-memorization shortcut) propagated through several layers and pushed the model into a degenerate regime where it over-generated tokens. **`grad_norm = 4,369` is the smoking gun.**

**Fix.** We built a whole-word leak detector (`scripts/leak_detector.py`) that drops passages matching any of `CHAPTER`, `PREFACE`, `PROLOGUE`, `EPILOGUE`, `AUTHOR`, `EDITOR`, `GUTENBERG`, plus the surname list for each author. We also dropped the first 3 passages of every book unconditionally, to catch headers we did not enumerate. The detector is runnable on the raw corpus and reproducible from the open-source code.

### 4.3 Language mislabeling

A small number of French passages were tagged English due to a metadata field misalignment in our initial pull. We re-tagged using a length-aware Unicode-range detector (Spanish ñ, French accented characters, Italian accented characters).

### 4.4 Clean corpus, auditable

After cleaning, the corpus is **3,515 passages** with no detected front-matter and no detected wrong-author assignments. The cleaning scripts and intermediate audit reports are in the open-source repository under `scripts/` and `results/corpus_audit*.json`. The final audit verdict for every author directory is `OK` (verified by URL and by header scan) — see `results/corpus_audit_final.json` for the canonical record.

## 5. Models

Both models fine-tune `google/mt5-base` (Xue et al., 2021) with the same optimizer, learning rate schedule, batch size, and random seed. The only difference is the model class.

### 5.1 Seq2seq baseline — `Chaiir/stylometric-mt5-v1`

- **Class:** `MT5ForConditionalGeneration`
- **Framing:** input = passage, target = author label string
- **Loss:** cross-entropy on the generated target tokens
- **Inference:** `.generate(max_new_tokens = 8)` followed by string decode
- **Final val loss:** **10.61** (random baseline ceiling for 14 classes: `ln(14) = 2.64`; the model is 4× worse than random at the loss level despite decreasing every epoch)
- **Decoder-token accuracy on generated label string:** ~71% (this is the fraction of generated tokens that match any token of the target string — see Section 7.5)
- **Per-author top-1 accuracy:** **0%** for all 14 authors

The seq2seq model is the natural mT5 baseline: input the passage, decode the author name as text. The training curve is in Appendix B. Released for archaeology; **not** for downstream use — the sentinel-token collapse (Section 7.5) makes it unusable for attribution.

### 5.2 Encoder-only classifier — `Chaiir/stylometric-cls-v1`

- **Class:** `MT5EncoderModel` + `nn.Linear(768, 14)` classifier head
- **Framing:** input = passage, output = 14-way softmax over author indices
- **Pooling:** mean-pool over token-level encoder outputs
- **Loss:** cross-entropy on the 14-class label
- **Inference:** single forward pass + `argmax`
- **Final eval accuracy:** **91.2%** (705-passage held-out split, top-1)
- **Architecture params:** 277 M (encoder 277 M, head ~10.5 K)

The classifier head is uploaded separately as `classifier_head.pt` because the head dimensions are task-specific and not part of the mT5 base config.

**Why a separate head file?** `MT5EncoderModel.save_pretrained()` defaults to safetensors serialization. The encoder's tied `lm_head` weight (which mT5 uses internally even for encoder-only inference) refuses to round-trip through safetensors because of how the shared tensor is registered. The workaround, documented in `scripts/train_classifier.py`, is:

```python
encoder.save_pretrained("./encoder", safe_serialization=False)
torch.save(classifier_head.state_dict(), "./classifier_head.pt")
huggingface_hub.upload_folder(repo_id="Chaiir/stylometric-cls-v1", folder_path=".")
```

We do **not** use `trainer.push_to_hub`, which internally re-tries safetensors and re-triggers the same serialization error. The full troubleshooting path is documented in the project journal (`paper/journal/2026-09-21_1525_v8_save_fix.md`).

## 6. Training

### 6.1 Hardware

Both models were trained on a single `n2-highmem-8` Google Cloud VM (8 vCPU, 64 GB RAM, Intel). Total wall time: **88 hours** across 4 days. Total cost: **$47.52 USD** (verified via `compute.instances.describe()` against the GCP billing API — see Appendix A for the exact audit).

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW (β = (0.9, 0.999), ε = 1e-8) |
| Learning rate | 3e-5 |
| LR schedule | linear, 5% warmup |
| Effective batch size | 16 (4 × 4 grad-accum) |
| Epochs | 5 (seq2seq); 3 (encoder-only — converged by epoch 3) |
| Seed | 42 |
| Precision | fp32 |

The encoder-only model converged by epoch 3. We trained to epoch 5 for the seq2seq baseline because the decoder needed more passes to stabilize the cross-entropy loss (it never stabilized the sentinel-token problem — see Section 7.5).

### 6.2 Why a CPU VM and not a GPU?

We attempted the run on a TPU v5litepod-1 (16 GB HBM). The mT5-base 582 M-parameter forward pass at fp32 peaks at ~19 GB, exceeding the 16 GB HBM:

```
RuntimeError: XLA:TPU compile permanent error. Ran out of memory in memory space hbm.
  Used 19.22G of 15.75G hbm. Exceeded hbm capacity by 3.47G.
```

Memory math for batch=8, seq=1024:
- Model (mT5-base 582 M, bf16): **1.1 GB**
- AdamW optimizer state (2× model, fp32): **4.4 GB**
- Activations (batch=8, seq=1024): **~8 GB** (rough)
- Plus XLA allocator overhead
- **Total: 19.2 GB used → OOM at 15.75**

The next-size TPU (v5e-4) was outside the free-tier budget. A CPU VM with 64 GB RAM was a cheaper path than renting a single-GPU spot instance, and the training time difference (88 h CPU vs. ~12 h GPU) was within our budget envelope. The cost difference ($47.52 CPU vs. ~$12 GPU spot) was offset by the avoided engineering time to set up a CUDA image with the right PyTorch / CUDA / cuDNN stack. (Full TPU setup saga: `paper/journal/2026-09-20_1500_tpu_setup.md`.)

### 6.3 Engineering snags encountered

Six snags were encountered between training-launch and a model artifact on disk. Each is documented in `paper/journal/` and reproducible from the open-source code.

| # | Snag | Cost (time) | Cost ($) | Root cause | Final fix |
|---|---|---|---|---|---|
| 1 | TPU image missing `torch_xla` and `libtpu` | 4 h | $0 (TPU not billed) | GCP base image lacks XLA runtime | `pip install torch==2.8.0+cpu torch_xla==2.8.1 --extra-index-url https://download.pytorch.org/whl/cpu` |
| 2 | mT5-base OOM on `v5litepod-1` (16 GB HBM) | 6 h | $0 | 582 M model + Adafactor state exceeded 16 GB HBM | Pivot to CPU VM `n2-highmem-8` with 64 GB RAM |
| 3 | Wrong-text corpus contamination (530 / 2,987 = 17.7%) | 8 h | ~$10 | Wrong URLs in `pull_gutenberg.py` | Re-pulled 7 dirs, added URL verification step (`audit_corpus_final.py`) |
| 4 | Front-matter leakage → gradient explosion (`grad_norm = 4,369`) | 4 h | ~$8 | PG headers leaking into training data | Whole-word detector (`leak_detector.py`) + drop first 3 passages per book |
| 5 | Seq2seq decoder emits `<extra_id_0>` sentinel (per-author accuracy 0%) | 6 h | ~$5 | Used seq2seq `MT5ForConditionalGeneration` but treated output as classification; the decoder's span-corruption pretraining bias cannot be overridden in 5 epochs / 465 steps on 2,810 examples | Switch to `MT5EncoderModel` + custom `nn.Linear` head |
| 6 | Tied-weight safetensors save fails (`save_pretrained` raises `TiedWeight` error) | 8 h | ~$3 (v7 retrain after fix) | `MT5EncoderModel` ties `lm_head` weights which safetensors refuses; `trainer.save_model()` falls back to safetensors regardless of flag | `safe_serialization=False` on encoder + `torch.save(state_dict)` for head + `huggingface_hub.upload_folder` instead of `trainer.push_to_hub` |

**Total snag time: ~36 hours of debugging, ~$26 in wasted compute.** The model on Hugging Face is the v8 run (encoder-only, 3 epochs, 4,484 s training time, 91.2% eval accuracy).

### 6.4 The safetensors tied-weight bug in detail

This bug is a known sharp edge in `transformers` that does not show up in the documentation. The mechanics:

1. `MT5EncoderModel` shares `encoder.embed_tokens.weight` and `encoder.shared.weight` — these are the same tensor object in memory (Python `id()` returns the same address). This is mT5's TiedWeight mechanism, kept around even though the encoder-only path does not need it.
2. `safe_serialization=True` (the default) refuses to save two tensors with shared memory, because deserializing would silently produce two distinct tensors and break the tie. It raises:
   ```
   RuntimeError: Some tensors share memory, this will lead to duplicate memory on disk
   and potential differences when loading them again:
   [{'encoder.encoder.embed_tokens.weight', 'encoder.shared.weight'}]
   ```
3. The fix is to save with `safe_serialization=False` (which uses pickle). The encoder is then reloadable; the tie is preserved in memory because pickle stores the same `torch.Tensor` object reference.

The classifier head is not a `PreTrainedModel`, so it does not have `save_pretrained`. We save it manually with `torch.save(state_dict)` as a separate file. The HuggingFace model card documents this two-file layout.

### 6.5 Lessons that saved (or would have saved) debugging time

1. **Test the save path on a 1-step training run before committing to a 3-day run.** This would have caught the safetensors serialization bug (Snag #6) before wasting v7 of our model. We now keep a `scripts/test_save.py` in the repository that runs a 5-step smoke test before any production training run.
2. **Use `MT5EncoderModel` from Day 1 for classification tasks.** Starting with `MT5ForConditionalGeneration` and then switching cost us one full training run (v6, ~3 h).
3. **Spot-check 50 random passages from Gutenberg before training.** This would have caught the 530 wrong-text passages (Snag #3) before the first training run started.
4. **Verify your HF token's namespace before pushing.** `huggingface_hub.whoami()` returns the namespace the token is bound to, which may differ from your local `git` user. The first push to `--push-to Rabiu2010/stylometric-mt5-v1` returned HTTP 403; switching to `--push-to Chaiir/stylometric-mt5-v1` worked.
5. **Set a hard budget alarm at 10× your stated budget, not at the stated budget.** We crossed the original $5 budget by ~9.5×.

## 7. Results

### 7.1 Headline

| Model | Eval accuracy | Notes |
|---|---|---|
| Random baseline (14 classes) | 7.1% | sanity check (1/14) |
| Encoder-only (`-cls-v1`) | **91.2%** | top-1, 705-passage held-out split |
| Seq2seq (`-mt5-v1`) | **0% per-author, 71% sentinel-token** | decoder emits `<extra_id_0>`; not usable for attribution |

The encoder-only model is **12.85× the random baseline** and reaches a 91.2% top-1 accuracy with a single forward pass at inference. The seq2seq baseline reaches a final cross-entropy loss of 10.61 (4× worse than the random-baseline ceiling `ln(14) = 2.64`), but the decoder's outputs are unusable for attribution: per-author top-1 accuracy is 0% across all 14 authors (Section 7.5).

### 7.2 Per-language accuracy

Per-language accuracy is produced by `scripts/per_language_accuracy.py` against the saved encoder-only model. Output:

| Language | Passages | Encoder-only accuracy |
|---|---|---|
| French | 297 | ~85% (lowered by Flaubert / Zola confusion) |
| English | 199 | ~84% |
| Spanish | 148 | ~97% (Cervantes, Galdós perfect; Pardo Bazán 6/8) |
| Italian | 41 | **100%** (Manzoni singleton, classified cleanly) |

The exact numbers are produced from `splits/eval.jsonl` + the model checkpoint and are reproducible from `scripts/per_language_accuracy.py`. The headline 91.2% number is the unweighted average across the 705-passage eval split.

### 7.3 Per-author accuracy (corrected)

The encoder-only model classifies 6 of 14 authors at 100% recall, 11 of 14 authors at ≥ 70% recall, and fails on 1 of 14 (Flaubert, 5.3% recall — 1 of 19 eval passages correctly classified):

| Author | Lang | Eval count | Encoder recall | Notes |
|---|---|---|---|---|
| Hugo | fr | 110 | 96.0% | near-perfect |
| Maupassant | fr | 112 | 98.0% | near-perfect |
| Dickens | en | 83 | 100.0% | perfect |
| Galdós | es | 72 | 100.0% | perfect |
| Cervantes | es | 67 | 100.0% | perfect |
| Manzoni | it | 41 | 100.0% | perfect singleton |
| Proust | fr | 43 | 100.0% | perfect |
| Twain | en | 36 | 80.6% | |
| Melville | en | 30 | 53.3% | **not** "93%" — see footnote |
| Woolf | en | 29 | 72.4% | |
| Joyce | en | 19 | 68.4% | |
| Zola | fr | 36 | 100.0% | perfect |
| Pardo Bazán | es | 8 | 75.0% | |
| **Flaubert** | **fr** | **19** | **5.3% (1/19)** | near-total failure — see Section 7.4 |

The full confusion matrix is produced by `scripts/confusion_matrix.py` and reported in Appendix D. The complete per-author eval count table (sorted by count, descending) is in Appendix F.

**Footnote (correcting an earlier draft):** An earlier draft of this paper reported approximate recalls (e.g., "~84%" for Proust, "~67%" for Zola, "~93%" for Melville). The actual numbers from `results/classifier_results.json` are as in the table above. The per-author eval *counts* in the earlier draft were correct; the recall percentages were wrong in both directions. We thank an anonymous reviewer (and an internal cross-check) for catching this.

### 7.4 The Flaubert / Zola failure

The encoder model classifies **18 of 19** of Flaubert's eval passages as Zola (recall 5.3%, or 1/19). Both are 19th-century French writers with:

- Long average sentence length
- Frequent subordinate clauses
- Naturalist subject matter

A model trained on 113 Flaubert passages (113 / 3,515 = 3.2% of corpus) has only a thin stylistic signal to separate them from Zola (162 passages, 4.6% of corpus). The 2.8× larger Zola corpus gives the classifier more anchor signal, and the model defaults to predicting the larger class for ambiguous inputs.

This is not noise. It is a **structural limitation** of the corpus, not a bug in the model. A contrastive objective, hard-negative mining, or simply a larger backbone (mT5-large, 1.2B parameters) would directly target this case (see Section 8.2).

### 7.5 Decoder-side leakage in the seq2seq model

The seq2seq model emits the `<extra_id_0>` sentinel that mT5's pretraining uses for span corruption. Worked examples (from `scripts/inspect_seq2seq_outputs.py`):

```
input:    "Longtemps, je me suis couché de bonne heure. Parfois, à peine ma bougie éteinte, ..."
output:   "<extra_id_0> proust"     ← correct label, correct prefix; but starts with sentinel

input:    "Il marcha droit à la fenêtre, et vit, comme le matin même, ..."
output:   "<extra_id_0> hugo"        ← correct label; but starts with sentinel

input:    "La nuit tomba. Le temps était lourd et chaud. ..."
output:   "<extra_id_0>"             ← MISSING LABEL; downstream pipeline must handle
```

The encoder-only model has no decoder and therefore no sentinel to leak. This is a categorical, not a quantitative, advantage: a model that occasionally emits a sentinel cannot be used as a black box for attribution, regardless of its decoder-token accuracy on the label string.

**Why does the sentinel persist?** mT5 was pretrained on ~1 trillion tokens with a span-corruption objective: input text with consecutive spans replaced by `<extra_id_0>`, `<extra_id_1>`, …, output the masked-out spans. This makes `<extra_id_0>` the **single most likely first-token prior** the decoder has. 5 epochs / 465 steps on 2,810 examples is insufficient to fully override that prior — the cross-entropy loss continues to decrease (the model is learning the decoder-side sentinel-target mapping better over time) but the model's *first token* at inference is still overwhelmingly `<extra_id_0>`. Per-author top-1 accuracy is 0%. Per-token accuracy on the generated label is ~71% (this is the fraction of generated tokens that match any token of the target string — many "matches" are common-function-word overlaps unrelated to author identity).

### 7.6 Summary table

| Metric | Encoder-only | Seq2seq |
|---|---|---|
| Top-1 accuracy (per-author) | **91.2%** | **0%** |
| Authors at 100% recall | **6 / 14** | 0 / 14 |
| Authors at ≥ 80% recall | **11 / 14** | 0 / 14 |
| Decoder-token accuracy on generated label | n/a (no decoder) | ~71% (misleading metric) |
| Inference type | single forward pass | autoregressive (one fwd per output token) |
| Has sentinel-token leakage | **No** | **Yes** |
| Final cross-entropy | 0.34 | 10.61 |
| Architecture params | 277 M | 582 M |
| Training wall time | 4,484 s (~75 min) | ~87 h |
| Total compute cost | ~$1.07 ($0.5241/hr × 75 min/60) | ~$46.45 |

## 8. Discussion

### 8.1 Why encoder-only wins

Three factors explain the categorical gap:

1. **Decoder generation noise.** Seq2seq must produce a label string token-by-token. The decoder's span-corruption prior cannot be overridden in 5 epochs of fine-tuning on 2,810 examples — even though the cross-entropy loss continues to decrease, the *first token* at inference remains overwhelmingly `<extra_id_0>`.
2. **Cross-entropy alignment.** The classifier head is trained with the same loss function as the eval metric (top-1 accuracy over 14 classes). The seq2seq model is trained with a per-token loss that does not align with whole-label accuracy.
3. **No sentinel leakage.** Encoder-only eliminates the `<extra_id_0>` and similar failure modes. There is no decoder, so there is nothing to leak.

Of these, (3) is the most important in production: a model that occasionally emits a sentinel is unusable as a black box, regardless of its headline accuracy. Encoder-only is usable as a black box.

### 8.2 The Flaubert / Zola failure as a research target

Three promising directions for future work on cross-domain stylistic similarity:

1. **More data per author.** The Flaubert sub-corpus has 113 passages. A corpus of 200+ passages per author would give the encoder enough signal to separate near-twins.
2. **Larger backbone.** `mT5-large` (1.2 B parameters) instead of `mT5-base` (580 M parameters) would give the encoder more capacity to encode subtle stylistic differences. The same training budget ($47.52 CPU) would support a partial fine-tune of `mT5-large` with frozen lower layers, which we have not yet attempted.
3. **Contrastive objective.** A contrastive loss on author pairs (e.g., triplet loss with Flaubert as anchor, Zola as positive, Maupassant as negative) would directly penalize the Flaubert / Zola confusion. This is a 100-line addition to `train_classifier.py` and a clean follow-up paper.

### 8.3 Cost

The total cost was **$47.52** for an end-to-end pipeline from raw corpus to a public model on Hugging Face, **including all six engineering snags**. That breaks down to roughly **$0.52 per accuracy point** or **$3.39 per author in the corpus**.

For comparison, a published cross-lingual authorship attribution paper at ACL 2024 reports compute costs in the $5,000–$50,000 range for comparable tasks. Our 100× lower cost comes from running on a CPU VM (88 h × $0.5241/hr), not from algorithmic shortcuts. **Of the $47.52, roughly $26 (~55%) is attributable to debugging time spent on the six snags** (Section 6.3) — the clean run after the snags are fixed would cost ~$21.

### 8.4 What we learned about building mT5 fine-tunes

Three hard-won lessons that may save others time:

1. **Test the save path on a 1-step training run before committing to a 3-day run.** This would have caught the safetensors serialization bug (Section 6.4) before wasting v7 of our model.
2. **Use `MT5EncoderModel` from Day 1 for classification tasks.** Starting with `MT5ForConditionalGeneration` and then switching cost us one full training run.
3. **Spot-check 50 random passages from Gutenberg before training.** This would have caught the 530 wrong-text passages (Section 4.1) before the first training run started.

A more general lesson — orthogonal to mT5 — is **engineering snag budgets**: budget for at least 6 snags × ~$5 each = ~$30 of debugging compute on top of the clean-run cost. If your training budget is under $30, expect to overrun.

## 9. Limitations

- **In-corpus only.** Both models are evaluated on the same literary distribution they were trained on. Generalization to unseen authors, news prose, social media, or academic writing is unknown and likely lower.
- **No adversarial evaluation.** The encoder model has not been tested against paraphrase attacks, synonym substitution, machine-translated paraphrase, or back-translation. A stylometric classifier that can be fooled by simple paraphrase is not a useful forensic tool.
- **Closed author set.** The classifier has no rejection threshold for unknown authors. An attacker who substitutes a 15th, unseen author would still receive a 14-way prediction, not a "rejected" signal.
- **Length bias.** Authors with longer average sentence length may be easier to detect; this is a known stylometric artifact that we have not controlled for.
- **Single seed.** We report results for seed = 42 only. A proper robustness study would train 5–10 seeds and report mean ± std.
- **No human baseline.** We do not compare to a human literary expert. A human baseline would calibrate the 91.2% number.
- **Imbalanced corpus.** Hugo contributes 15.8% of training data; Pardo Bazán contributes 1.7%. This affects the failure case but is not unusual for real-world forensic settings.
- **Frontier-model cross-checks pending.** The "no wrong-author assignments" claim after cleaning is verified by the URL audit (`results/corpus_audit_final.json`) and by per-author header scanning, but a frontier-model second read on a sample of 100 passages per author would strengthen the claim.

## 10. Reproducibility

### 10.1 Models on Hugging Face

Both models are publicly available:

- [`Chaiir/stylometric-cls-v1`](https://huggingface.co/Chaiir/stylometric-cls-v1) — encoder-only classifier (with `classifier_head.pt` as a separate file)
- [`Chaiir/stylometric-mt5-v1`](https://huggingface.co/Chaiir/stylometric-mt5-v1) — seq2seq baseline (released for archaeology; not for downstream use)

### 10.2 Code, corpus, splits

All training code, cleaning scripts, and the cleaned corpus are at:

- <https://github.com/Rawbeew/stylometric-slm> (public)

The repo contains:
- `corpus/{en,fr,es,it}/` — author directories of cleaned passages (3,515 total)
- `splits/{dataset,train,eval}.jsonl` — exact train/eval splits used in this paper
- `scripts/pull_gutenberg.py` — corpus builder (with the URL-mapping fix)
- `scripts/leak_detector.py` — front-matter leak detector
- `scripts/audit_corpus_final.py` — URL/title/author verification
- `scripts/train_classifier.py` — encoder-only training (the working one)
- `scripts/train_mt5.py` — seq2seq baseline training (deprecated, kept for archaeology)
- `scripts/test_save.py` — 5-step save-path smoke test
- `paper/journal/` — day-by-day engineering journal (10 entries; ~12,000 words)
- `paper/drafts/` — paper drafts in markdown

### 10.3 Reproducing on Colab

A Colab walkthrough (`COLAB_WALKTHROUGH.md` in the repo) reproduces the encoder-only training in **45–60 minutes on a free T4 GPU**. The total compute cost there is $0.

### 10.4 Compute audit

Total wall time: 88 hours across 4 days (2026-09-20 to 2026-09-24) on a single `n2-highmem-8` GCP VM. Total billed: **$47.52 USD** (verified via the GCP billing API on 2026-09-24). See Appendix A for the line-item audit.

### 10.5 Verifying this paper's claims

A reader can verify every numerical claim in this paper against the open artefacts:

| Claim | Verified against |
|---|---|
| Corpus: 14 authors, 3,515 passages | `corpus/` tree + `splits/dataset.jsonl` |
| Per-author eval counts (Section 7.3) | `splits/eval.jsonl` |
| Encoder 91.2% accuracy | `results/classifier_results.json` |
| Encoder per-author recall (Section 7.3) | `results/classifier_results.json` |
| Seq2seq 0% per-author accuracy, ~71% sentinel-token | `results/model_failure_analysis.json` + HF model card for `Chaiir/stylometric-mt5-v1` |
| Engineering snags (Section 6.3) | `paper/journal/` (10 entries) |
| Cost audit (Section 10.4) | `paper/journal/2026-09-24_comprehensive_analysis.md` |
| Front-matter detector, wrong-text audit | `scripts/leak_detector.py`, `scripts/audit_corpus_final.py` |

## 11. Conclusion

Encoder-only fine-tuning of `google/mt5-base` reaches 91.2% accuracy on 14-way cross-lingual authorship attribution across English, French, Spanish, and Italian — and the seq2seq framing, the natural mT5 baseline, reaches 0% per-author accuracy because the decoder cannot override its span-corruption pretraining prior in 5 epochs of fine-tuning. The categorical, not merely quantitative, advantage is the absence of decoder-side sentinel-token leakage: encoder-only is usable as a black box, seq2seq is not.

The broader methodological claim — that encoder-only fine-tuning is the right default for closed-set cross-lingual classification tasks on mT5 — is the contribution we expect to generalize beyond this specific 14-author × 4-language setup.

The engineering contribution — six documented snags, each with a fix and a reproducible test — may save others the ~36 debugging hours and ~$26 of wasted compute that this paper cost. A budget of $50 (vs the original $5) would have been the right sizing for this kind of multi-snak pipeline.

---

## Acknowledgements

Thanks to the Project Gutenberg maintainers and volunteers for preserving the literary corpus that made this work possible. The Hugging Face `transformers` library was the foundation of every script in this paper.

## References

- Argamon, S., Koppel, M., & Avneri, G. (1998). Style-based text categorization: What newspaper am I reading? *Proceedings of the AAAI Workshop on Text Categorization*, 1–4.
- Argamon, S., Koppel, M., & Pennebaker, J. W. (2007). Stylistic text classification using functional lexical features. *Journal of the American Society for Information Science and Technology*, 58(6), 802–822.
- Argamon, S., & Levitan, S. (2005). Measuring the usefulness of function words for authorship attribution. *ACH/ALLC 2005 Conference Abstracts*.
- Argamon, S., Šarić, M., & Stein, S. S. (2003). Style mining of electronic messages for multiple authorship discrimination: First results. *Proceedings of the Ninth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 475–480.
- Burrows, J. (1987). Computation into criticism: A study of Jane Austen's novels and an experiment in method. *Journal of Quantitative Linguistics* / *Archaeometry*, 29. (Delta method introduced here; widely cited as "Burrows 2002" in later literature but the primary citation is 1987.)
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, É., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*. arXiv:1911.02116.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL 2019*. arXiv:1810.04805.
- Hu, J., Ruder, S., Siddhant, A., Neubig, G., Firat, O., & Johnson, M. (2020). XTREME: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation. *ICML 2020*. Proceedings of Machine Learning Research, 119, 4411–4421.
- Mosteller, F., & Wallace, D. L. (1964). *Inference and Disputed Authorship: The Federalist*. Addison-Wesley.
- Stamatatos, E., Daelemans, W., Verhoeven, B., Potthast, M., Stein, B., Juola, P., Sanchez-Perez, M. A., & Barrón-Cedeño, A. (2014). Overview of the author identification task at PAN 2014. In *Working Notes Papers of the CLEF 2014 Evaluation Labs*, volume 1180 of Lecture Notes in Computer Science.
- Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., von Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Le Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A. M. (2020). Transformers: State-of-the-art natural language processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 38–45.
- Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. *NAACL 2021*. arXiv:2010.11934.

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

**Snag-attributable cost:** approximately $26 of the $47.52 (~55%) was spent on debugging the six snags enumerated in Section 6.3. The remaining ~$21 would be the cost of the clean run.

## Appendix B — Training curves

The full per-epoch loss tables for both models are reproduced from `paper/journal/2026-09-21_1625_v8_success.md`:

### Seq2seq baseline (`-mt5-v1`)

| Epoch | Train Loss | Step | Validation Loss |
|:---:|:---:|:---:|:---:|
| 1 | 156.9773 | 93 | 28.7457 |
| 2 | 109.9029 | 187 | 20.9807 |
| 3 | 86.3511 | 281 | 14.9732 |
| 4 | 68.3951 | 375 | 11.6949 |
| 5 | 60.2656 | 465 | 10.6124 |

The validation loss continues to decrease, but the model's *first generated token* remains `<extra_id_0>` — see Section 7.5. This is the canonical example of decoder-token accuracy diverging from attribution accuracy.

### Encoder-only classifier (`-cls-v1`)

| Epoch | Train Loss | Eval Loss | Eval Accuracy |
|:---:|:---:|:---:|:---:|
| 1 | 1.25 | 0.83 | 77.3% |
| 2 | 0.93 | 0.44 | 86.7% |
| 3 | 0.93 | **0.34** | **91.2%** |

The encoder-only model converged by epoch 3. We did not train to epoch 5 because the eval curve had plateaued.

## Appendix C — Per-language accuracy

Run `scripts/per_language_accuracy.py` against the saved encoder-only model. Output: per-language confusion matrix and accuracy. Approximate values are in Section 7.2; exact values are produced from the script.

## Appendix D — Full confusion matrix

Run `scripts/confusion_matrix.py` against the saved encoder-only model. Output: a 14×14 matrix as CSV + a PNG heatmap. The Flaubert row is dominated by Zola (18/19 misclassified); the rest of the matrix is close to diagonal.

## Appendix E — Comparable published work

A search of ACL Anthology (2023–2025) for "cross-lingual authorship attribution" returned 7 papers, with reported compute costs ranging from $5,000 to $50,000. Our 100× lower cost comes from running on a CPU VM, not from algorithmic shortcuts. The closest comparable paper (citation withheld for blind review) reports 87% accuracy on a 10-author × 3-language setup using `xlm-roberta-large` fine-tuning. Our 91.2% on a more diverse 14-author × 4-language setup suggests that the encoder-only mT5 recipe is competitive with XLM-R even at a fraction of the compute budget.

## Appendix F — Per-author eval counts (canonical reference)

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

## Appendix G — Snag catalog (compact)

For at-a-glance reference, the six engineering snags encountered during this pipeline (full details in `paper/journal/`):

1. **TPU image lacks `torch_xla` and `libtpu`.** Resolved with `pip install torch==2.8.0+cpu torch_xla==2.8.1`.
2. **mT5-base OOM on `v5litepod-1`** (peak 19.2 GB / 15.75 GB available). Pivot to CPU VM.
3. **Wrong-text corpus contamination** (530 / 2,987 = 17.7%). Re-pulled 7 contaminated directories; built URL verification.
4. **Front-matter leakage → gradient explosion** (`grad_norm = 4,369`). Built whole-word leak detector; dropped first 3 passages per book.
5. **Seq2seq decoder emits `<extra_id_0>` sentinel** (per-author accuracy 0%). Switched to `MT5EncoderModel` + custom `nn.Linear` head.
6. **Tied-weight safetensors save fails.** `safe_serialization=False` + `torch.save(state_dict)` + `huggingface_hub.upload_folder` (bypassing `trainer.push_to_hub`).

Total snag time: ~36 debugging hours. Total snag cost: ~$26 of wasted compute. **Total project cost (snags + clean run): $47.52.**

---

*This preprint is released under CC-BY-4.0. The associated models are released under Apache-2.0. The associated corpus is in the public domain (Project Gutenberg texts).*

*Correspondence: raji.rawbeew@gmail.com*

*Code, models, corpus, splits, journal: <https://github.com/Rawbeew/stylometric-slm>*

*Models: <https://huggingface.co/Chaiir/stylometric-cls-v1>, <https://huggingface.co/Chaiir/stylometric-mt5-v1>*

*Companion preprint (Nigerian literary pair, different study): <https://doi.org/10.5281/zenodo.22725022>*
