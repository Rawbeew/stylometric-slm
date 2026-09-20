# stylometric-slm

Multilingual authorship attribution as a small language model.

**Task.** Given a passage (~1000 words), predict its author. 14 authors across 4 languages (English, French, Spanish, Italian).

**Approach.** Fine-tune `google/mt5-base` (encoder-decoder, 580M params, multilingual) on the task as text-to-text classification: input is `"classify authorship: <passage>"`, output is `"<author>"`.

**Why mT5.** Encoder-decoder is the canonical T5 architecture for classification. mT5 was pretrained on 100+ languages including all four in our corpus. Free Colab T4 fits mT5-base full fine-tune (no quantization tricks needed at this size). Trains in ~45 min on free GPU.

## Status

| Component | State |
|---|---|
| Corpus: 14 authors × 4 languages | ✅ **2,987 passages on disk** |
| Train/eval split (deterministic 80/20) | ✅ **2,382 train + 605 eval** |
| Feature-engineering baseline (logistic regression) | ✅ **73.6% EN, 87.9% ES, 66.6% FR, 42.9% IT** |
| mT5 fine-tune | ⏳ **Runs in Colab (one-click notebook)** |
| Paper | ⏳ After training completes |

## One-click training (recommended path)

See `COLAB_WALKTHROUGH.md`. Summary:
1. Get an HF token at https://huggingface.co/settings/tokens
2. Open `notebooks/one_click_train.ipynb` in Colab (free T4 GPU)
3. Set `HF_TOKEN` in Colab secrets
4. Edit cell #1 with your HF username
5. Runtime → Run all
6. ~45 min later: trained model on your HF account + accuracy table

## Authors covered (v1)

| Language | Authors | Passages |
|---|---|---|
| English | Dickens, Twain, Woolf, Joyce, Melville | 967 |
| French | Hugo, Maupassant, Proust, Flaubert, Zola | 1,453 |
| Spanish | Cervantes, Pardo Bazán, Galdós | 511 |
| Italian | Manzoni | 56 |

**Total: 2,987 passages.** The Spanish and Italian coverage is thin — adding more authors is a v2 task. (Chinese was attempted via ctext.org but the source is bot-walled; deferred to v2.)

## Repository layout

```
stylometric-slm/
├── corpus/                      # raw passages (per-language, per-author, 1000 words each)
│   ├── en/dickens/passage_0000.txt
│   ├── en/twain/...
│   ├── fr/hugo/...
│   ├── es/cervantes/...
│   └── it/manzoni/...
├── splits/
│   ├── train.jsonl              # 80% per author, deterministic hash split
│   ├── eval.jsonl               # 20% per author, never seen during training
│   └── dataset.jsonl            # combined + split label, for single-file HF upload
├── scripts/
│   ├── pull_gutenberg.py        # EN/FR/ES/IT from Project Gutenberg
│   ├── pull_ctext.py            # ZH from ctext.org (deferred; ctext bot-walled)
│   ├── build_split.py           # deterministic 80/20 split per author
│   ├── baseline_local_cpu.py    # feature-engineering baseline (logistic regression)
│   ├── train_mt5.py             # mT5 fine-tune (works on any GPU box)
│   └── push_to_hf.py            # upload corpus + splits to HF Hub
├── notebooks/
│   └── one_click_train.ipynb    # the one-click Colab training notebook
├── results/
│   └── baseline_logreg.json     # baseline accuracy numbers
├── paper/                       # draft + figures (post-training)
├── COLAB_WALKTHROUGH.md         # step-by-step Colab guide
└── README.md
```

## Build

### Local (corpus + baseline only, no GPU needed)

```bash
# 1. Pull European-language authors (no auth)
python scripts/pull_gutenberg.py --all

# 2. Build train/eval split (deterministic, no auth)
python scripts/build_split.py

# 3. Run feature-engineering baseline
python scripts/baseline_local_cpu.py
# → results/baseline_logreg.json
```

### Cloud (the real fine-tune)

**Option A — Colab (free, one click):** Open `notebooks/one_click_train.ipynb` in Colab, follow `COLAB_WALKTHROUGH.md`.

**Option B — Any GPU box:** Set `HF_TOKEN`, then:
```bash
python scripts/train_mt5.py --epochs 5 --push-to your-username/stylometric-slm-mt5
```

## Baselines

Two baselines on the held-out split:

1. **Feature-engineering baseline.** Stylometric feature set (TTR, syllables/word, sentence length, function-word ratio, punctuation density) → logistic regression. Achieves **73.6% EN, 87.9% ES, 66.6% FR, 42.9% IT** on the eval split.
2. **Frozen-base-with-RAG baseline.** (Implemented in v2 plan; deferred from v1.)

The fine-tuned mT5 SLM is compared against the feature-engineering baseline on per-author, per-language accuracy + cross-language transfer tests.

## Eval

Held-out 20% per author (deterministic hash split — re-running produces the same split). Per-author per-language accuracy. Cross-language transfer: train on 3 languages, test on the 4th, measure how much Romance-language knowledge transfers to non-Romance and vice versa.

## License

- **Corpus:** All works are public domain in the US and most other jurisdictions.
- **Code:** MIT.
- **Fine-tuned model:** Apache 2.0 (inherited from mT5).

## Citation

```
@misc{stylometric-slm-2026,
  author = {Rawbeew},
  title = {Stylometric SLM for Multilingual Authorship Attribution},
  year = {2026},
  howpublished = {Hugging Face and Zenodo},
}
```
