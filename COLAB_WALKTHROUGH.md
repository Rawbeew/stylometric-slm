# One-Click Colab Training

> ⚠️ **Important — both `notebooks/one_click_train.ipynb` and `notebooks/finetune_mt5.ipynb` train the seq2seq variant. They reproduce the negative-result model (`Chaiir/stylometric-mt5-v1`), NOT the working classifier (`Chaiir/stylometric-cls-v1`).**
>
> If you want to reproduce the 91.2% published encoder classifier on Colab, see the encoder swap recipe below. As shipped, these notebooks are useful for **seeing the decoder bias live** and for **training your own seq2seq variant** (negative result confirmation).

## Encoder swap — reproduce the 91.2% model on Colab

To convert `notebooks/one_click_train.ipynb` to train the encoder-only classifier:

1. Open `notebooks/one_click_train.ipynb` in Colab (free T4).
2. In **Stage 4: Load mT5-base**, replace the `AutoModelForSeq2SeqLM` lines with:
   ```python
   from transformers import MT5EncoderModel
   from torch import nn

   class Classifier(nn.Module):
       def __init__(self, encoder, n_classes):
           super().__init__()
           self.encoder = encoder
           self.head = nn.Linear(encoder.config.d_model, n_classes)
       def forward(self, input_ids, attention_mask=None, labels=None):
           out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
           mask = attention_mask.unsqueeze(-1).float()
           pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
           logits = self.head(pooled)
           loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
           return type("O", (), {"loss": loss, "logits": logits})()

   encoder = MT5EncoderModel.from_pretrained(BASE_MODEL)
   N_CLASSES = 14  # 14 authors in the corpus
   model = Classifier(encoder, N_CLASSES)
   ```
3. In **Stage 6: Train**, replace the `DataCollatorForSeq2Seq` with the default collator and pass `labels` as a `torch.long` class-index tensor (not a string).
4. Total runtime: ~45–60 min on Colab free T4. Result: a model on your HuggingFace namespace matching `Chaiir/stylometric-cls-v1`'s eval profile.

The full rationale is in `paper/journal/2026-09-21_1040_v7_classification_fix.md` (the v6→v7 encoder-swap journal) and `paper/journal/2026-09-21_1525_v8_save_fix.md` (the safetensors tied-weight workaround). A PR adding the encoder variant as `notebooks/one_click_train_cls.ipynb` is on the roadmap but not shipped.

## Original walkthrough (seq2seq path — negative result)

The seq2seq notebook trains `AutoModelForSeq2SeqLM` and demonstrates the failure mode (decoder emits `<extra_id_0>` sentinel). Total time: ~45–60 min on free Colab T4 GPU.

### Step 1 — Get a Hugging Face token

If you don't have one:
1. Create a free account at https://huggingface.co
2. Go to https://huggingface.co/settings/tokens
3. Click "New token"
4. Type: **Write**
5. Name: anything (e.g. "colab-stylometric")
6. Copy the token (starts with `hf_...`)

### Step 2 — Open the notebook in Colab

Upload `notebooks/one_click_train.ipynb` to Google Drive, or open it directly:

1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Select `notebooks/one_click_train.ipynb`

### Step 3 — Set HF_TOKEN in Colab secrets

1. In the left sidebar, click the **key icon** (Secrets)
2. Click "+ Add new secret"
3. Name: `HF_TOKEN`
4. Value: paste your `hf_...` token
5. Toggle **ON** for this notebook

### Step 4 — Set your HF username

In cell #1 ("Config"), change this line:
```python
HF_USERNAME = "your-username"   # <-- change this
```
to your actual HF username.

### Step 5 — Run

1. Runtime → Change runtime type → **T4 GPU**
2. Runtime → **Run all**
3. Wait ~45–60 min
4. The notebook will:
   - Install dependencies
   - Upload the dataset to your HF repo (one-time)
   - Load mT5-base from HF
   - Fine-tune for 5 epochs (seq2seq)
   - Push the trained model to your HF repo
   - Print per-author per-language accuracy (will be near 0% for seq2seq)
   - Save `results.json` for download

### Step 6 — Get the results

After training finishes, look in the **Files panel** (folder icon, left sidebar) in Colab:
- `results.json` — accuracy table, download this
- `mt5-stylometric/` — the fine-tuned model checkpoints

Your trained model will also be live at:
```
https://huggingface.co/<your-username>/stylometric-slm-mt5
```

## What if something goes wrong

| Problem | Fix |
|---|---|
| "HF_TOKEN missing" | Add the secret in Colab's left sidebar |
| Out of memory | Reduce `BATCH_SIZE` from 8 → 4, or `MAX_INPUT_LEN` from 1024 → 512 |
| Colab disconnects mid-training | Re-run, the trainer saves checkpoints at every epoch |
| Dataset upload fails | Run `python scripts/push_to_hf.py` locally first (uses `~/.cache/huggingface/token`) |

## What you get (seq2seq path)

After training, you have:
- A published seq2seq model on your HF account (negative result, near-0% attribution accuracy)
- Per-author per-language accuracy numbers showing the decoder bias
- A reproducible build (anyone can clone the dataset + notebook and reproduce)

The next step is to swap to the encoder architecture (see top of this file) and re-train to reach the published 91.2%.