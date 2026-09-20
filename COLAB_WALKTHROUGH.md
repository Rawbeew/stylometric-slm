# One-Click Colab Training

## What this is

`notebooks/one_click_train.ipynb` is a self-contained Colab notebook that fine-tunes `google/mt5-base` on the stylometric corpus and pushes the result to your Hugging Face Hub.

## How to run it

**Total time: ~45–60 min on free Colab T4 GPU.**

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
   - Fine-tune for 5 epochs
   - Push the trained model to your HF repo
   - Print per-author per-language accuracy
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

## What you get

After training, you have:
- A published mT5 model on your HF account
- Per-author per-language accuracy numbers
- A reproducible build (anyone can clone the dataset + notebook and reproduce)

That's a real paper's worth of work. The next step is the paper writeup — let me know when the training finishes and I'll draft it.
