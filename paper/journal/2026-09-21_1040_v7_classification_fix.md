# Journal Entry 7: v7 — Classification head fix (2026-09-21 10:40)

## What went wrong with v6
- mT5-base fine-tuned as **seq2seq** (input text → output author name)
- After 5 epochs (465 steps, 4h13m), `eval_loss` only descended to **10.61** vs random baseline `ln(14) = 2.64`
- Direct generation test showed the model outputs **`<extra_id_0>` sentinel tokens** instead of author names
- Per-author accuracy: **0% for all 14 authors**
- Model pushed to `Chaiir/stylometric-mt5-v1` but unusable

## Root cause (confirmed)
mT5 is pretrained with **span corruption**: input text → output is a sequence of `<extra_id_0>`, `<extra_id_1>`, ... sentinels bracketing masked spans. When we fine-tune for `input → author_name`, the decoder has a **strong inductive bias to start with `<extra_id_0>`** because that's what 1T-token pretraining taught it. 5 epochs of ~3000 examples isn't enough to fully override that prior; loss decreases but the model still emits sentinels at inference.

This is a **well-known issue** with using mT5/T5 for classification-style tasks. The recommended fix is either:
- Fine-tune for many more epochs / much more data (impractical here)
- Use `AutoModel` + classification head instead of `AutoModelForSeq2SeqLM`
- Use a different base model (e.g. BERT)

## Fix: classification head on the encoder only
`scripts/train_classifier.py` (new, committed at `5d5f931`):
- Uses `MT5EncoderModel` (encoder only, no decoder pretrain bias)
- Mean-pooled hidden states → linear layer → 14-way softmax
- Cross-entropy loss over author IDs (clean, well-defined target)
- `compute_metrics` returns `eval_accuracy` so we see real per-epoch accuracy

## Implementation details (lessons learned during this session)
1. `AutoModel.from_pretrained("mt5-base")` returns the **full encoder-decoder**; calling it without `decoder_input_ids` raises ValueError. Fix: use `MT5EncoderModel` directly.
2. `MT5EncoderModel` has **tied weights** (`encoder.shared.weight` ↔ `encoder.encoder.embed_tokens.weight`). HuggingFace `safetensors` refuses to save models with aliased tensors — error message: `Some tensors share memory, this will lead to duplicate memory on disk`. Fix: call `model.save_pretrained(..., safe_serialization=False)` to write a pickle `.bin` instead.
3. Trainer 4.46.0 doesn't accept `safe_serialization` kwarg on `TrainingArguments`; must call `save_pretrained` directly after `train()`.
4. `TiedWeight` re-aliases itself after `nn.Parameter(clone)` — must clone BOTH `encoder.shared.weight` AND `encoder.encoder.embed_tokens.weight` and verify `data_ptr()` differ before save.
5. CPU matmul was running with only 4 threads on 8-core VM. Fix: `torch.set_num_threads(os.cpu_count())` early.

## Smoke test results (dry-run)
- 8 steps on 32 examples, batch=4, 1 epoch: 44s, final train_loss = 2.654 (random baseline 2.639, so model is barely above random as expected with 32 examples × 1 epoch)
- 8 steps on 128 examples, batch=16, len=256: 66s, 8.3s/step

## Full training run (started 2026-09-21 10:32 UTC)
- Hyperparameters: 3 epochs, batch=16, lr=2e-4, max_input_len=256
- 2810 train examples / 705 eval examples
- Total steps: **528** (3 epochs × 176 steps)
- ETA: **~73 min** (8.3s/step × 528 = 73 min)
- Estimated cost: **$0.52** (73 min × $0.43/hr)
- Pushing to: `Chaiir/stylometric-cls-v1`
- Monitor: `C:/tmp/clf_status.json` (refreshed every 10 min)

## Snag chain (7 total now)
1. TPU OOM
2. VM OOM
3. Protobuf missing
4. Wrong URLs (530 contaminated passages)
5. Front-matter leakage (gradient explosion)
6. mT5 span-corruption pretrain bias (outputs `<extra_id_0>`)
7. **MT5EncoderModel tied-weight + safetensors** (resolved in same session by manual save)

## Cost summary (cumulative through 2026-09-21 10:40 UTC)
| Item | Cost |
|---|---|
| v4-v5 (TPU + early VM) | ~$2.36 |
| v6 training (5 epochs, 4h13m) | ~$1.81 |
| v7 setup (dry-runs, ~30 min VM) | ~$0.22 |
| **Spent so far** | **~$4.39** |
| v7 estimated (73 min × $0.43/hr) | +$0.52 |
| **Projected total** | **~$4.91 of $5** |

## Expected v7 outcomes (educated guess)
- Loss should descend below 1.5 within first epoch (encoder is good at representation)
- Final eval_accuracy should be **>50%** on random baseline of 7% (14 classes)
- Per-author accuracy should be non-zero across all authors

If accuracy is still near 0% after epoch 1, abort — there's something deeper wrong.
