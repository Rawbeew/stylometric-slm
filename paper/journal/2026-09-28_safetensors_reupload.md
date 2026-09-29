# Journal Entry 13: cls-v1 safetensors re-upload (2026-09-28)

## Trigger
Multi-scanner cross-check on the HF Hub repo `Chaiir/stylometric-cls-v1` (added 2026-09-22 after v8 was uploaded as pickle) showed:

| Scanner | Verdict | Notes |
|---|---|---|
| HF Picklescan | clean | Reports pytorch_model.bin + classifier_head.pt as "not a pickle" (PyTorch ZIP containers, not raw pickle streams) |
| VirusTotal | clean | No signatures |
| JFrog Xray | clean | No vulnerabilities |
| Protect AI | **false positive** | Flagged `PAIT-PYTCH-101` on the pickle files |
| ClamAV | **false positive** | Flagged `Py.Malware.Obfuscation___builtin___getattr_GLOBAL` on the same files |

**Verdict:** Two of five scanners tripped on the same legitimate HF artefact pattern. Per the standing cross-check policy, a 2-of-5 cascade on a known pattern is a false positive, not real malware. The pickles are safe to keep.

## Decision
Replace them anyway, for two reasons:

1. **Reader signal.** A scanner cross-check page is a defensive-bonus artefact; the false-positive cascade, while technically a noise event, occupies reviewer attention. Removing the pickles sidesteps the question entirely.
2. **Format recommendation.** Safetensors has been the HF-recommended format since 2024. The original pickle save was a side-effect of the v8 safetensors-tied-weight workaround (see entry 8). The safetensors-only upload is the modern equivalent.

## Procedure (2026-09-28)
```bash
# Pull the existing model + classifier head as state dicts
python scripts/reupload_safetensors.py \
    --src-repo Chaiir/stylometric-cls-v1 \
    --dst-repo Chaiir/stylometric-cls-v1
# → writes model.safetensors + classifier_head.safetensors to a new commit
# → deletes pytorch_model.bin + classifier_head.pt from the same commit
# → updates model-index to point at the safetensors paths
```

The re-upload script converts the pickle state dicts to safetensors (one-to-one byte mapping via `safetensors.torch.save_file`), preserves all tokenizer + config files, and removes the pickles. Total transfer: ~2.06 GB, completed at ~125 MB/s.

## Post-state (verified by HF API on 2026-09-29)

Files in `Chaiir/stylometric-cls-v1` after re-upload:
```
.gitattributes
README.md
classifier_head.safetensors     ← new
config.json
model.safetensors                ← new (replaces pytorch_model.bin)
special_tokens_map.json
spiece.model
tokenizer.json
tokenizer_config.json
```

No `.pt` or `.bin` files remain. The model card "Security" section was updated to reflect this state (last-verified timestamp 2026-09-28).

## Why the HF card references this entry

The model card's "Security" section explains the picker-cascade removal. A reader cross-checking the card against `paper/journal/2026-09-21_1625_v8_success.md` (v8 entry, which described the original pickle save) would see a contradiction unless they knew about this entry. Addendum line added to v8_success (see bottom of that file).

## Cost
- Re-upload: free (HF Hub has no upload cost; bandwidth covered by HF).
- Total added wall time: ~17 minutes (download 2 GB + convert + re-upload 2 GB).
- No additional GCP spend (VM was terminated 2026-09-24, see entry 12).