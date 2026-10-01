#!/usr/bin/env python3
"""cls_v8_confusion.py — full-705 confusion matrix for the published classifier.

Runs one passage at a time on CPU with aggressive memory hygiene:
- model loaded once, fp32, eval mode
- batch size 1, del + gc after each forward pass
- results appended to results/cls_v8_confusion_progress.jsonl so a crash
  can resume

Output: results/cls_v8_confusion_705.json + results/confusion_cls_v8_705.csv
"""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROGRESS = REPO / "results" / "cls_v8_confusion_progress.jsonl"


def main():
    import torch
    from transformers import AutoTokenizer, MT5EncoderModel
    import safetensors.torch as st

    tok = AutoTokenizer.from_pretrained("Chaiir/stylometric-cls-v1")
    model = MT5EncoderModel.from_pretrained("Chaiir/stylometric-cls-v1")
    head_path = REPO / "classifier_head.safetensors"
    if head_path.exists():
        state = st.load_file(str(head_path))
    else:
        from huggingface_hub import hf_hub_download
        state = st.load_file(hf_hub_download("Chaiir/stylometric-cls-v1",
                                             "classifier_head.safetensors"))
    W, b = state["weight"], state["bias"]
    model.eval()
    print(f"loaded; head {tuple(W.shape)}", flush=True)

    AUTHORS = ['cervantes', 'dickens', 'flaubert', 'galdos', 'hugo', 'joyce',
               'manzoni', 'maupassant', 'melville', 'pardo_bazan', 'proust',
               'twain', 'woolf', 'zola']
    AIDX = {a: i for i, a in enumerate(AUTHORS)}

    eval_ = [json.loads(l) for l in open(REPO / "splits" / "eval.jsonl", encoding="utf-8")]

    # resume support
    done_ids = set()
    if PROGRESS.exists():
        for line in open(PROGRESS, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["passage_id"])
            except Exception:
                pass
    print(f"{len(done_ids)} already done; {len(eval_) - len(done_ids)} to go", flush=True)

    cm = np.zeros((14, 14), dtype=int)
    t0 = time.time()
    n_new = 0
    with open(PROGRESS, "a", encoding="utf-8") as pf:
        for k, r in enumerate(eval_):
            pid = f"{r['language']}/{r['author']}/{r['passage_id']}"
            if pid in done_ids:
                continue
            enc = tok(r["text"], truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1)
                logits = pooled @ W.T + b
            pred = AUTHORS[int(logits.argmax())]
            cm[AIDX[r["author"]], AIDX[pred]] += 1
            n_new += 1
            pf.write(json.dumps({"passage_id": pid, "true": r["author"],
                                 "pred": pred, "probs_idx": int(logits.argmax())}) + "\n")
            if n_new % 25 == 0:
                pf.flush()
                print(f"  {n_new} scored in {time.time()-t0:.0f}s", flush=True)
            del enc, out, mask, pooled, logits
            gc.collect()

    # rebuild full confusion from progress file (covers resumed runs)
    cm = np.zeros((14, 14), dtype=int)
    for line in open(PROGRESS, encoding="utf-8"):
        d = json.loads(line)
        cm[AIDX[d["true"]], AIDX[d["pred"]]] += 1

    acc = np.trace(cm) / max(1, cm.sum())
    per = {a: {"correct": int(cm[i, i]), "total": int(cm[i].sum()),
               "accuracy": round(float(cm[i, i] / max(1, cm[i].sum())), 3)}
           for i, a in enumerate(AUTHORS)}
    print(f"\nfull-705 accuracy: {acc:.4f}")
    print("flaubert:", per["flaubert"], "zola:", per["zola"])
    np.savetxt(REPO / "results" / "confusion_cls_v8_705.csv", cm, fmt="%d",
               delimiter=",", header=",".join(AUTHORS))
    (REPO / "results" / "cls_v8_confusion_705.json").write_text(
        json.dumps({"method": "cls_v8_full_705",
                    "accuracy": round(float(acc), 4),
                    "n_scored": int(cm.sum()),
                    "per_author": per,
                    "flaubert_errors_to_zola": int(cm[AIDX['flaubert'], AIDX['zola']]),
                    "runtime_note": "CPU, batch 1, max_input_len=256 (training setting)"},
                   indent=2), encoding="utf-8")
    print("wrote results/cls_v8_confusion_705.json")


if __name__ == "__main__":
    main()
