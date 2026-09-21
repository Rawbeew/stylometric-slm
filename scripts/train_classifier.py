"""
train_classifier.py — Stylometric classification with mT5 encoder + linear head.

Why this exists: v6 (`train_mt5.py`) failed because mT5's span-corruption pretrain
biases the decoder toward emitting `<extra_id_0>` sentinels at inference, never
fully overridden after 5 epochs of fine-tuning. This script keeps mT5's encoder
(small LM, multilingual) but drops the decoder and adds a 14-way classification
head — a standard, well-behaved architecture for authorship attribution.

Reuses `check_leakage`, `load_data`, `detect_device`, `uniform_sample` from
train_mt5.py so we don't fork data-loading logic.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow importing train_mt5 helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mt5 import detect_device, load_data, uniform_sample, check_leakage

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
PREFIX = "classify authorship: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/mt5-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-input-len", type=int, default=512)
    ap.add_argument("--push-to", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    # Use all available CPU cores for matmul ops (CPU is the bottleneck)
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN")
    if args.push_to and not token:
        print("HF_TOKEN required for --push-to", file=sys.stderr)
        sys.exit(1)

    device_kind, device_obj = detect_device()
    print(f"[device] {device_kind}")

    train_rows, eval_rows = load_data(argparse.Namespace(
        dataset_repo=None, max_train=args.max_train, max_eval=args.max_eval, seed=args.seed))
    print(f"[data] train={len(train_rows)}, eval={len(eval_rows)}")
    check_leakage(train_rows, "train")
    check_leakage(eval_rows, "eval")

    if args.dry_run:
        args.epochs = 1
        if not args.max_train:
            args.max_train = 64
        if not args.max_eval:
            args.max_eval = 32

    from transformers import AutoTokenizer, MT5EncoderModel, TrainingArguments, Trainer

    print(f"[model] loading {args.base_model} (encoder only)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    encoder = MT5EncoderModel.from_pretrained(args.base_model, token=token)
    hidden_size = encoder.config.d_model

    # Author label encoding (deterministic order)
    authors = sorted({r["author"] for r in train_rows} | {r["author"] for r in eval_rows})
    author2id = {a: i for i, a in enumerate(authors)}
    id2author = {i: a for a, i in author2id.items()}
    n_classes = len(authors)
    print(f"[model] {n_classes} classes: {authors}")

    import torch.nn as nn

    class Classifier(nn.Module):
        def __init__(self, encoder, hidden, n_classes):
            super().__init__()
            self.encoder = encoder
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, input_ids, attention_mask, labels=None):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            logits = self.head(pooled)
            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

    model = Classifier(encoder, hidden_size, n_classes)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Untie tied weights early so intermediate epoch saves also work.
    # MT5EncoderModel structure: model.encoder.shared ↔ model.encoder.encoder.embed_tokens
    # Both reference the SAME tensor (TiedWeight). We deep-copy both so safetensors
    # doesn't refuse to save (it requires unique memory addresses per parameter).
    shared = model.encoder.shared if hasattr(model.encoder, "shared") else None
    inner = model.encoder.encoder if hasattr(model.encoder, "encoder") else model.encoder
    if hasattr(inner, "embed_tokens"):
        with torch.no_grad():
            inner.embed_tokens.weight = nn.Parameter(
                inner.embed_tokens.weight.detach().clone()
            )
            if shared is not None:
                shared.weight = nn.Parameter(shared.weight.detach().clone())

    from datasets import Dataset

    def preprocess(batch):
        inputs = [PREFIX + t for t in batch["text"]]
        enc = tokenizer(inputs, max_length=args.max_input_len,
                        truncation=True, padding="max_length")
        enc["labels"] = [author2id[a] for a in batch["author"]]
        return enc

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    train_cols = train_ds.column_names
    eval_cols = eval_ds.column_names
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_cols)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_cols)
    print(f"[tokenize] train={len(train_ds)}, eval={len(eval_ds)}")

    out_dir = args.output_dir or str(RESULTS_DIR / "classifier-stylometric")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def collate(batch):
        return {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10 if args.dry_run else 50,
        # No intermediate saves: tied-weight safetensors issue requires manual
        # save at end with safe_serialization=False. Save final state directly.
        save_strategy="no",
        eval_strategy="epoch" if not args.dry_run else "no",
        bf16=device_kind in ("cuda",),
        fp16=False,
        max_grad_norm=1.0,
        push_to_hub=bool(args.push_to),
        hub_model_id=args.push_to,
        hub_token=token,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2 if device_kind != "xla" else 0,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": (preds == labels).mean().item()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not args.dry_run else None,
        data_collator=collate,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"[train] epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}, lr={args.lr}")
    trainer.train()

    if args.dry_run:
        print("[dry-run] complete")
        return

    print(f"[save] final model to {out_dir}/final")
    # Trainer's save_model doesn't accept safe_serialization kwarg in 4.46.0
    # and the tied-weight issue requires us to use pickle. Call save_pretrained
    # directly with safe_serialization=False to write a .bin instead of .safetensors.
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir + "/final", safe_serialization=False)
    else:
        trainer.save_model(out_dir + "/final")
    tokenizer.save_pretrained(out_dir + "/final")

    # Save author vocabulary
    (Path(out_dir) / "final" / "author_vocab.json").write_text(json.dumps({
        "authors": authors, "author2id": author2id,
    }, indent=2, ensure_ascii=False))

    # ---- Per-author per-language accuracy ----
    print("[eval] computing per-author-per-language accuracy...")
    correct = defaultdict(int)
    total = defaultdict(int)
    model.eval()
    device = next(model.parameters()).device

    # Sample 5 per author for fair per-author eval
    by_author = defaultdict(list)
    for r in eval_rows:
        by_author[r["author"]].append(r)
    sampled = []
    per_author_count = 50
    rng = random.Random(args.seed + 2)
    for a in sorted(by_author):
        pool = by_author[a]
        sampled.extend(rng.sample(pool, min(per_author_count, len(pool))))

    with torch.no_grad():
        for ex in sampled:
            inp = tokenizer(
                PREFIX + ex["text"],
                return_tensors="pt", truncation=True, max_length=args.max_input_len,
            ).to(device)
            logits = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])["logits"]
            pred_id = logits.argmax(-1).item()
            pred = id2author[pred_id]
            key = (ex["language"], ex["author"])
            total[key] += 1
            if pred == ex["author"]:
                correct[key] += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "classifier_results.json"
    out_path.write_text(json.dumps({
        "model": "classifier (mT5 encoder + linear head)",
        "per_author_per_lang": {
            f"{lang}/{author}": {
                "correct": correct.get((lang, author), 0),
                "total": total.get((lang, author), 0),
                "accuracy": (
                    correct.get((lang, author), 0) / total.get((lang, author), 1)
                    if total.get((lang, author)) else 0.0
                ),
            }
            for (lang, author) in sorted(total.keys())
        },
        "overall": {
            "correct": sum(correct.values()),
            "total": sum(total.values()),
            "accuracy": sum(correct.values()) / max(1, sum(total.values())),
        },
    }, indent=2, ensure_ascii=False))
    print(f"[eval] wrote {out_path}")

    if args.push_to:
        try:
            trainer.push_to_hub(commit_message="mT5 encoder + linear head for authorship attribution")
            print(f"[push] OK: https://huggingface.co/{args.push_to}")
        except Exception as e:
            print(f"[push] FAILED: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
