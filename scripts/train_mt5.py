"""
train_mt5.py — fine-tunes google/mt5-base on multilingual authorship attribution.

Hardened version (Sept 20 2026):
- TPU/XLA support via torch_xla when available
- Deterministic sampling for --max-train/--max-eval (uniform across authors)
- Checkpoint resume on preemption
- Push-to-hub wrapped in try/except so partial failures don't kill script
- Reproducible via --seed
- --dry-run flag for 5-step validation without committing full run

Usage:
  # CPU smoke test (1 epoch, 32 rows, 5 steps only)
  python scripts/train_mt5.py --epochs 1 --max-train 32 --max-eval 16 --dry-run

  # CPU validation (small but full pass)
  python scripts/train_mt5.py --epochs 1 --max-train 200 --max-eval 50

  # GPU box (full run)
  python scripts/train_mt5.py --epochs 5 --push-to user/repo

  # TPU VM (full run, XLA)
  python scripts/train_mt5.py --epochs 5 --push-to user/repo
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Avoid tokenizer/datasets fork-bomb on small VMs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "splits"
RESULTS_DIR = REPO_ROOT / "results"
PREFIX = "classify authorship: "


def detect_device():
    """Return ('cuda'|'cpu'|'xla', device_object)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.device("cuda")
    except ImportError:
        pass

    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        import torch_xla  # noqa: F401
        # If we got here without ImportError, XLA is installed.
        # xm.xla_device() will fail loudly if no TPU is actually present.
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return "xla", device
        except Exception as e:
            print(f"[warn] torch_xla installed but xla_device() failed: {e}", file=sys.stderr)
    except ImportError:
        pass

    import torch
    return "cpu", torch.device("cpu")


def uniform_sample(rows, n, seed=0):
    """Take n rows uniformly across authors, not just the first n."""
    if n is None or n >= len(rows):
        return rows
    by_author = defaultdict(list)
    for r in rows:
        by_author[r["author"]].append(r)
    rng = random.Random(seed)
    out = []
    keys = sorted(by_author.keys())
    per_author = max(1, n // len(keys))
    for k in keys:
        pool = by_author[k]
        take = min(per_author, len(pool))
        out.extend(rng.sample(pool, take))
    # If we undershot due to short authors, top up from the remainder
    if len(out) < n:
        remaining = [r for r in rows if r not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: n - len(out)])
    rng.shuffle(out)
    return out[:n]


def load_data(args):
    """Returns (train_rows, eval_rows). Handles both local + HF Hub."""
    if args.dataset_repo:
        from datasets import load_dataset
        ds = load_dataset(args.dataset_repo, token=os.environ.get("HF_TOKEN"))
        # Try both 'eval' and 'test' split names since we've used both across versions
        if "train" in ds:
            train_rows = list(ds["train"])
        else:
            train_rows = list(ds[list(ds.keys())[0]])
        eval_rows = list(ds.get("eval", ds.get("test", [])))
    else:
        train_path = SPLITS_DIR / "train.jsonl"
        eval_path = SPLITS_DIR / "eval.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(
                f"{train_path} missing. Run scripts/pull_gutenberg.py + scripts/build_split.py first, "
                f"or pass --dataset-repo user/repo to load from HF Hub."
            )
        train_rows = [json.loads(l) for l in train_path.open(encoding="utf-8")]
        eval_rows = [json.loads(l) for l in eval_path.open(encoding="utf-8")]

    if args.max_train:
        train_rows = uniform_sample(train_rows, args.max_train, seed=args.seed)
    if args.max_eval:
        eval_rows = uniform_sample(eval_rows, args.max_eval, seed=args.seed + 1)

    return train_rows, eval_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/mt5-base")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-input-len", type=int, default=1024)
    ap.add_argument("--max-target-len", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-eval", type=int, default=None)
    ap.add_argument("--push-to", default=None)
    ap.add_argument("--dataset-repo", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="Run 5 training steps only, skip eval/push. For pipeline validation.")
    ap.add_argument("--resume-from", default=None,
                    help="Path to checkpoint to resume from (for preemption recovery)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    token = os.environ.get("HF_TOKEN")
    if args.push_to and not token:
        print("HF_TOKEN required for --push-to", file=sys.stderr)
        sys.exit(1)

    # ---- Device ----
    device_kind, device_obj = detect_device()
    print(f"[device] {device_kind}")

    # ---- Data ----
    train_rows, eval_rows = load_data(args)
    print(f"[data] train={len(train_rows)}, eval={len(eval_rows)}")

    if args.dry_run:
        args.epochs = 1
        # Force tiny subset even if not specified
        if not args.max_train:
            args.max_train = 32
        if not args.max_eval:
            args.max_eval = 16

    # ---- Model ----
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )

    print(f"[model] loading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model, token=token)
    print(f"[model] params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ---- Tokenize ----
    from datasets import Dataset

    def preprocess(batch):
        inputs = [PREFIX + t for t in batch["text"]]
        targets = batch["author"]
        model_inputs = tokenizer(
            inputs, max_length=args.max_input_len,
            truncation=True, padding="max_length",
        )
        labels = tokenizer(
            targets, max_length=args.max_target_len,
            truncation=True, padding="max_length",
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    train_cols = train_ds.column_names
    eval_cols = eval_ds.column_names
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_cols)
    eval_ds = eval_ds.map(preprocess, batched=True, remove_columns=eval_cols)
    print(f"[tokenize] train={len(train_ds)}, eval={len(eval_ds)}")

    # ---- Output dir ----
    out_dir = args.output_dir or str(REPO_ROOT / "results" / "mt5-stylometric")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ---- Training args ----
    bf16 = device_kind in ("cuda", "xla")
    fp16 = device_kind == "cpu"  # cpu can't do bf16/fp16, this is just to suppress the warning

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=10 if args.dry_run else 50,
        save_strategy="steps" if args.dry_run else "epoch",
        save_steps=5 if args.dry_run else 500,
        save_total_limit=2,
        eval_strategy="no" if args.dry_run else "epoch",
        bf16=bf16,
        fp16=False,  # bf16 is sufficient; fp16 causes loss scaling issues on TPU
        push_to_hub=bool(args.push_to),
        hub_model_id=args.push_to,
        hub_token=token,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2 if device_kind != "xla" else 0,  # XLA workers need special setup
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds if not args.dry_run else None,
        data_collator=data_collator,
        processing_class=tokenizer,  # modern API; falls back if older transformers
    )

    # ---- Train ----
    print(f"[train] starting: epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}, lr={args.lr}")
    if args.resume_from:
        print(f"[train] resuming from {args.resume_from}")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from)
    except KeyboardInterrupt:
        print("[train] interrupted — checkpoint saved to output_dir")
        sys.exit(130)

    if args.dry_run:
        print("[dry-run] complete — pipeline validated, no eval/push performed")
        return

    # ---- Save final ----
    print(f"[save] final model to {out_dir}/final")
    trainer.save_model(out_dir + "/final")

    # ---- Push to Hub (wrapped, so failure doesn't kill script) ----
    if args.push_to:
        try:
            trainer.push_to_hub(commit_message="mt5-base fine-tuned on multilingual authorship attribution")
            print(f"[push] OK: https://huggingface.co/{args.push_to}")
        except Exception as e:
            print(f"[push] FAILED: {e}", file=sys.stderr)
            print(f"[push] local copy is at {out_dir}/final — upload manually later")

    # ---- Per-author per-language accuracy ----
    print("[eval] computing per-author-per-language accuracy...")
    correct = defaultdict(int)
    total = defaultdict(int)
    model.eval()
    device = model.device  # works for cuda/cpu/xla since Trainer places model
    with torch.no_grad():
        for ex in eval_rows:
            inp = tokenizer(
                PREFIX + ex["text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_len,
            ).to(device)
            out = model.generate(**inp, max_length=args.max_target_len)
            pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            key = (ex["language"], ex["author"])
            total[key] += 1
            if pred == ex["author"]:
                correct[key] += 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "mt5_results.json"
    out_path.write_text(json.dumps({
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
    }, indent=2))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
