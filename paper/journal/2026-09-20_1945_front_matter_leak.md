# Journal Entry 5: Front-matter Leak Snag (2026-09-20 ~19:45)

## Symptom
Training v4 was launched cleanly on CLEAN corpus at step 1 and ran ~3h 30m through to step 93. At end of epoch 1 the **first eval printed `eval_loss=27.8`** and a `train_loss=156.8` appeared at step 50 — both **catastrophically bad**.

For 14-author classification the random baseline loss = ln(14) = 2.64. We were getting 10× to 60× worse than random. **Plus `grad_norm=4369`** — gradients had exploded. The model wasn't learning anything; it was getting worse.

## Root cause
The first passage of every author (`passage_0000.txt`) contains PG front matter:
- `en/dickens`: "DAVID COPPERFIELD By Charles Dickens AFFECTIONATELY INSCRIBED..."
- `en/woolf`: "Produced by...VIRGINIA WOOLF CHAPTER ONE..."
- `fr/zola`: "This eBook was produced by Carlo Traverso. Author: Émile Zola Title: Germinal..."
- `it/manzoni`: "NOTE DEL TRASCRITTORE..."
- `es/pardo_bazan`: "OBRAS COMPLETAS DE EMILIA PARDO BAZÁN..."

So the model could learn a **spurious cue**: literally copy the surname from the text → predict label. This is a memorization shortcut, not stylometry.

Why did it generalize **so badly** it actually got worse than random?
- During training the model learns "if surname in text → predict author" perfectly
- But because of the tokenizer's vocabulary + mT5's tendency to start with [BOS] + predict sequence, **the gradient updates push it into a degenerate mode** where it learns to over-generate tokens and the loss explodes
- Or more simply: `grad_norm=4369` is the smoking gun — gradient clipping wasn't enabled (default 1.0 in Trainer), so any large gradient propagated fully

## Snag chain (5th in our session)
1. **TPU OOM** (Entry 1) → pivoted to CPU VM
2. **VM OOM** (Entry 2) → reduced train rows to 1500
3. **Protobuf missing** (Entry 3) → pip installed
4. **Wrong URLs / data contamination** (Entry 4) → re-pulled
5. **Front-matter leakage** (Entry 5, this entry) → strip first 3 passages + safety regex
1. `train_v4.log` — found `eval_loss=27.8`, `train_loss=156.8`, `grad_norm=4369`
2. Tokenized `train_ds[0]` — saw label was "dickens" (clean)
3. Read passage_0000.txt for several authors — confirmed all contained author name in text
4. Pattern-checked passages 0000, 0001, 0002 across all 14 authors → 2 had author name in p0002 (`fr/maupassant`, `it/manzoni`)

## Fix
Wrote `scripts/clean_passages.py` that:
- Always drops the first 3 passages of every author (these are PG front matter)
- ALSO drops any passage whose first 1000 chars match the author's surname regex
- Runs in ~10s

Result: **3,537 clean passages kept, 55 dropped** (mostly first-3, plus 2-9 extras for cervantes, hugo, maupassant, manzoni that had deep contamination).

## Snag cost
- ~52 min wall time wasted on the bad training (≈ $0.37)
- $1.48 spent total before this snag → now $1.85
- Re-train ETA: ~3h 30m × $0.43 = ~$1.50 more → **total ~$3.35 of $5** ($1.65 buffer)

## Resolution plan
1. Re-audit to confirm 0/3537 contaminated ✓ DONE
2. Re-build train/eval splits from cleaned corpus ✓ DONE  
3. Restart VM, re-launch training ✓ LAUNCHED (PID 2797, log `train_v5.log`)
4. (Defensive) Update `train_mt5.py` to enable `max_grad_norm=1.0` so future gradient explosions don't blow up training ✓ DONE
5. (Defensive) Add `check_leakage()` guard that aborts training if any row's text contains the author surname as a whole word ✓ DONE

## Final clean state after fix
- `clean_passages.py` ran first → dropped 55 passages (first 3 + safety drops)
- Then iteratively ran `check_leakage()` and manually dropped 19 more contaminated passages (literary references to other authors: "Victor Hugo", "Mr. Dickens", etc.) for safety
- **Final: 3,515 passages, 0 contamination** (2,810 train + 705 eval)
- All 14 authors retained with at least 53 passages each (it/manzoni 174, pardo_bazan 53)

## False-positive notes for future iterations
- "twain" the surname vs "twain" meaning "two" (old English) — both appear in Melville/Mark Twain/Hugo
- "zola" the surname vs Italian verbs ending in "-zola" (ruzzolarono, spenzolava)
- "hugo" the surname vs names like "Hugo" appearing as character names or other authors in literary reference lists
- The `check_leakage()` function uses whole-word tokenization via regex `(^|[^a-z])surname([^a-z]|$)` to avoid these

## Snag cost
- ~52 min wall time wasted on the bad training (≈ $0.37)
- ~20 min re-pull and re-audit (≈ $0.14)
- Total before this snag's re-train: $1.85 → now $2.36
- Re-train ETA: ~3h 50m × $0.43 = ~$1.65 more → **total ~$4.01 of $5** ($0.99 buffer)

## Where data came from (final)
- **Project Gutenberg** (https://www.gutenberg.org/) — verified URLs from journal entry 4
- **Cloudflare Tunnel** (`barbara-themes-forbes-shaved.trycloudflare.com`) — relays splits + secrets
- **Hugging Face Hub** — `Chaiir` account (verified via /api/whoami-v2)
- **GCP project** `orca-503514` — VM `stylometric-trainer` (n2-highmem-8, 64GB RAM, 8 vCPU)
