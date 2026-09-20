# Journal Entry 4: Data Integrity Snag (2026-09-20 ~18:10)

## Trigger
User instruction: *"If there are data integrity issues we have to retry before training slm"* + requirement that the journal should load where we hit snags, what snags, how solved, and include where we got all data from.

## What happened
Training of mT5-base was running (PID 3820 on VM, ~3h38m ETA, step 9/465 reached) when the user requested data provenance verification. I ran a forensic audit of the corpus before continuing.

## Snag: training was running on BAD DATA
The corpus pull script (`scripts/pull_gutenberg.py`) embedded a curated list of Project Gutenberg URLs, one or more per (lang, author). Audit showed **7 of 14 author directories got text from wrong Gutenberg IDs** — completely different authors / books / languages than intended.

### The 7 problems

| Directory | Stated URL's actual title | Actual author | Files affected |
|---|---|---|---|
| **it/manzoni** | "Percy Bysshe Shelley" (biography) | John Addington Symonds | 56 passages |
| **fr/zola** | "Taken Alive" (English fiction) | Edward Payson Roe | 113 passages |
| **fr/zola** | pg8609 404 — no Germinal data | — | (Germinal missing) |
| **fr/flaubert** | pg26839 404 — no Salammbô data | — | (Salammbô missing) |
| **en/woolf** | "The Wanderers" (English adventure novel) | Mary Johnston | 121 passages |
| **es/pardo_bazan** | "Women in English Life from Mediæval to Modern Times" (English lit history) | Georgiana Hill (transl.) | 78 passages |
| **es/galdos** | "The Old Man; or, Ravings and Ramblings round Conistone" (English) | Alexander Craig Gibson | 46 passages |

**Total contamination: 530 passages (~18% of 2,987).**

### Why this matters
If we trained on the existing data:
- The model would have learned "Italian" stylometry on Symonds' Victorian English prose
- "Zola" model would have learned E. P. Roe's 19th-century American religious fiction
- "Woolf" model would learn Mary Johnston's adventure-novel writing
- "Spanish authors" would learn English (the language!) — bias-ridden

This would have produced a paper with fundamentally bogus authorship-attribution results — exactly the hallucination/data-integrity risk the user wanted to catch.

## Detection method

### Step 1: Initial suspicion
Spot-checked the first file of each author directory (head 200 chars). Saw obvious mismatches:
- `it/manzoni/passage_0000.txt` opened with `"PERCY BYSSHE SHELLEY BY JOHN ADDINGTON SYMONDS"`
- `fr/zola/passage_0000.txt` opened with `"The Works of E. P. Roe VOLUME ELEVEN TAKEN ALIVE"`

### Step 2: Direct URL check
Issued `curl -sL <url>` on each URL in `pull_gutenberg.py`, grabbed the `Title:` and `Author:` lines from the Gutenberg header. Confirmed which authors the URLs actually serve.

### Step 3: Programmatic audit
Wrote `scripts/audit_corpus_final.py`:
1. Parses `pull_gutenberg.py` URL_INDEX (embedded above) — yes I hardcoded the URLs because the file uses dict literals which are easy to extract; in a future iteration we'd auto-parse
2. For each URL: HEADs the Gutenberg file, parses Title/Author from the PG header
3. Compares declared author to directory name (last-token substring match)
4. Outputs OK / MIXED / WRONG verdict per author dir, plus on-disk file count

### Step 4: Re-run the audit on disk
The full audit is in `results/corpus_audit_final.json`. At runtime:
```
Total passages CLEAN: 2457
Total passages CONTAMINATED: 530
Total: 2987
```

### False positives caught and discarded
Earlier audit iterations (`audit_corpus.py`, `v2.py`, `v3.py`) had noisy heuristics — they flagged authors when their names appeared anywhere in narrative prose (e.g. Dickens quoting "Goethe"). v3 caught 124 "WRONG_DIR" but most were these false positives. The final audit uses URL-level canonical verification, which is binary correct/incorrect.

## Action taken

1. **Killed in-flight training**: `pkill -9 -f train_mt5.py` on VM (PID 3820 was at step 9/465, ETA 3h33m)
2. **Stopped VM (not deleted)**: `gcloud compute instances stop stylometric-trainer`. This stops the billing clock but keeps the disk and IP for when we restart.
3. **Committed audit scripts to repo**: `audit_corpus*.py` + the JSON report.
4. **Pending:** corrected URL list and re-pull.

## Data provenance — where it ALL came from

### Direct sources used (verified):

| Source | Used for | Verified |
|---|---|---|
| **Project Gutenberg** (https://www.gutenberg.org/) | Raw public-domain text for 14 author directories | ✅ Listed in `pull_gutenberg.py` URLs |
| **Hugging Face Hub** (`huggingface.co/api/whoami-v2`) | Auth: identity verification for token | ✅ Returns user name + role:write |
| **Cloudflare Tunnel** (`barbara-themes-forbes-shaved.trycloudflare.com`) | Bridge: serve `.env` keys + dataset to remote compute | ✅ `/health`, `/secret/HF_TOKEN`, `/data/train.jsonl` all working |
| **GCP** (project `orca-503514`) | Compute: TPU v5litepod-1 (deleted) and VM `stylometric-trainer` | ✅ gcloud describe confirms |

### Things we did NOT use:
- No web scraping of bot-walled sources (ctext.org for Chinese is deferred)
- No LLM-generated synthetic data (everything is real, public-domain text)
- No social handles assumed from GitHub username (we whoami'd the HF token to discover the real namespace was "Chaiir")

## Cross-check commands for entry 4
```bash
# 1. Re-run the audit (needs internet for URL probes)
cd /c/Users/alaga/ghwork/stylometric-slm
/c/Users/alaga/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe scripts/audit_corpus_final.py

# 2. Confirm VM is stopped (cost saving)
/c/tmp/gcloud_wrapper.cmd compute instances describe stylometric-trainer \
  --zone=us-west1-c --project=orca-503514 \
  --format="value(name,status)" 2>&1 | tail -1

# 3. Confirm training process gone
ssh -i C:/tmp/gce_key -o UserKnownHostsFile=C:/tmp/vm_hosts \
  alaga@136.118.30.79 "pgrep -f train_mt5.py || echo dead"
```

### Verified results (re-run 2026-09-20 18:15 UTC)
```
=== Cross-check 1: audit re-run ===
... (outputs the same 2457 CLEAN / 530 CONTAMINATED summary)

=== Cross-check 2: VM state ===
stylometric-trainer  STOPPED

=== Cross-check 3: training process ===
dead
```

## Resolution plan
1. **Identify correct Gutenberg URLs** for the 7 affected authors (using the Project Gutenberg search API or manual lookup) ✅ DONE
2. **Update `pull_gutenberg.py`** with the correct URLs ✅ DONE
3. **Re-pull** all author directories — overwrites existing bad data ✅ DONE
4. **Re-audit** to confirm 100% clean ✅ DONE — **0 contaminated of 3592**
5. **Re-split** train/eval (deterministic seed 42) ✅ DONE
6. **Restart VM, re-train** with the corrected corpus ✅ LAUNCHED

This took longer than simply patching the contaminated rows because we wanted a clean re-pull — but it's the only way to be sure no other silent contamination snuck in.

### URL replacements applied

| Directory | Old (wrong) URL | New (verified) URL | Notes |
|---|---|---|---|
| en/woolf | pg57496 (Mary Johnston, EN) | pg1245 Night and Day + pg5670 Jacob's Room | Both verified Virginia Woolf |
| fr/zola | pg8609 (404) + pg5320 (E.P. Roe EN) | pg6497 L'Assommoir FR + pg5711 Germinal FR | Both French original |
| fr/flaubert | pg2413 + pg26839 (404) | pg2413 only (Madame Bovary FR) | Salammbô FR not in PG; dropped |
| es/galdos | pg56462 (A.C. Gibson EN) | pg17013 Fortunata y Jacinta + pg17340 Marianela | Both Spanish |
| es/pardo_bazan | pg49990 (Hill EN) | pg17491 La Tribuna + pg68452 La piedra angular | Both Pardo Bazán, Spanish |
| it/manzoni | pg4555 (Symonds EN, Shelley bio!) | pg45334 I promessi sposi | Italian |

### Bugs found and fixed during re-pull

1. **`strip_gutenberg_boilerplate` was clipping the wrong side of `*** START OF`**
   - Old code kept chars AROUND the marker, throwing away the entire book body
   - Old: `text = text[keep_pre:after+1]` → 863 chars of pre-marker only
   - Fixed: `text = text[nl + 1:]` → full body from after marker onwards
2. **CRLF line endings not handled**
   - PG sometimes returns `\r\n`; `text.find("\n")` missed the LF
   - Fixed: explicit `text.replace("\r\n", "\n").replace("\r", "\n")` first

### Audit output (re-run 2026-09-20 18:30 UTC)
```
Total passages CLEAN: 3592
Total passages CONTAMINATED: 0
Total: 3592
```

Splits regenerated: **2875 train + 717 eval** (vs old 2382/605 because more data per author).

### Re-train status

- **Started**: PID 1339 on VM `stylometric-trainer` (n2-highmem-8, 64GB)
- **Config**: 5 epochs, batch=4, grad_accum=4, seq=512, 1500 train rows (downsampled for budget)
- **Total steps**: 1500/(4×4) = 93 steps/epoch × 5 = **465 steps**
- **Step time observed**: ~35s/step (first 2 steps; 43 then 35 — speeding up as expected)
- **ETA**: ~4.5h
- **Cost**: 4.5 × $0.43 = **~$1.93**
- **Log**: `/home/alaga/train_v4.log`

## Cost impact
- Training killed at step 9/465 → 9 × 28s = ~5 min of waste (~$0.04)
- VM stopped: 1 hour of $0.43 saved immediately, more as we re-pull
- TPU cost on prior failed attempt: ~$0.44
- Total spent so far: ~$1.30 of $5 budget
- Future re-train estimate: ~$1.93
- **Total expected: ~$3.23** of $5 budget (leaves ~$1.77 buffer)
