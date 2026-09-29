# Reproducibility Journal — Stylometric mT5 Fine-Tune

**Purpose:** Verified, timestamped log of every command, output, error, fix, and decision during the stylometric-mt5 training run. Designed to enable step-by-step reproduction.

**Source-of-truth rules:**
1. Every command is quoted verbatim.
2. Every output (stdout/stderr) is captured as-is, with exit codes.
3. Every "fixed by" line cites the exact patch/file change.
4. Cross-checking: for any claim "X worked", the corresponding output snippet must be present in this journal.
5. The LLM-as-judge verifier at the end scans this journal against the final repo state.

**Naming:** `YYYY-MM-DD_HHMM_topic.md` for each entry.

**Validation pass:** At the end, all commands in this journal must be re-runnable in sequence on a fresh GCP project to reproduce the trained model.

---

## Entry index (13 entries)

### Training-track entries (in chronological order)

1. `2026-09-20_1400_infra_decisions.md` — Cloudflare tunnel, GH repo, GCP project setup
2. `2026-09-20_1500_tpu_setup.md` — TPU creation, libtpu missing, torch_xla version dance
3. `2026-09-20_1730_pivot_to_cpu_vm.md` — User decision ($5 budget, more RAM), n2-highmem-8 deploy. **Cost note:** the $5 budget + 12h estimate were corrected in entry 11 to $47.52 / 88h actual.
4. `2026-09-20_1810_data_integrity_snag.md` — 530 contaminated passages from wrong PG URLs; re-pulled from correct IDs; 717 eval passages
5. `2026-09-20_1945_front_matter_leak.md` — Front-matter copyright headers leaking into eval split; gradient explosion; cleanup + re-split
6. `2026-09-21_0958_v6_results.md` — v6 encoder baseline numbers; showed 71% accuracy, prompted the encoder swap
7. `2026-09-21_1040_v7_classification_fix.md` — `MT5EncoderModel` + linear head swap (the working architecture)
8. `2026-09-21_1525_v8_save_fix.md` — Tied-weight safetensors error; pickle workaround (`safe_serialization=False` + `torch.save`). **See entry 13 addendum: pickles later replaced with safetensors.**
9. `2026-09-21_1530_v8_progress.md` — v8 retrain live progress
10. `2026-09-21_1625_v8_success.md` — v8 final: 91.2% full / 88.6% per-author sampled. **See addendum at bottom for safetensors replacement.**
11. `2026-09-24_comprehensive_analysis.md` — Full project summary, cost reconciliation ($47.52 actual), all-snags tally

### Post-release entries

12. (reserved — VM termination + budget closeout — see entry 11 §9)

13. `2026-09-28_safetensors_reupload.md` — Multi-scanner cross-check on the v8 pickle save (HF Picklescan/VirusTotal/JFrog clean; Protect AI/ClamAV false positives); re-uploaded with safetensors; same weights, no retraining. **This is the entry the HF model card "Security" section references.**

## How to read this journal

For reproduction: read entries 1–11 in order; entry 11 is the comprehensive summary. The method-paper draft at `paper/drafts/encoder_only_beats_seq2seq_v2.md` references entries 3–11.

For the HF card cross-reference: entry 13 explains the safetensors-only state on the HF Hub repo.

For budget / cost: ignore any number before entry 11. Entry 11 §2 is authoritative.

## Cross-checking

`results/llm_judge_report.json` is the LLM-as-judge audit pass: every "this worked" claim in the journal has its corresponding output snippet quoted in the entry where the claim was made, verified by an independent model.