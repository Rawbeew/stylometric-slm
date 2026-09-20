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

## Entry index
- `2026-09-20_1400_infra_decisions.md` — Cloudflare tunnel, GH repo, GCP project setup
- `2026-09-20_1430_colab_attempts.md` — First Colab runs, wget-404 bug, training script hardening
- `2026-09-20_1500_tpu_setup.md` — TPU creation, libtpu missing, torch_xla version dance
- `2026-09-20_1600_tpu_training.md` — Real training attempts, OOM diagnostics, final state
- `2026-09-20_1700_decision_and_next.md` — Path forward, validation strategy
- `2026-09-20_1730_pivot_to_cpu_vm.md` — User decision ($5 budget, more RAM), n2-highmem-8 deploy
