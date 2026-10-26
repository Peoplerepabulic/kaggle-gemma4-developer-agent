# Experiments log

| Date | Config (model / LoRA / harness) | Dev pass rate | Public LB | Wall time | Min/task | Notes |
|---|---|---|---|---|---|---|
| 2026-10-22 02:30 CDT | **Baseline #1 SUBMITTED** — official starter v3 unchanged. Actual model = **12B** (`gemma-4-12b-it-qat-w4a16-ct`) — starter v3's cell 2 defaults to 12B, NOT 31B (the 31B name is only the sample_submission default; our Kaggle version name `v1-baseline-31b-starter` is therefore a misnomer, kept as submitted). Notebook: https://www.kaggle.com/code/cheeway/getting-started-gemma-4-developer-agent-edcf40 | n/a (starter local sample eval only) | **0.03** (reported by Qiwei 2026-10-22; ≈3% of public hidden tasks passed — the unchanged-starter floor, consistent with the local timeouts) | notebook run 19m09s (1149s, successful, L4x4) | — | Submission entered 2026-10-22 02:30:49 CDT, the day's 1 submission used. Starter's LOCAL sample eval showed 2 agent timeouts → NO_PATCH: `fastapi_15661`, `fastapi_15588` ("Task ... agent timed out", resolved=False) — watch per-task time caps in our harness. submission.zip was 0.00 MiB (config-only payload, expected for starter). |
| — | 31B comparison run (20 dev tasks) | | | | | Now reframed: baseline above IS the 12B datapoint. Optional 31B run only if we want the comparison; decision rule unchanged (PLAN.md Week 1) |
| 2026-10-22 ~11:42 CDT | **Baseline #1 PUBLIC SCORE on leaderboard** — account `Cheeway` shows exactly **1 entry**, public score **0.03**, rank **3248**, "Last 9h" (entry ~02:30–02:45 CDT, matches the 02:30 CDT submission). This is the 12B baseline anchor. Scoring duration on Kaggle side not shown (only leaderboard "Last 9h" elapsed time visible). **Caveat**: My Submissions page not reachable (session not logged into Kaggle; no saved login permitted for kaggle.com), so the score↔v1-baseline-31b-starter link is inferred from account + single entry + timing, not directly verified. Public LB uses ~50% of test data (per Kaggle note). |

## Proposals for Qiwei (Claude, 2026-10-22 — not applied to CLAUDE.md hard rules)

1. **Hard rule 6 references a `retry:` section that the harness cannot read.** The harness only reads `evaluation:` from `eval_config.yaml` (starter Cell 5, HARNESS_README §7.1), runs exactly one agent session per task, and the submission is declarative YAML (no Python). So there is no harness-level retry/best-of-n and no view of the *global* 12 h budget from inside the agent; `get_status` only reports the *per-task* remaining time. The `retry:` block was removed from `eval_config.yaml` in patch 0002. Proposed rewording: "per-task `evaluation.max_time_minutes` × ~120 tasks + setup must stay under 12 h; any best-of-n must fit inside one task's session budget."
2. **P3 consequence:** verifier/best-of-n must be built inside one session — e.g. a root agent that calls a patch-writer sub-agent twice and compares test results — not as outer retries. Worth re-scoping before Week 5.
3. **P2 A/B needs held-out tasks.** `collect_trajectories.py` excludes a deterministic ~25% held-out split by default (`holdout_ids.txt`); report the A/B pass rate on those IDs only, otherwise the LoRA is scored on its own training tasks.
4. **12 h budget is unverified for 4 min/task.** The baseline used the sample's 1 min / 10 calls. Before the first 4-min submission, check the baseline's scoring duration on Kaggle (or the next run's) to confirm setup time per task.

## 2026-10-22 — P1 smoke test on Kaggle (notebook `gemma4-collect-smoke`, L4x4)

- **Transfer route solved:** competition notebooks cannot enable Internet with a
  GPU accelerator (Kaggle error, verbatim in notebook). Code now travels as the
  PRIVATE dataset `cheeway/gemma4-repo` (mount `/kaggle/input/datasets/cheeway/gemma4-repo`);
  update it with a new dataset version when the repo changes.
- **Pipeline VERIFIED end-to-end:** Cell 1 wheelhouse install OK (41 wheels,
  52 s); vLLM started on our declared 12B; the harness read our `evaluation:`
  budgets live (4 min / 40 calls / 60 turns / 120 s / 300 s) — P0's
  "harness compiles our agent/" caveat is now closed. ATIF trace capture
  works: `trace_to_messages` produced 48 messages from
  `collect/results/s0/traces/trace_fastapi_15588.json`.
- **Result:** 1 task × 1 sample (`--limit 1 --n 1`), whole cell 450 s, sample
  183 s. `fastapi_15588` (also a baseline timeout task) **resolved=False,
  diff=0**: the agent exhausted the 40 tool-call budget on exploration
  (heavy `code_analyzer` use), correctly localized the bug
  (`fastapi/sse.py`, planned a validator fix) but never emitted a patch.
  Time was NOT the binding constraint (183 s < 240 s cap) — tool calls were.
- **Tuning implications (for the next patch):** raise `max_tool_calls`
  (40 → ~70) and/or cap `code_analyzer` invocations per task in the prompt
  (≤2) and push earlier patching; keep `max_time_minutes` 4–5. Scale math at
  3.05 min/sample: ~97 train tasks × n=4 ≈ 19–20 GPU-hours single-stream —
  plan on resumable multi-session collection with concurrency.

- 0003 tuning: max_tool_calls 40→70, analyzer caps, first-patch deadline — pending re-smoke on fastapi_15588

## Proposal (Claude, 2026-10-22, with 0003 — not applied)

- **`max_turns: 60` may now bind before `max_tool_calls: 70`.** If a turn = one LLM call (the starter accepts `max_llm_calls` as an alias) and the model makes one tool call per turn, the root agent gets at most ~60 calls, possibly fewer if `code_analyzer`'s LLM calls count too. Unlike the tool-call cap, running out of turns triggers no `budget_warning`, so the agent gets no "submit now" signal. In the re-smoke, compare the trace's LLM-call count with `tool_calls`; if turns are within ~10 of 60 at termination, raise `max_turns` to ~90 (it costs nothing on its own; time stays capped at 4 min).
- **Time will likely be the next binding cap**, not a reason to raise it yet: at the smoke rate (~4.6 s/call) 240 s ≈ 52 calls. Keep 4 min until a scored run's wall time shows setup cost per task; 5 min × 120 = 10 h of agent time leaves only ~2 h for setup + vLLM startup.

## 2026-10-22 — Re-smoke after 0003 (same task, fastapi_15588, dataset v2)

- **Success criterion met: diff=8 > 0.** The agent patched this time — 13 tool
  calls of 70 (no budget termination), 21 messages, 224 s. The patch (938
  chars) adds validators to `ServerSentEvent` in `fastapi/sse.py` — the same
  fix the smoke-1 agent had planned but never wrote.
- **resolved=False**: Phase-2 tests failed (exit 1). The fix idea is wrong or
  incomplete for this task, and with only 13 calls the agent's own VERIFY step
  was thin. Failure mode flipped vs smoke 1 (all-explore-no-patch →
  quick-patch-thin-verify). This task is a known-hard outlier (baseline also
  failed it); do not over-tune on n=1.
- max_turns did not bind (13 calls / 21 msgs). Trace source this run:
  `result.trace` attribute (vs file in smoke 1) — both paths work.
- Next: 30-task collection (--limit 30 --n 4 --concurrency 4) for the first
  real pass-rate / gold-silver yield numbers; est. ~2–3 h wall in one session
  at ~3.7 min/sample. Fix-quality work (deeper verify, verifier selection)
  belongs to P2/P3, informed by that yield.

## 2026-10-23 — 30-task collection COMPLETE (v3-collect-30x4, results read 2026-10-25)

Notebook Output persisted after all (the "no saved version" the user saw was a
display quirk; the run finished and its Output is intact and attachable).

| metric | value |
|---|---|
| tasks covered / samples | 30 / 120 (all 30×4 ran) |
| sample pass rate | **0.0583** (7/120) |
| task pass@4 | **0.0667** (2/30) |
| gold | 2 tasks, **7 rows** (195.56 kB sft_gold.jsonl) |
| silver / divergent | 0 / 0 — every passing sample converged with another |
| avg diff size (kept) | 2 lines |
| avg minutes/sample | **3.98** (≈ the 4-min cap — time IS binding in collection) |
| kept rows missing messages | 0 (all gold rows SFT-ready) |

Readings:
- Concurrency barely helped: ~478 sample-minutes took 6+ wall hours
  (effective parallelism ≈1.3×, sandbox CPU contention). Plan future wall
  time as samples × ~4 min / 1.3.
- 7 gold rows is far too few for P2 SFT. At this rate the full ~97-task
  train split yields only ~20-25 gold rows. Yield must be raised BEFORE
  mass collection: the lever is collection-only budget overrides (the
  collect script's --max-time-minutes; scored-run config stays at 4 min),
  since avg sample time sits at the cap.
- This is the first real dev-set row for the Paper Track table:
  harness-v1.1 (12B, no LoRA) = 2/30 tasks, 5.8% sample pass.

- 0004 P2 SFT implemented (pipeline validation scope) — pending Kaggle smoke on 7 gold rows

## 2026-10-25 — P2 dry-run VERIFIED on Kaggle (notebook gemma4-train-smoke, CPU session)

- Template fidelity: `port == tokenizer.apply_chat_template on all 7 rows` —
  the hand-ported Gemma 4 renderer is byte-exact against the real tokenizer.
- All 7 gold rows tokenize with supervised tokens in the 1–40% band (no WARN):
  total 41,086 tokens, 6,627 supervised (16.13%). Gold tasks: fastapi_14492
  (3 samples) + fastapi_14786 (4 samples). One row truncated (10,330 → 8,015,
  7 steps dropped) per the designed policy. Masked-sample inspection correct
  (system/issue masked; patch turns + final summary trained).
- Env: torch 2.11 / transformers 5.16.1 / peft 0.20 / accelerate 1.14 all in
  the base image (trl absent, not needed). bf16 twin mount:
  /kaggle/input/models/google/gemma-4/transformers/gemma-4-12b-it/2
  (tokenizer MUST point at the /2 subdir).
- **Accelerator constraint discovered:** L4x4 is offered only on
  competition-attached notebooks; a fresh non-competition notebook offers
  only None / T4 x2 / TPU. Stage-2 training therefore runs in a copy of the
  collect notebook (which retains L4x4), with the transformers-variant model
  attached additionally. Internet is available in non-competition notebooks.

## 2026-10-25 — Stage 2: training WORKS; serve test blocked on vLLM CUDA-graph OOM

Notebook gemma4-train-l4 (copy of collect notebook, L4x4). Environment had to
be hand-fixed in-session (starter Cell 1 was not in the copy; a hand-rolled
pip line failed on dependency resolution — the wheelhouse lacks
google-cloud-bigquery-storage, so installs need --no-deps + forced upgrades:
transformers 5.13.1, peft 0.19.1, torchao uninstalled for peft compat).

- **Training (max_len 2048): SUCCESS in 106 s.** 65,568,768 trainable params
  (0.5453%), 12 steps, loss 6.13 → 0.91 (train_loss 2.657). Adapter saved:
  262.4 MB, base google/gemma-4-12b-it, keys text. (At 2048 the set degrades
  to 6 examples / 218 supervised tokens — fine for a mechanics smoke; the
  real run needs the chunked-loss fix in 0005 to train at 8192.)
- **Packaging: SUCCESS.** agent_wired (root `adapter: sft_v1`) validates;
  submission zip 240.8 MB.
- **Serve test: BLOCKED.** vLLM (0.19.1, hand-installed --no-deps after 5
  missing-module rounds) started, model loaded (2.35 GiB/GPU), then OOM in
  CUDA graph capture (Worker TP3, capture_end). The adapter was never
  reached, so key-style is still unverified. Differences vs the working
  collect runs: LoRA enabled, and this session's package versions drifted
  from the clean starter install. Fix path (0005): plumb --enforce-eager /
  --gpu-memory-utilization into collect's VllmConfig for serve tests, and
  re-test in a clean collect-notebook environment with the adapter attached
  as a notebook-output input. Adapter + wired agent preserved in train-l4's
  working dir (to be snapshotted as a saved version).
- Correction to the above: Quick Save does NOT persist /kaggle/working as
  version Output (verified 2026-10-25 — Version 1 of gemma4-train-l4 has 0
  output files; only Save & Run All commits working files). The smoke
  adapter (218 supervised tokens) is a throwaway and will NOT be preserved;
  after 0005 lands, retrain (2 min) + serve test rerun in a clean
  collect-copy environment, committed via Save & Run All.

- 0005: chunked CE loss (default; --loss-mode naive kept) + collect --enforce-eager / --gpu-memory-utilization; Cell 1 verbatim in baseline-runbook — pending re-smoke: train 7 gold rows at --max-len 8192, then serve test
