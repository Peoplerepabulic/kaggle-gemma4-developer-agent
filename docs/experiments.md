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
