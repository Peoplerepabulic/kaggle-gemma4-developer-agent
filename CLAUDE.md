# CLAUDE.md — Implementation spec for Claude

You are implementing the code for this repo. The human (Qiwei) owns all Kaggle
interactions and final decisions. Read `README.md`, `PLAN.md`, and
`docs/competition-facts.md` first — they are the source of truth. This file
turns them into rules you must not break and tasks with acceptance criteria.

## The competition in 6 lines

- Kaggle: **Gemma 4 Developer Agent Competition** — post-train Gemma 4 into an
  OFFLINE coding agent that fixes hidden Python bugs (~120 tasks, SWE-bench
  style: a task passes only if its validation tests pass after the patch).
- Main deadline **2026-12-02 23:59 UTC**; entry/team gate **2026-11-25**.
- Score = % tasks passed; the private half of the leaderboard decides prizes.
- Total budget per scored run: **12 hours** for ALL tasks (incl. sandbox and
  vLLM startup), **1 submission/day**.
- Hardware at scoring: Kaggle L4x4 (96 GB VRAM), **internet OFF**.

## Hard rules (violating any of these can zero or disqualify a submission)

1. **Exactly one base model** per submission, chosen from:
   `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`,
   `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`.
   Every agent and sub-agent uses that same model. Different PEFT LoRA
   adapters per agent ARE allowed. Default is **12B**; do not change the model
   without Week-1 timing data logged in `docs/experiments.md`.
   Keep the harness **model-agnostic** (model name is config, never hard-coded
   in logic) — an E4B variant is planned for the $10k Small Model Bonus.
2. **Submission format:** a Kaggle notebook produces `submission.zip` at
   `/kaggle/working/submission.zip`, with `agent.yaml` at the ZIP ROOT.
   Optional folders only: `configs/`, `prompts/`, `sub_agents/`, `skills/`
   (each skill has a `SKILL.md` with YAML frontmatter incl. `name`),
   `adapters/` (one folder per adapter: `adapter_config.json` +
   `adapter_model.safetensors`). Zip ≤ **3 GiB**, **no symlinks**, no `../`
   paths, only allowed file extensions. Validate every change with
   `python scripts/build_submission.py` — it must print OK.
3. **Agent format:** Google ADK Agent Config (YAML). The official starter
   notebook's schema is authoritative: when this repo's templates and the
   starter disagree on a key name, the starter wins — fix the template.
   Note: for the 12B model, vLLM needs the `hf-overrides` setting that the
   starter notebook already includes; the starter's dummy LoRA adapters are
   31B-shaped and must be removed for any other model size.
4. **Tools:** only harness-provided tools may be called:
   `run_command, submit_patch, get_status, read_file, edit_file, write_file,
   get_code_neighbors, search_similar_code, get_code_subgraph`, plus
   `run_skill_script` / `load_skill_resource` for skills, and custom
   sub-agents via `agent_tool`. No other tools, no shell escapes outside
   `run_command`, no code execution outside the sandbox.
5. **Offline:** no network calls, no API keys, no pip installs, no model
   downloads at inference or scoring time. Everything ships in the zip or the
   competition's offline wheelhouse. Context limit: vLLM `max_model_len` is
   32768 — prompts and tool outputs must be budgeted accordingly; truncate
   long test output, never dump whole files blindly.
6. **Time discipline:** the harness reads ONLY the `evaluation:` mapping of
   `agent/eval_config.yaml` and runs exactly one agent session per task —
   there is no harness-level retry or best-of-n, and no view of the global
   12 h budget from inside a session (`get_status` reports per-task time
   only). So: per-task `evaluation.max_time_minutes` × ~120 tasks + sandbox
   setup + vLLM startup must stay under 12 h, and any best-of-n must fit
   inside one task's session budget (see P3). (Reworded 2026-10-22 per
   Claude's patch-0002 proposal, approved by Qiwei.)
7. **Sharing:** never suggest posting this code anywhere except the
   competition's own Kaggle forums/notebooks (private cross-team sharing is a
   rules breach). The GitHub repo is the human's decision, not yours.

## Coding requirements

- Python 3.12, type hints on public functions, no new heavy dependencies
  (scored environment is offline; local training deps live in
  `requirements.txt` only).
- Every script must run `--help` without GPU/torch imports (lazy-import heavy
  deps inside `main()`, as the existing skeletons do).
- Tests: `scripts/verifier_select.py` has a self-test — keep it passing.
  Add a self-test or `tests/` case for anything you add to `scripts/`.
- Never commit: datasets, `*.safetensors` weights, `submission.zip`, `.env`,
  API keys/tokens (see `.gitignore`). SFT data (JSONL) is generated, not
  committed.
- Log every scored or dev run in `docs/experiments.md` (date, model, LoRA,
  harness version, dev pass rate, public LB if submitted, wall time,
  min/task). No silent experiments.
- Agent behavior contract (enforced by `agent/prompts/system.md` — edit the
  prompt, not around it): reproduce before patching; smallest possible diff;
  never weaken/delete tests; max 2 repair rounds; never submit an empty
  patch when a plausible minimal fix exists.

## Tasks, in priority order (acceptance criteria included)

### P0 — Valid baseline packaging
**Status 2026-10-22: DONE** — validation half merged (patch 0001); first
Kaggle run 2026-10-22 (collect smoke) proved the harness compiles and runs
our `agent/` and reads its `evaluation:` budgets live. Schema half (patch 0002): `agent/` rewritten to the official
sample_submission schema + HARNESS_README (tools list, bare model alias,
`generate_content_config`, `evaluation:` budgets, `sub_agents/code_analyzer.yaml`);
the validator now rejects unknown agent / generation / eval keys, and
`tests/fixtures/starter_agent.yaml` (official file, verbatim) passes it.
Not yet proven: that the harness compiles our `agent/` — first notebook run will show.
`agent/` template + `scripts/build_submission.py` validate cleanly, and the
template's keys match the official starter notebook's `agent.yaml` schema.
✅ Done when: build script prints OK and a diff against the starter schema
shows no unknown keys.

### P1 — Trajectory collection (finish `scripts/collect_trajectories.py`)
**Status 2026-10-22: implemented + smoke-tested on Kaggle** (patch 0002;
smoke run in docs/experiments.md). Trace capture (ATIF → messages) VERIFIED
(48 messages on the smoke task). Uses the starter's Evaluator (Phase 2
verification = run_tests), resumable raw log, held-out split
(sha256(id) % 4 == 0) excluded by default. Still unverified: multi-sample
concurrency behavior at n=4 (smoke ran n=1) — watch the first 30-task run.
Smoke finding: 40 tool calls is too few when `code_analyzer` is used heavily
(task localized the bug but never patched) — tune budgets/prompt next.
Implement `run_agent()` / `run_tests()` against the competition harness
(inside a Kaggle notebook, where repos/sandbox exist). Sample n=4 runs per
dev task (129 public tasks: fastapi, rich, requests, httpx), execute each
candidate patch's tests, classify:
- **gold**: ≥2 independent runs converge to an equivalent passing patch
  (equivalence via the existing `patch_signature()`);
- **silver**: exactly 1 passing run; **drop**: everything else.
✅ Done when: a run over ≥30 dev tasks produces `sft_gold.jsonl` /
`sft_silver.jsonl` with the documented schema and a printed summary
(tasks covered, pass rate, avg diff size).

### P2 — LoRA SFT (finish `scripts/train_lora_sft.py`)
Wire the HF Trainer loop: inputs = issue + trajectory messages, labels only
on patch-producing turns (mask the rest). Train on the UNQUANTIZED twin of
the chosen QAT base; export PEFT adapter to `agent/adapters/sft_v1/`.
✅ Done when: training runs end-to-end on the gold set, adapter files exist,
checkpoints/weights are NOT committed, and a dev-split A/B (harness-only vs
harness+LoRA) row is logged in `docs/experiments.md`.

### P3 — Verifier integration
Move `verifier_select.select_patch` policy into the agent flow: among
candidate patches prefer (a) all-tests-passing, then (b) smallest diff;
if none passes, fewest failing tests, then smallest diff. Gate extra samples
on remaining time per `eval_config.yaml`.
**In-session only (per rule 6):** the harness offers no outer retry loop, so
implement this inside one task session — e.g. the root agent invokes a
patch-writer sub-agent (agent_tool) up to N times, runs the targeted tests
itself via `run_command`, applies the selection policy, and submits the
winner. `scripts/verifier_select.py` stays the offline reference
implementation of the policy.
✅ Done when: dev-split run with verifier ≥ verifier-off run, logged.

### P4 — E4B small-model variant
Same harness, `model:` swapped to E4B (+ its own LoRA only if time allows).
✅ Done when: it packages cleanly and has a dev score logged; it exists to
fill Final Submission slot #2 for the $10k Small Model Bonus.

## Out of scope for you

- Submitting anything to Kaggle, accepting rules, team merges, selecting
  Final Submissions — the human does these.
- The Paper Track writeup text (see `PAPER.md`) — you may be asked to produce
  experiment tables/figures for it, nothing more.
- Changing deadlines, the model decision, or this file's hard rules — propose
  it in `docs/experiments.md` notes and let the human decide.
