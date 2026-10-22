# Gemma 4 Developer Agent — Kaggle

Qiwei Li's starter repo for the **Gemma 4 Developer Agent Competition** (Google DeepMind / Kaggle).

**Goal:** Post-train an open Gemma 4 model into an offline coding agent that fixes hidden Python bugs — scored SWE-bench style: a task passes only if its validation tests pass after your patch.

## Verified facts (checked 2026-10-22, see `docs/competition-facts.md`)

| Item | Value |
|---|---|
| Main deadline | **Wed 2026-12-02 23:59 UTC** (17:59 Chicago) |
| Entry / team-merger gate | **Wed 2026-11-25 23:59 UTC** — accept rules before this or you cannot submit |
| Prize pool | **$110,000 total** (official Kaggle page, corrected 2026-10-22) — main track $65,000 (37k / 18k / 10k) + Paper Track $35,000 + **$10,000 Small Model Bonus** (highest-ranked E2B/E4B submission; must be one of your 2 final submissions) |
| Paper Track | Closes Thu 2026-11-12 23:59 UTC — **reconsidered 2026-10-22, see `PAPER.md`** (narrow scope, hard decision gate Nov 2; it is non-archival, so the writeup can be reused elsewhere) |
| Hidden tasks | ~120 Python bug-fix tasks, public/private split 50/50; private decides prizes |
| Dev set | 129 public tasks from `fastapi`, `rich`, `requests`, `httpx`; dataset 22.42 GB |
| Allowed models (pick exactly one) | `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct` |
| Submission | `submission.zip` (≤3 GiB) built by a Kaggle notebook at `/kaggle/working/submission.zip`, `agent.yaml` at zip root |
| Scoring budget | 12 hours total for all tasks (incl. sandbox + vLLM startup, excl. patch validation); **1 submission/day** |
| Hardware | Kaggle L4x4 (4×L4, 96 GB total), internet OFF, GPU quota charged at 2× rate |
| Context | vLLM `max_model_len=32768` in starter notebook |

## Strategy in one paragraph

Don't try to out-train Google. Win on **harness + verifier + trajectory quality**:
1. Strong ADK agent loop first (localize → reproduce → patch → run tests → repair), no LoRA.
2. Generate multiple patch trajectories on the 129 dev tasks, keep **only test-passing, minimal-diff** trajectories (multipath-consistent filtering — same idea as ThinkPRM SFT v2).
3. LoRA-SFT on those filtered trajectories (PEFT, `.safetensors`), one adapter per role if useful.
4. Calibrated best-of-n patch selection at inference: run the task's tests, prefer the smallest passing patch; use confidence to decide when to spend a retry.

**Small-model bonus play:** reserve Final Submission slot #2 for an **E4B** version of the same harness if it scores respectably on dev — it competes for the extra $10,000 bonus pool (far fewer entrants) while staying eligible for everything else. The harness must therefore stay model-agnostic from day 1.

**Default model decision (to confirm in Week 1):** start with **12B**. Budget math: 12 h / 120 tasks = 6 min/task average. The 31B is stronger per call but risks timing out and leaves no room for retries; the 12B + verifier + retry loop is the better expected-score play. Switch to 31B only if Week-1 timing on 20 dev tasks shows <4 min/task at a clearly higher pass rate.

## Repo layout

```
agent/                  # the actual submission payload (zipped by scripts/build_submission.py)
  agent.yaml            # root ADK agent config
  eval_config.yaml      # per-task limits (time, tool calls)
  configs/sampling.yaml
  prompts/              # system + repair prompts
  skills/debug-loop/    # SKILL.md for the reproduce-fix-verify loop
  adapters/             # PEFT LoRA adapters go here (one folder per adapter)
  sub_agents/           # optional sub-agent YAMLs (agent_tool)
notebooks/              # Kaggle notebook workflow notes
scripts/
  build_submission.py   # validate + zip agent/ -> submission.zip (checks: root agent.yaml, no symlinks, allowed exts, ≤3 GiB)
  collect_trajectories.py  # run agent on dev tasks, keep passing trajectories as SFT data
  train_lora_sft.py     # PEFT LoRA SFT skeleton (train on unquantized Gemma 4 variant)
  verifier_select.py    # best-of-n patch selection with test execution + calibrated scoring
docs/
  competition-facts.md  # every deadline/rule with its source
  checklist.md          # submission-day checklist
PLAN.md                 # week-by-week plan (8–10 h/week, avoids 11/13 exam + 12/1 PhD deadlines)
data/                   # NOT committed — Kaggle dataset is downloaded inside Kaggle notebooks
```

## First 30 minutes

1. Go to the competition page on Kaggle, **Join + Accept Rules now** (unlocks the dataset; the 11/25 gate is hard).
2. Copy the official Getting Started notebook (Kaggle Staff, Ryan Holbrook), run it **unchanged** once on L4x4, submit that baseline.
3. Record baseline: public score, wall time, time/task — put it in `docs/experiments.md`.
4. Then start Week 1 in `PLAN.md`.

## Rules that silently kill entries

- Mixing two base models, or a sub-agent on a different model — not allowed; every agent uses the same chosen model (different LoRA adapters per agent are allowed).
- `../` paths or symlinks in the zip; files with disallowed extensions; zip > 3 GiB.
- Internet assumptions: evaluation is offline. No API calls, no pip installs at scoring time — everything ships in the zip / offline wheelhouse.
- Private code/data sharing outside your team is a breach; public sharing only via the competition forums/notebooks.
- Winners must hand over training + inference code and reproducible docs — keep this repo tidy from day 1.
