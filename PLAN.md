# PLAN — Gemma 4 Developer Agent (2026-10-22 → 2026-12-02)

Budget: **8–10 h/week** (7 courses + PhD apps). Total ≈ 50 h. One submission/day, each scoring run can take up to 12 h — so the calendar, not the code, is the constraint. Freeze early.

## Hard dates (all 23:59 UTC; Chicago = UTC−6 after Nov 1)

| Date | What | Note |
|---|---|---|
| Thu 2026-10-22 | **Accept Kaggle rules today** | Unlocks dataset; costs 2 minutes |
| Fri 2026-11-13 | MATH3010 Exam 2 | **No Kaggle work Nov 9–13** |
| Thu 2026-11-12 | Paper Track deadline 23:59 UTC = 17:59 Chicago | Reconsidered — see `PAPER.md`; submit by **Nov 11**, decision gate Nov 2, hard cap ~12 h total, exam prep wins any conflict |
| Wed 2026-11-25 | Entry + team merger deadline | After this: no joining, no team changes |
| Mon 2026-12-01 | WashU / UIUC / GT PhD deadlines | Kaggle freeze must be *before* this |
| **Wed 2026-12-02** | **Final submission deadline 23:59 UTC = 17:59 Chicago** | Select up to 2 Final Submissions |

Submission math: from Nov 26 → Dec 2 there are only ~6 submission slots, and a run can queue/score for 15 h+. **Last realistic new submission: Sun Nov 29.** Nov 30–Dec 2 is only for selecting finals from already-scored runs.

## Week by week

### Week 1 — Oct 22–28 (8 h) · Baseline or nothing
- [ ] Accept rules; copy official starter notebook; run unchanged on L4x4; submit baseline #1
- [ ] Log baseline in `docs/experiments.md`: public score, total wall time, min/task
- [ ] Model decision test: run 20 dev tasks on 31B vs 12B — record pass rate + min/task for each; pick one (default 12B) and write the reason in experiments.md
- [ ] Read `HARNESS_README.md` end to end; list every harness tool in `docs/harness-notes.md`
- Exit criterion: a scored baseline exists. Everything else is optional.

### Week 2 — Oct 29–Nov 4 (9 h) · Harness loop v1 (no training yet)
- [ ] Implement the debug-loop skill: localize (code graph/search tools) → write failing repro → patch → run relevant tests → repair once
- [ ] Tighten `eval_config.yaml`: per-task time/tool-call caps so one stuck task can't eat the 12 h budget
- [ ] Prompt v2 in `agent/prompts/`: minimal-diff rule, "run the failing test before and after", no unrelated refactors
- [ ] Submit #2 and #3 on different days; compare against baseline
- Exit criterion: harness-only version beats baseline on dev split (target: +5–10 pts on dev).

### Week 3 — Nov 5–8 only (5 h, short week) · Trajectory collection
- [ ] Run `scripts/collect_trajectories.py` on all 129 dev tasks, n=4 samples/task
- [ ] Keep only trajectories whose patch passes the task tests AND is minimal (multipath-consistent filter from ThinkPRM: keep tasks where ≥2 independent samples converge to an equivalent passing patch — those are the trustworthy SFT examples)
- [ ] Build SFT dataset (JSONL) + stats: tasks covered, pass rate, avg diff size
- [ ] **Nov 9–13: STOP. Exam week.**

### Week 4 — Nov 14–20 (10 h) · LoRA SFT v1
- [ ] `scripts/train_lora_sft.py`: LoRA on the *unquantized* Gemma 4 variant of the chosen size, PEFT `.safetensors` output into `agent/adapters/`
- [ ] Ship adapter behind the same harness; A/B on dev split: harness-only vs harness+LoRA
- [ ] Submit the better one (#4/#5); keep the loser as fallback final candidate
- Exit criterion: one scored submission with LoRA beats harness-only on public LB, or LoRA is dropped with a written reason.

### Week 5 — Nov 21–25 (9 h) · Verifier / best-of-n
- [ ] `scripts/verifier_select.py` logic into the agent: generate up to n patches only when time budget allows, execute tests, pick smallest passing patch
- [ ] Calibration tweak (ThinkPRM angle): if no patch passes its tests, submit the patch with the fewest failing tests / smallest diff — never submit empty
- [ ] **Nov 25: entry/team gate — confirm membership is final**
- [ ] Submit #6 by Nov 25
- Exit criterion: candidate A (safe) and candidate B (aggressive) both scored.

### Week 6 — Nov 26–29 (6 h) · Freeze
- [ ] Nov 26–27: at most one final experiment, only if it's a strict improvement on dev
- [ ] **Sun Nov 29: last new submission.** After this, no code changes.
- [ ] Nov 30–Dec 1: PhD deadlines — Kaggle is closed. Select the 2 Final Submissions from scored runs: slot 1 = best public score (12B main config); slot 2 = the **E4B small-model version** if its public score is within ~10 pts (it unlocks the $10k Small Model Bonus pool), otherwise the safer main config
- [ ] Dec 2 17:59 Chicago: deadline passes with finals already selected. Done.

## Risk register

| Risk | Mitigation |
|---|---|
| Scoring queue >15 h near deadline | Freeze Nov 29; never plan a Dec 1–2 submission |
| 31B too slow (timeout = 0 on unfinished tasks) | Week-1 timing gate; default 12B |
| LoRA adapter shaped for wrong model (starter's dummy adapters are 31B-shaped) | If using 12B, delete starter adapters; train fresh |
| L4x4 burns GPU quota at 2× | Dev experiments on smaller splits; full 129-task runs max 2×/week |
| Exam + PhD crunch | Plan already removes Nov 9–13 and Nov 30–Dec 1 |
| Zip validation failure | `build_submission.py` checks root agent.yaml, symlinks, extensions, 3 GiB before every upload |

## What success looks like (realistic)

- Minimum: scored, reproducible submission + public repo + writeup you can link in PhD SOPs / interviews as an agentic post-training project.
- Good: top-25% public leaderboard.
- Stretch: private-leaderboard money (top 3). Treat it as a lottery ticket, not the plan.
