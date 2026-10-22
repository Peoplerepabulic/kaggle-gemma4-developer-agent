# Submission checklist

## Today (Oct 22)
- [x] Kaggle: Join competition + Accept Rules — **done 2026-10-22, BOTH tracks (main + paper)**
- [ ] Solo confirmed (teams ≤5 optional; merger deadline Nov 25)
- [ ] Baseline: follow `docs/baseline-runbook.md` — starter notebook unchanged, submit, log score + timing

## Every submission
- [ ] `python scripts/build_submission.py` passes (root agent.yaml, no symlinks, ≤3 GiB)
- [ ] Exactly one base model everywhere, incl. sub-agents
- [ ] No internet/API assumptions; no leftover starter dummy adapters for the wrong model size
- [ ] Ran the notebook end-to-end on L4x4 with internet OFF
- [ ] Logged the run in docs/experiments.md (date, config, public score, wall time)
- [ ] Waited for previous submission to finish scoring (1/day limit)

## Freeze (by Sun Nov 29)
- [ ] Last new submission sent
- [ ] Two Final Submissions selected (best public + safest config)
- [ ] Training + inference code tidy in this repo (winners must hand it over)
