# Competition facts — sources checked 2026-10-22

Primary guide (built from the official Kaggle pages, starter notebook and staff posts, checked 2026-10-17):
https://reconscribe.com/gemma-4-developer-agent-competition-2026/

Official site: arcprize-style entry is on Kaggle — search "Gemma 4 Developer Agent Competition" on kaggle.com/competitions (accept rules there; this also unlocks the dataset).

- Main track deadline Wed 2026-12-02 23:59 UTC; entry/team-merger gate Wed 2026-11-25 23:59 UTC; Paper Track Thu 2026-11-12 23:59 UTC.
- Prize pool **$110,000** (official Kaggle competition page, fetched 2026-10-22): main $65,000 (37k/18k/10k), Paper $35,000 (15k/10k/10k), plus **Small Model Bonus $10,000** for the highest-ranking submission using Gemma E2B or E4B — eligible only if that submission is one of the team's two selected finals; LoRA allowed; cross-eligible for main + paper prizes; same deadlines. Main page: https://www.kaggle.com/competitions/gemma-4-developer-agent — Paper Track: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper (Writeup ≤3,000 words: title/subtitle, abstract, intro, research description incl. methods+experiments, related works+citations; optional public notebook or arXiv-ready PDF via Project Links; drafts/unsubmitted writeups are NOT judged; submission is non-archival).
- Task: post-train Gemma 4 into an offline coding agent fixing hidden Python bugs; ~120 hidden tasks, public/private halves; dev set 129 tasks (fastapi, rich, requests, httpx); dataset 22.42 GB, Apache-2.0; includes repo snapshots, code graphs, embeddings, offline wheels, Docker sandbox spec, sample_submission.
- Submission: Google ADK Agent Config; root `agent.yaml`; optional `configs/`, `prompts/`, `sub_agents/`, `skills/` (SKILL.md each), `adapters/` (PEFT LoRA, adapter_config.json + adapter_model.safetensors); `eval_config.yaml` for per-task limits. Zip ≤3 GiB, no symlinks, allowed extensions only, no `../`.
- Models: exactly one of the four QAT variants (31B / 12B / E4B / E2B `*-it-qat-w4a16-ct`). All agents/sub-agents share it; LoRA adapters may differ per agent. 12B needs the vLLM `hf-overrides` setting already in the starter notebook. Starter dummy LoRA adapters are 31B-shaped and are removed by the starter if you switch size.
- Harness tools: run_command, submit_patch, get_status, read_file, edit_file, write_file, get_code_neighbors, search_similar_code, get_code_subgraph; skills via run_skill_script / load_skill_resource.
- Scoring: SWE-bench style PASS/FAIL per task (patch applied, validation tests run); score = % tasks passed; private leaderboard decides prizes; ties → earlier submission wins. 12 h total per run incl. sandbox + vLLM startup, excl. patch validation. 1 submission/day, must wait for previous to finish scoring. Early-Oct staff note: queues exceeded 15 h at one point (fixed + pending limit added) — do not submit on the final day.
- Compute: Kaggle L4x4 notebooks only for this competition, internet disabled, GPU quota at 2× rate; vLLM max_model_len 32768.
- Eligibility: 18+, worldwide except Crimea/DNR/LNR/Cuba/Iran/North Korea and sanctioned persons; teams ≤5, one team per person; no private cross-team code/data sharing.
- Winners must deliver training code, inference code, compute-environment description and reproducible documentation.
