# Baseline runbook — run the official starter, unchanged, once

Goal: one scored baseline + timing data for the 31B-vs-12B decision.
Rules already accepted on both tracks (done 2026-10-22). Time needed:
~20 min of your attention; the scored run itself can take hours in queue —
you do NOT need to watch it.

## Steps (Kaggle web UI)

1. Open the main competition → **Code** tab → find the official
   **Getting Started** notebook by Kaggle Staff (Ryan Holbrook) →
   **Copy and Edit**.
2. In your copy: right sidebar **Settings / Session options**:
   - Accelerator: **GPU L4 x4**
   - Internet: **OFF**
   - Persistence: files only (default is fine)
3. Confirm inputs are attached (right sidebar **Input**): the competition
   dataset + the Gemma 4 model the starter references. **Correction
   (verified 2026-10-22): starter v3 actually defaults to the 12B QAT model**
   (`MODEL_NAME='gemma-4-12b-it-qat-w4a16-ct'` in cell 2; 31B is only the
   sample_submission default) — leave it as is for the baseline run. The
   baseline submitted 2026-10-22 is therefore a **12B** baseline despite its
   version name `v1-baseline-31b-starter`.
4. **Run All.** Don't edit any cell. The notebook: installs the offline
   wheelhouse → starts vLLM → Phase 1 generates patches on sample tasks →
   Phase 2 verifies → packages `/kaggle/working/submission.zip`.
   - If a cell fails, copy the FULL error text + cell number and send it to
     Muse before re-running — failed runs still burn session time.
5. When it finishes: **Save Version → Save & Run All (Commit)**, then
   **Submit** that version to the competition from the notebook's output /
   the competition **Submissions** tab.
6. Record in `docs/experiments.md` (or just send these to Muse):
   - Public leaderboard score
   - Total wall time of the scoring run (Submissions tab shows it)
   - Any queue wait you noticed
   - Notebook version name (rename it `v1-baseline-31b-starter`)

## Then: the model timing test (same day if quota allows)

L4x4 burns GPU quota at **2× rate** — do this test on a *subset*, not full runs.

1. Copy your baseline notebook → in the model cell, switch to the **12B**
   QAT model. Remember: 12B needs the `hf-overrides` vLLM setting (already in
   the starter) and the starter's 31B-shaped dummy LoRA adapters must be
   removed (the starter does this — verify the cell ran).
2. Limit the run to **20 dev tasks** (use the notebook's sample-task setting;
   do not run all 129).
3. Record per model, in `docs/experiments.md`:
   tasks attempted / passed, total minutes, **minutes per task**.
4. Decision rule (from PLAN.md): pick 12B unless 31B shows **<4 min/task**
   AND a clearly higher pass rate on the same 20 tasks. Budget math:
   12 h ÷ ~120 hidden tasks = 6 min/task average, including vLLM startup.

## Pitfalls from experience

- Kaggle draft sessions can hang at "Starting Draft Session" — if it sits
  >15 min, stop the session and start a fresh one rather than waiting.
- One submission per day, and you must wait for the previous one to finish
  scoring — today's baseline uses today's slot; don't submit a second
  version today "just to test".
- Don't run the full 129-task dev set on L4x4 casually; subset runs only,
  full runs max ~2×/week (quota).
