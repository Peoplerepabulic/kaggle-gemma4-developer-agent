# PAPER.md — Paper Track plan (reconsidered 2026-10-22)

Status: **CONDITIONAL GO.** The paper is worth doing only in a narrow scope,
reusing experiments the main track already needs. Hard decision gate below.

## Facts

- Separate Kaggle competition: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper
  (join + accept its rules separately from the main track)
- Deadline: **Thu 2026-11-12 23:59 UTC = 17:59 Chicago** — the day before
  MATH3010 Exam 2 (Nov 13). Our internal deadline is **Nov 11**.
- Format: Kaggle Writeup, **≤3,000 words** — title/subtitle, abstract,
  introduction, research description (methods + experiments), related works
  with citations. Optional: public notebook in Project Links, or an
  arXiv-ready PDF instead of the Writeup body.
- **Drafts are not judged** — the Writeup must be Submitted, not just saved.
- Prizes: $35,000 total — Best Paper $15,000, Best New Resource $10,000,
  Best New Application $10,000. Judged on novelty / quality / relevance /
  verifiability / clarity (20% each).
- **Non-archival:** entering does not block submitting the work elsewhere —
  the same material can feed the ThinkPRM workshop paper.
- Topics suggested by the host: Tuning & Optimization (PEFT/RL for SWE
  agents), Code Comprehension, Tasks & Benchmarks, Graph Reasoning.

## Recommended angle (one only)

**"Multipath-consistent trajectory filtering + calibrated verifier selection
for small SWE agents"** — i.e., exactly the main track's P1–P3 pipeline,
framed as research:
- Method 1: keep SFT trajectories only when ≥2 independent samples converge
  to an equivalent test-passing patch (quality over quantity).
- Method 2: at inference, select patches by test execution with a
  minimal-diff prior and a failing-test-count fallback (calibration angle
  from ThinkPRM).
- Why it's credible: both claims are testable on the 129 public dev tasks
  with the harness we must build anyway. No extra experiments invented for
  the paper.

Alternative if P1 data is weak: a **Tasks & Benchmarks** resource paper —
release the dev-task trajectory dataset + analysis as the "new resource"
(targets the $10k Best New Resource award, lighter experiments).

## Word budget (3,000 max)

Abstract 200 · Intro 450 · Method 800 · Experiments 900 (tables don't count
toward prose discipline — keep them tight) · Related work 400 · Limitations
+ conclusion 250.

## Timeline (cap: ~12 h total; exam prep always wins)

| When | Work |
|---|---|
| Oct 29–Nov 1 | Experiments ARE Week-2/3 main-track work (harness A/B, trajectory stats) — log with the paper in mind |
| **Sun Nov 2** | **DECISION GATE:** if there is no dev-split result table by today (baseline vs +filtering vs +verifier), DROP the paper. No heroics. |
| Nov 3–6 | Draft from the experiment logs (Claude can draft; Qiwei reviews) |
| Nov 7–8 | Polish, citations, optional public notebook cleanup |
| Nov 9–11 | Buffer only. **Submit by Nov 11 17:59 Chicago.** Nov 12 is exam eve — no paper work. |

## Risks

- Stealing MATH3010 time in exam week → mitigated by the Nov 2 gate + Nov 11 internal deadline.
- Weak results → a clean negative/ablation result is still submittable; judges score verifiability and clarity, not only gains. If results are embarrassing AND the draft isn't done by Nov 8, drop it.
- Disclosure: the Writeup and any attached private notebook become public after the deadline. Nothing about other projects goes in.
