#!/usr/bin/env python3
"""Collect patch trajectories on the 129 public dev tasks and build SFT data.

Idea (ThinkPRM multipath-consistent filtering):
- For each dev task, sample n independent agent runs.
- Execute the task's validation tests for each candidate patch.
- Keep a task for SFT only if >=2 independent runs converge to an equivalent
  PASSING patch (high-trust examples); keep single-pass tasks in a separate
  'silver' file, and failures out entirely.

This is a skeleton: plug run_agent() / run_tests() into the competition
harness (see HARNESS_README.md in the Kaggle dataset) inside a Kaggle
notebook, where the repos and sandbox exist.

Output: data/sft_gold.jsonl, data/sft_silver.jsonl — one JSON object per
trajectory: {"task_id", "repo", "issue", "messages", "patch", "tests_passed"}
"""
import argparse, json, hashlib
from pathlib import Path

def patch_signature(patch: str) -> str:
    """Crude equivalence: normalize whitespace, hash the diff body."""
    lines = [l.strip() for l in patch.splitlines() if l.strip() and not l.startswith(("+++", "---", "@@"))]
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="dev tasks dir from the Kaggle dataset")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    raise SystemExit(
        "Skeleton — implement run_agent()/run_tests() against the competition "
        "harness inside a Kaggle notebook (repos/sandbox are not in this repo). "
        "Filtering rule is implemented in patch_signature() + PLAN.md Week 3."
    )

if __name__ == "__main__":
    main()
