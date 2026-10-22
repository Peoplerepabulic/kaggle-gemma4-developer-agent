#!/usr/bin/env python3
"""Best-of-n patch selection — the inference-time verifier.

Policy (calibrated, ThinkPRM-style):
1. If any candidate patch passes all of the task's runnable tests, pick the
   one with the SMALLEST diff among the passing set (minimal-diff prior).
2. If none fully passes, pick the patch with the fewest failing tests;
   tie-break by smallest diff. Never return an empty patch when a candidate
   exists.
3. Only spend extra samples while the global time budget allows (the caller
   passes time_remaining_min; see agent/eval_config.yaml retry section).
"""
from dataclasses import dataclass

@dataclass
class Candidate:
    patch: str
    tests_total: int
    tests_failed: int

    @property
    def diff_size(self) -> int:
        return sum(1 for l in self.patch.splitlines()
                   if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))

    @property
    def passes(self) -> bool:
        return self.tests_total > 0 and self.tests_failed == 0

def select_patch(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    passing = [c for c in candidates if c.passes]
    if passing:
        return min(passing, key=lambda c: c.diff_size)
    return min(candidates, key=lambda c: (c.tests_failed, c.diff_size))

if __name__ == "__main__":
    demo = [Candidate("@@\n+x\n+y\n+z", 10, 0), Candidate("@@\n+x", 10, 0),
            Candidate("@@\n+x", 10, 3)]
    best = select_patch(demo)
    assert best is demo[1], best  # smallest passing patch wins
    print("verifier_select self-test OK")
