#!/usr/bin/env python3
"""Best-of-n patch selection — the P3 in-session verifier, offline reference.

The scored agent is declarative YAML, so the policy actually runs as prompt rules in
agent/prompts/system.md (Phases L, C/V, S). This module is the executable reference
for that prompt and reads its numbers and shell commands FROM the prompt, so the two
cannot drift:
- load_pacing(): K_MAX and the phase fractions (PACING block, the only place they live)
- load_commands(): the CHECK_CANDIDATE / APPLY_CANDIDATE command templates
- score_candidate() / select_best(): the Phase S ranking
- budget_fraction() / plan_k(): pacing from get_status
- run_p3_session(): the root agent's workflow against a harness object (tests drive it
  with a fake harness backed by a real git sandbox)

Phase S ranking (score_candidate): eligible (non-empty diff touching library code; the
harness strips test/config hunks before grading) > reproduction passes > more tests
passed > smaller diff > earlier candidate. Test failures are not a separate key: every
candidate runs the same test command, so more passed means fewer failed unless a
candidate breaks collection, which already shows as fewer passed.

Pacing signal: get_status time fields (agent_elapsed_seconds / max_time_minutes), because
time is the binding constraint in every logged run (samples average 93-99% of the
4-minute cap) while tool calls are not. Tool-call counts are only the fallback when the
time fields are missing.

Design decisions (0006):
- The agent cannot run this file: the submission dir is not mounted in the task sandbox,
  and an embedded helper script would cost the 12B model ~1 min of output tokens to write
  out. So Phase S is three plain ranking rules in the prompt, mirrored here and tested.
- Candidates share the single task sandbox. CHECK_CANDIDATE saves each candidate's diff to
  /tmp/cand_<K>.patch, verifies that working tree in the same command (one tool call per
  candidate), then resets /workspace; APPLY_CANDIDATE re-applies the winner before submit.
- K = 1 is today's flow done by the root itself (no patch_writer cold start), per the spec's
  "single-candidate fast path".
- The command timeouts (30 s repro + 60 s tests) stay under the harness's 120 s command cap.
- Unverified risk for the A/B: whether sub-agent LLM calls count toward max_turns (60).
  Estimated ~30 turns per K = 3 session, so it should fit either way.

Legacy contract kept: Candidate / select_patch and the CLI self-test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parent.parent
SYSTEM_MD = ROOT / "agent" / "prompts" / "system.md"
CONFIG_FILES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg", ".pytest.ini",
                "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}
TEST_DIRS = {"tests", "test", "testing"}


# ------------------------------------------------------------------ legacy API

@dataclass
class Candidate:
    patch: str
    tests_total: int
    tests_failed: int

    @property
    def diff_size(self) -> int:
        return diff_size(self.patch)

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


# ------------------------------------------------------------- P3 selection

def diff_size(patch: str) -> int:
    return sum(1 for l in patch.splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))


def changed_files(patch: str) -> list[str]:
    files = []
    for l in patch.splitlines():
        m = re.match(r"diff --git a/(\S+) b/(\S+)", l)
        if m and m.group(2) not in files:
            files.append(m.group(2))
    return files


def is_protected(path: str) -> bool:
    """Files the harness strips from agent patches before grading (HARNESS_README §8.1)."""
    p = Path(path)
    name = p.name
    if name in CONFIG_FILES or name.endswith(".pth"):
        return True
    if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")
                                 or any(part in TEST_DIRS for part in p.parts[:-1])):
        return True
    return False


def _size(rec: dict[str, Any]) -> int:
    return int(rec["diff_size"]) if rec.get("diff_size") is not None else diff_size(rec.get("patch", ""))


def _files(rec: dict[str, Any]) -> list[str]:
    return list(rec["files"]) if rec.get("files") is not None else changed_files(rec.get("patch", ""))


def is_eligible(rec: dict[str, Any]) -> bool:
    return _size(rec) > 0 and any(not is_protected(f) for f in _files(rec))


def repro_passed(rec: dict[str, Any]) -> bool:
    r = rec.get("repro")
    return r is True or (isinstance(r, str) and r.lower() == "pass")


def score_candidate(rec: dict[str, Any]) -> tuple:
    """Larger is better: (eligible, repro pass, tests passed, -diff size, -candidate index)."""
    return (is_eligible(rec), repro_passed(rec), int(rec.get("tests_passed") or 0), -_size(rec),
            -int(rec.get("k") or 0))


def select_best(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best eligible candidate, verified or not (best effort); None if nothing is eligible."""
    eligible = [r for r in records if is_eligible(r)]
    return max(eligible, key=score_candidate) if eligible else None


# --------------------------------------------------------------- prompt parsing

def load_pacing(path: Path = SYSTEM_MD) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"```text\n# PACING\n(.*?)```", text, re.S)
    if not m:
        raise ValueError(f"no PACING block in {path}")
    out: dict[str, float] = {}
    for line in m.group(1).splitlines():
        if "=" in line:
            k, v = (s.strip() for s in line.split("=", 1))
            out[k] = int(v) if v.isdigit() else float(v)
    return out


def load_commands(path: Path = SYSTEM_MD) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip() for m in re.finditer(r"```bash\n# (\w+)\n(.*?)\n```", text, re.S)}


def fill(template: str, k: int, test_command: str = "") -> str:
    return template.replace("<K>", str(k)).replace("<TEST_COMMAND>", test_command)


# ---------------------------------------------------------------------- pacing

def budget_fraction(status: dict[str, Any]) -> tuple[float, str]:
    """(fraction of the per-task budget used, signal name) from a get_status response."""
    total = float(status.get("max_time_minutes") or 0) * 60
    if total > 0 and status.get("agent_elapsed_seconds") is not None:
        return float(status["agent_elapsed_seconds"]) / total, "time"
    if total > 0 and status.get("time_seconds_remaining") is not None:
        return 1 - float(status["time_seconds_remaining"]) / total, "time"
    calls = status.get("max_tool_calls")
    if calls:
        return float(status.get("tool_calls_used") or 0) / float(calls), "tool_calls"
    return 0.0, "none"


def plan_k(f: float, pacing: dict[str, float]) -> int:
    return 1 if f > pacing["SINGLE_PATH_AT"] else int(pacing["K_MAX"])


# ------------------------------------------------------------------ check output

def parse_check_output(text: str, k: int, repro_known: bool = True) -> dict[str, Any]:
    """Parse CHECK_CANDIDATE output (DIFF_LINES, file list, REPRO, pytest summary line)."""
    lines = text.splitlines()
    m = re.search(r"DIFF_LINES=(\d+)", text)
    files, in_files = [], False
    for l in lines:
        if l.startswith("DIFF_LINES="):
            in_files = True
            continue
        if l.startswith("REPRO="):
            break
        if in_files and l.strip():
            files.append(l.strip())
    rm = re.search(r"REPRO=(PASS|FAIL)", text)
    after = text[rm.end():] if rm else text
    passed = sum(int(x) for x in re.findall(r"(\d+) passed", after))
    failed = sum(int(x) for x in re.findall(r"(\d+) (?:failed|errors?)\b", after))
    repro = (rm.group(1).lower() if rm else None) if repro_known else None
    return {"k": k, "diff_size": int(m.group(1)) if m else 0, "files": files, "repro": repro,
            "tests_passed": passed, "tests_failed": failed}


# --------------------------------------------------------------- reference root

STRATEGIES = (
    "Strategy 1, minimal: make the smallest change at the line where the wrong behavior happens.",
    "Strategy 2, root cause: fix the place where the bad value or state is first produced, "
    "even if that is a different function.",
    "Strategy 3, defensive: validate or normalize the input at the public entry point named in the issue.",
)


class Harness(Protocol):
    def localize(self) -> dict[str, str]: ...
    def get_status(self) -> dict[str, Any]: ...
    def patch_writer(self, request: str) -> str: ...
    def run_command(self, command: str) -> str: ...
    def root_patch(self) -> None: ...
    def submit_patch(self) -> dict[str, Any]: ...


def run_p3_session(h: Harness, test_command: str, repro_known: bool = True,
                   pacing: dict[str, float] | None = None, commands: dict[str, str] | None = None) -> dict:
    """The root agent's workflow from system.md, step for step."""
    pacing = pacing or load_pacing()
    commands = commands or load_commands()
    info = h.localize()                                                   # Phase L
    f, signal = budget_fraction(h.get_status())
    k_plan = plan_k(f, pacing)
    trace: dict[str, Any] = {"signal": signal, "f_after_localize": f, "k": k_plan, "records": [],
                             "stopped_early": False, "selected": None, "fallback": False}
    if k_plan == 1:                                                       # fast path
        h.root_patch()
        trace["submitted"] = h.submit_patch()
        return trace
    for k in range(1, k_plan + 1):                                        # Phase C/V
        if k > 1 and budget_fraction(h.get_status())[0] >= pacing["VERIFY_STOP"]:
            trace["stopped_early"] = True
            break
        h.patch_writer(f"{STRATEGIES[(k - 1) % len(STRATEGIES)]}\nBUG: {info.get('bug', '')}\n"
                       f"LOCATION: {info.get('location', '')}\nREPRO: python /tmp/repro.py")
        out = h.run_command(fill(commands["CHECK_CANDIDATE"], k, test_command))
        trace["records"].append(parse_check_output(out, k, repro_known))
    best = select_best(trace["records"])                                  # Phase S
    if best is None:
        trace["fallback"] = True
        h.root_patch()
    else:
        trace["selected"] = best["k"]
        h.run_command(fill(commands["APPLY_CANDIDATE"], best["k"]))
    trace["submitted"] = h.submit_patch()
    return trace


if __name__ == "__main__":
    demo = [Candidate("@@\n+x\n+y\n+z", 10, 0), Candidate("@@\n+x", 10, 0),
            Candidate("@@\n+x", 10, 3)]
    best = select_patch(demo)
    assert best is demo[1], best  # smallest passing patch wins
    recs = [{"k": 1, "diff_size": 1, "files": ["pkg/a.py"], "repro": "fail", "tests_passed": 9},
            {"k": 2, "diff_size": 4, "files": ["pkg/a.py"], "repro": "pass", "tests_passed": 3},
            {"k": 3, "diff_size": 1, "files": ["tests/test_a.py"], "repro": "pass", "tests_passed": 10}]
    assert select_best(recs)["k"] == 2  # verified beats more tests; test-only is ineligible
    p = load_pacing()
    assert 0 < p["LOCALIZE_SOFT"] < p["SINGLE_PATH_AT"] < p["VERIFY_STOP"] < 1 and p["K_MAX"] >= 1, p
    assert {"CHECK_CANDIDATE", "APPLY_CANDIDATE"} <= set(load_commands())
    print("verifier_select self-test OK")
