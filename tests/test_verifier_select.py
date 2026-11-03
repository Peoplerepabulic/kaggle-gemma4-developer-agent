"""Run: python -m unittest discover -s tests

Scoring/selection unit tests, plus fake-harness sessions: the root workflow
(verifier_select.run_p3_session) runs the exact CHECK_CANDIDATE / APPLY_CANDIDATE
commands from agent/prompts/system.md in a real throwaway git repo, with scripted
patch_writer edits and a scripted clock behind get_status.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import verifier_select as vs  # noqa: E402

PACING = vs.load_pacing()


def rec(k, diff=2, files=("pkg/a.py",), repro="fail", passed=0, failed=0):
    return {"k": k, "diff_size": diff, "files": list(files), "repro": repro,
            "tests_passed": passed, "tests_failed": failed}


class ScoringTests(unittest.TestCase):
    def test_repro_pass_beats_tests_only(self) -> None:
        best = vs.select_best([rec(1, passed=50, repro="fail"), rec(2, passed=1, repro="pass")])
        self.assertEqual(best["k"], 2)

    def test_more_passing_tests_wins_ties(self) -> None:
        best = vs.select_best([rec(1, passed=3, repro="pass"), rec(2, passed=5, repro="pass", diff=9)])
        self.assertEqual(best["k"], 2)

    def test_smaller_diff_wins_final_ties(self) -> None:
        best = vs.select_best([rec(1, diff=6, passed=4, repro="pass"), rec(2, diff=2, passed=4, repro="pass")])
        self.assertEqual(best["k"], 2)

    def test_full_tie_prefers_earlier_candidate(self) -> None:
        self.assertEqual(vs.select_best([rec(1), rec(2)])["k"], 1)

    def test_empty_and_test_only_diffs_are_ineligible(self) -> None:
        empty = rec(1, diff=0, files=(), repro="pass", passed=9)
        tests_only = rec(2, files=("tests/test_a.py", "pkg/conftest.py"), repro="pass", passed=9)
        config_only = rec(3, files=("pyproject.toml",), repro="pass", passed=9)
        self.assertIsNone(vs.select_best([empty, tests_only, config_only]))
        weak = rec(4, passed=0, repro="fail")
        self.assertEqual(vs.select_best([empty, tests_only, config_only, weak])["k"], 4)

    def test_mixed_test_and_library_change_is_eligible(self) -> None:
        self.assertTrue(vs.is_eligible(rec(1, files=("tests/test_a.py", "pkg/a.py"))))

    def test_all_unverified_best_effort(self) -> None:
        recs = [rec(1, passed=2, diff=5), rec(2, passed=4, diff=7), rec(3, passed=4, diff=3), rec(4, repro=None, passed=1)]
        self.assertEqual(vs.select_best(recs)["k"], 3)

    def test_unknown_repro_ranks_like_fail(self) -> None:
        self.assertEqual(vs.select_best([rec(1, repro=None, passed=3), rec(2, repro="fail", passed=2)])["k"], 1)

    def test_score_from_raw_patch(self) -> None:
        patch = ("diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
                 "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
                 "@@ -1 +1 @@\n-y\n+z\n")
        r = {"k": 1, "patch": patch, "repro": "pass", "tests_passed": 1}
        self.assertEqual(vs.changed_files(patch), ["pkg/a.py", "tests/test_a.py"])
        self.assertEqual(vs.score_candidate(r), (True, True, 1, -4, -1))

    def test_is_protected_matches_harness_rules(self) -> None:
        for p in ("tests/test_x.py", "src/pkg/tests/helpers.py", "test_x.py", "pkg/x_test.py", "conftest.py",
                  "a/pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "x.pth"):
            self.assertTrue(vs.is_protected(p), p)
        for p in ("pkg/a.py", "fastapi/sse.py", "pkg/testing_utils.py", "docs/tests.md"):
            self.assertFalse(vs.is_protected(p), p)

    def test_legacy_select_patch_contract(self) -> None:
        demo = [vs.Candidate("@@\n+x\n+y", 10, 0), vs.Candidate("@@\n+x", 10, 0), vs.Candidate("@@\n+x", 10, 3)]
        self.assertIs(vs.select_patch(demo), demo[1])
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "verifier_select.py")],
                             capture_output=True, text=True)
        self.assertEqual(res.stdout.strip(), "verifier_select self-test OK", res.stderr)


class PacingTests(unittest.TestCase):
    def test_constants_live_only_in_the_prompt(self) -> None:
        self.assertEqual(set(PACING), {"K_MAX", "LOCALIZE_SOFT", "SINGLE_PATH_AT", "VERIFY_STOP"})
        self.assertTrue(0 < PACING["LOCALIZE_SOFT"] < PACING["SINGLE_PATH_AT"] < PACING["VERIFY_STOP"] < 1)
        hits = [p for p in (ROOT / "agent").rglob("*") if p.is_file() and "K_MAX =" in p.read_text()]
        self.assertEqual([p.relative_to(ROOT).as_posix() for p in hits], ["agent/prompts/system.md"])

    def test_time_signal_first_then_tool_calls(self) -> None:
        st = {"agent_elapsed_seconds": 60, "max_time_minutes": 4.0, "tool_calls_used": 60, "max_tool_calls": 70}
        self.assertEqual(vs.budget_fraction(st), (0.25, "time"))
        self.assertEqual(vs.budget_fraction({"time_seconds_remaining": 180, "max_time_minutes": 4}), (0.25, "time"))
        self.assertEqual(vs.budget_fraction({"tool_calls_used": 35, "max_tool_calls": 70}), (0.5, "tool_calls"))

    def test_plan_k(self) -> None:
        self.assertEqual(vs.plan_k(PACING["SINGLE_PATH_AT"], PACING), PACING["K_MAX"])
        self.assertEqual(vs.plan_k(PACING["SINGLE_PATH_AT"] + 0.01, PACING), 1)

    def test_prompt_strategies_match_reference(self) -> None:
        text = vs.SYSTEM_MD.read_text()
        for s in vs.STRATEGIES:
            self.assertIn(s, text)
        self.assertEqual(len(vs.STRATEGIES), PACING["K_MAX"])

    def test_check_command_fits_harness_command_timeout(self) -> None:
        import yaml

        budget = yaml.safe_load((ROOT / "agent" / "eval_config.yaml").read_text())["evaluation"]
        self.assertEqual(budget, {"max_time_minutes": 4, "max_tool_calls": 70, "max_turns": 60,
                                  "timeout_seconds": 120, "verification_timeout_seconds": 300})
        cmd = vs.load_commands()["CHECK_CANDIDATE"]
        self.assertLess(sum(int(x) for x in re.findall(r"timeout (\d+)", cmd)), budget["timeout_seconds"])
        self.assertNotIn("{", cmd)  # ADK would treat braces in the instruction as state placeholders

    def test_parse_check_output_real_pytest_summary(self) -> None:
        out = "DIFF_LINES=4\npkg/a.py\npkg/b.py\nREPRO=PASS\n1 failed, 12 passed, 2 errors in 0.31s\n"
        self.assertEqual(vs.parse_check_output(out, 2), {"k": 2, "diff_size": 4, "files": ["pkg/a.py", "pkg/b.py"],
                                                         "repro": "pass", "tests_passed": 12, "tests_failed": 3})
        self.assertIsNone(vs.parse_check_output(out, 2, repro_known=False)["repro"])
        self.assertEqual(vs.parse_check_output("DIFF_LINES=0\nREPRO=FAIL\nTraceback\n", 1)["tests_passed"], 0)


# ------------------------------------------------------------------ fake harness

BUGGY = "def clamp(x, lo, hi):\n    return max(lo, x)\n"
RUNNER = '''import sys, traceback
try:
    from pkg.calc import clamp
except Exception:
    traceback.print_exc(); sys.exit(2)
cases = [((5, 0, 10), 5), ((-1, 0, 10), 0), ((20, 0, 10), 10)]
passed = sum(1 for a, want in cases if clamp(*a) == want)
failed = len(cases) - passed
print(f"{failed} failed, {passed} passed in 0.01s" if failed else f"{passed} passed in 0.01s")
sys.exit(1 if failed else 0)
'''
REPRO = "from pkg.calc import clamp\nassert clamp(20, 0, 10) == 10, clamp(20, 0, 10)\n"


def edit(rel, old, new):
    def apply(repo: Path) -> None:
        p = repo / rel
        p.write_text(p.read_text().replace(old, new))
    return apply


FIX_SMALL = edit("pkg/calc.py", "return max(lo, x)", "return min(max(lo, x), hi)")
FIX_BIG = edit("pkg/calc.py", "    return max(lo, x)\n",
               "    if x > hi:\n        return hi\n    if x < lo:\n        return lo\n    return x\n")
WRONG = edit("pkg/calc.py", "return max(lo, x)", "return max(lo, x) if x < hi else hi - 1")
BREAK = edit("pkg/calc.py", "return max(lo, x)", "return max(lo, x")
TEST_ONLY = edit("tests/run.py", "((20, 0, 10), 10)", "((20, 0, 10), 20)")
NOOP = edit("pkg/calc.py", "zzz-not-there", "")

CAN_RUN = all(shutil.which(x) for x in ("git", "bash", "timeout"))


class FakeHarness:
    """Sandbox = a real git repo; /workspace and /tmp in commands are rewritten into it."""
    COST = {"localize": 40, "get_status": 2, "patch_writer": 25, "run_command": 8, "root_patch": 20}

    def __init__(self, writers, localize_s=40, writer_s=25, repro=True, root_edit=FIX_SMALL, time_fields=True):
        self.base = Path(tempfile.mkdtemp())
        self.repo, self.tmp, self.bin = self.base / "workspace", self.base / "tmp", self.base / "bin"
        for d in (self.repo / "pkg", self.repo / "tests", self.tmp, self.bin):
            d.mkdir(parents=True)
        (self.repo / "pkg" / "__init__.py").write_text("")
        (self.repo / "pkg" / "calc.py").write_text(BUGGY)
        (self.repo / "tests" / "run.py").write_text(RUNNER)
        if repro:
            (self.tmp / "repro.py").write_text(REPRO)
        os.symlink(sys.executable, self.bin / "python")
        self.env = {**os.environ, "PYTHONPATH": str(self.repo), "PATH": f"{self.bin}:{os.environ['PATH']}",
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                    "GIT_COMMITTER_EMAIL": "t@t", "PYTHONDONTWRITEBYTECODE": "1"}
        self.sh("git init -q && git add -A && git commit -qm baseline")
        self.writers, self.root_edit, self.time_fields = list(writers), root_edit, time_fields
        self.COST = {**self.COST, "localize": localize_s, "patch_writer": writer_s}
        self.clock, self.calls, self.log, self.dirty_at_writer = 0.0, 0, [], []
        self.submitted = None

    def close(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)

    def sh(self, cmd: str) -> str:
        subs = {"/workspace": str(self.repo), "/tmp/": f"{self.tmp}/"}  # one pass: mkdtemp lives under /tmp
        cmd = re.sub(r"/workspace|/tmp/", lambda m: subs[m.group(0)], cmd)
        r = subprocess.run(["bash", "-c", cmd], cwd=self.repo, env=self.env, capture_output=True, text=True)
        return r.stdout + r.stderr

    def tick(self, name: str) -> None:
        self.clock += self.COST[name]
        self.calls += 1
        self.log.append(name)

    def localize(self):
        self.tick("localize")
        return {"bug": "clamp ignores hi", "location": "pkg/calc.py:1-2 clamp"}

    def get_status(self):
        self.tick("get_status")
        st = {"tool_calls_used": self.calls, "max_tool_calls": 70, "max_time_minutes": 4.0}
        if self.time_fields:
            st |= {"agent_elapsed_seconds": self.clock, "time_seconds_remaining": 240 - self.clock}
        else:
            st.pop("max_time_minutes")
        return st

    def patch_writer(self, request: str) -> str:
        self.tick("patch_writer")
        self.dirty_at_writer.append(self.sh("git status --porcelain").strip())
        self.writers.pop(0)(self.repo)
        return "CANDIDATE: done"

    def run_command(self, command: str) -> str:
        self.tick("run_command")
        return self.sh(command)

    def root_patch(self) -> None:
        self.tick("root_patch")
        self.root_edit(self.repo)

    def submit_patch(self):
        self.submitted = self.sh("git add -N . && git diff")
        return {"status": "ok", "patch_size": len(self.submitted)}


@unittest.skipUnless(CAN_RUN, "git/bash/timeout not available")
class FakeHarnessSessionTests(unittest.TestCase):
    TEST_CMD = "python tests/run.py"

    def session(self, h: FakeHarness, **kw):
        self.addCleanup(h.close)
        return vs.run_p3_session(h, self.TEST_CMD, **kw)

    def test_selects_the_verified_candidate(self) -> None:
        h = FakeHarness([WRONG, TEST_ONLY, FIX_SMALL])
        tr = self.session(h)
        self.assertEqual((tr["k"], tr["selected"], tr["signal"]), (3, 3, "time"))
        by_k = {r["k"]: r for r in tr["records"]}
        self.assertEqual((by_k[1]["repro"], by_k[1]["tests_passed"], by_k[1]["tests_failed"]), ("fail", 2, 1))
        self.assertEqual(by_k[2]["files"], ["tests/run.py"])
        self.assertEqual((by_k[3]["repro"], by_k[3]["tests_passed"], by_k[3]["diff_size"]), ("pass", 3, 2))
        self.assertIn("+    return min(max(lo, x), hi)", h.submitted)
        self.assertNotIn("tests/run.py", h.submitted)
        self.assertEqual(h.dirty_at_writer, ["", "", ""])  # every candidate starts from a clean tree
        self.assertEqual(h.log.count("patch_writer"), 3)

    def test_smaller_verified_diff_wins(self) -> None:
        h = FakeHarness([FIX_BIG, FIX_SMALL, WRONG])
        tr = self.session(h)
        self.assertEqual(tr["selected"], 2)
        self.assertGreater(tr["records"][0]["diff_size"], tr["records"][1]["diff_size"])

    def test_all_fail_still_submits_best_effort(self) -> None:
        h = FakeHarness([BREAK, WRONG, NOOP])
        tr = self.session(h)
        self.assertEqual([r["repro"] for r in tr["records"]], ["fail", "fail", "fail"])
        self.assertEqual(tr["records"][2]["diff_size"], 0)
        self.assertEqual(tr["selected"], 2)  # 2 tests passed beats a broken import
        self.assertFalse(tr["fallback"])
        self.assertIn("hi - 1", h.submitted)

    def test_nothing_eligible_falls_back_to_root_edit(self) -> None:
        h = FakeHarness([NOOP, TEST_ONLY, NOOP], root_edit=WRONG)
        tr = self.session(h)
        self.assertIsNone(tr["selected"])
        self.assertTrue(tr["fallback"])
        self.assertIn("hi - 1", h.submitted)
        self.assertNotIn("tests/run.py", h.submitted)

    def test_slow_localization_collapses_k_to_1(self) -> None:
        h = FakeHarness([FIX_BIG, FIX_BIG, FIX_BIG], localize_s=0.55 * 240)
        tr = self.session(h)
        self.assertEqual(tr["k"], 1)
        self.assertGreater(tr["f_after_localize"], PACING["SINGLE_PATH_AT"])
        self.assertNotIn("patch_writer", h.log)
        self.assertIn("root_patch", h.log)
        self.assertIn("min(max(lo, x), hi)", h.submitted)

    def test_verify_stop_ends_candidate_loop(self) -> None:
        h = FakeHarness([WRONG, FIX_SMALL, FIX_BIG], writer_s=0.35 * 240)
        tr = self.session(h)
        self.assertTrue(tr["stopped_early"])
        self.assertEqual(len(tr["records"]), 2)
        self.assertEqual(tr["selected"], 2)
        self.assertIn("min(max(lo, x), hi)", h.submitted)

    def test_unknown_repro_ranks_by_tests(self) -> None:
        h = FakeHarness([WRONG, FIX_BIG, FIX_SMALL], repro=False)
        tr = self.session(h, repro_known=False)
        self.assertEqual([r["repro"] for r in tr["records"]], [None, None, None])
        self.assertEqual(tr["selected"], 3)  # 3 passed; smallest of the two full passes

    def test_tool_call_proxy_when_status_has_no_time(self) -> None:
        h = FakeHarness([WRONG, FIX_SMALL, FIX_BIG], time_fields=False)
        tr = self.session(h)
        self.assertEqual(tr["signal"], "tool_calls")
        self.assertEqual(tr["k"], PACING["K_MAX"])


if __name__ == "__main__":
    unittest.main()
