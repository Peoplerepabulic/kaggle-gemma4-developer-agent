"""Run: python -m unittest discover -s tests"""
import argparse
import asyncio
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import collect_trajectories as ct  # noqa: E402

P_A = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
P_A2 = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a  =  2 \n"
P_B = "diff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@ -1 +1,2 @@\n-b\n+c\n+d\n"

ATIF = {
    "schema_version": "ATIF-v1.7",
    "steps": [
        {"step_id": 1, "source": "user", "message": "Fix the bug"},
        {"step_id": 2, "source": "agent", "message": "", "reasoning_content": "look first",
         "tool_calls": [{"tool_call_id": "c1", "function_name": "read_file", "arguments": {"filepath": "x.py"}}],
         "observation": {"results": [{"source_call_id": "c1", "content": "a = 1"}]}},
        {"step_id": 3, "source": "agent", "message": "",
         "tool_calls": [{"tool_call_id": "c2", "function_name": "edit_file",
                         "arguments": {"filepath": "x.py", "old_string": "a = 1", "new_string": "a = 2"}}],
         "observation": {"results": [{"source_call_id": "c2", "content": "{\"status\": \"ok\"}"}]}},
        {"step_id": 4, "source": "agent", "message": "Done."},
    ],
}


def rec(tid: str, idx: int, resolved: bool, patch: str) -> dict:
    return {"task_id": tid, "repo": "r", "issue": "i", "sample_idx": idx, "resolved": resolved,
            "patch": patch, "signature": ct.patch_signature(patch) if patch else "",
            "diff_size": ct.diff_size(patch), "messages": [{"role": "user", "content": "x"}],
            "duration_s": 120}


class PureTests(unittest.TestCase):
    def test_signature_ignores_whitespace_and_hunk_headers(self) -> None:
        self.assertEqual(ct.patch_signature(P_A), ct.patch_signature(P_A2))
        self.assertNotEqual(ct.patch_signature(P_A), ct.patch_signature(P_B))

    def test_diff_size(self) -> None:
        self.assertEqual(ct.diff_size(P_A), 2)
        self.assertEqual(ct.diff_size(P_B), 3)

    def test_dev_split_deterministic_and_balanced(self) -> None:
        ids = [f"repo__{i}" for i in range(400)]
        first = [ct.dev_split(i) for i in ids]
        self.assertEqual(first, [ct.dev_split(i) for i in ids])
        frac = first.count("holdout") / len(ids)
        self.assertTrue(0.15 < frac < 0.35, frac)
        self.assertEqual(ct.dev_split("x", holdout_mod=1), "train")

    def test_classify(self) -> None:
        records = [
            rec("gold", 0, True, P_A), rec("gold", 1, True, P_A2), rec("gold", 2, True, P_B),
            rec("gold", 3, False, P_B),
            rec("silver", 0, True, P_A), rec("silver", 1, False, P_B), rec("silver", 2, False, ""),
            rec("divergent", 0, True, P_A), rec("divergent", 1, True, P_B),
            rec("fail", 0, False, P_A), rec("empty", 0, True, ""),
        ]
        gold, silver, extra = ct.classify(records)
        self.assertEqual({(r["task_id"], r["sample_idx"]) for r in gold}, {("gold", 0), ("gold", 1)})
        self.assertEqual([r["task_id"] for r in silver], ["silver"])
        self.assertEqual(extra["divergent_passing_tasks"], 1)
        row = gold[0]
        for key in ("task_id", "repo", "issue", "messages", "patch", "tests_passed"):
            self.assertIn(key, row)
        self.assertEqual((row["n_samples"], row["n_passed"]), (4, 3))

    def test_summarize(self) -> None:
        records = [rec("a", 0, True, P_A), rec("a", 1, True, P_A), rec("b", 0, False, P_B)]
        gold, silver, extra = ct.classify(records)
        s = ct.summarize(records, gold, silver, extra)
        self.assertEqual(s["tasks_covered"], 2)
        self.assertAlmostEqual(s["sample_pass_rate"], 0.6667)
        self.assertEqual(s["task_pass_at_n"], 0.5)
        self.assertEqual((s["gold_tasks"], s["gold_rows"], s["silver_tasks"]), (1, 2, 0))
        self.assertEqual(s["avg_diff_size_kept"], 2.0)

    def test_trace_to_messages(self) -> None:
        msgs = ct.trace_to_messages(ATIF)
        self.assertEqual([m["role"] for m in msgs],
                         ["user", "assistant", "tool", "assistant", "tool", "assistant"])
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(msgs[1]["tool_calls"][0]["function"]["arguments"]), {"filepath": "x.py"})
        self.assertEqual(msgs[1]["reasoning"], "look first")
        self.assertEqual(msgs[2], {"role": "tool", "tool_call_id": "c1", "content": "a = 1"})
        self.assertEqual([m.get("patch_producing") for m in msgs if m["role"] == "assistant"],
                         [False, True, False])

    def test_trace_unknown_shape(self) -> None:
        self.assertEqual(ct.trace_to_messages({"foo": 1}), [])
        self.assertEqual(ct.trace_to_messages(None), [])

    def test_sample_order_seeded_permutation(self) -> None:
        tasks = [SimpleNamespace(instance_id=f"t{i}") for i in range(20)]
        a = ct.sample_order(tasks, 1000)
        self.assertEqual([t.instance_id for t in a], [t.instance_id for t in ct.sample_order(tasks, 1000)])
        self.assertEqual(sorted(t.instance_id for t in a), sorted(t.instance_id for t in tasks))
        self.assertNotEqual([t.instance_id for t in a], [t.instance_id for t in ct.sample_order(tasks, 1001)])
        self.assertEqual([t.instance_id for t in tasks], [f"t{i}" for i in range(20)])  # input untouched

    def test_load_jsonl_tolerates_torn_line(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.jsonl"
            p.write_text(json.dumps({"a": 1}) + "\n{\"a\": ")
            self.assertEqual(ct.load_jsonl(p), [{"a": 1}])


class FakeEvaluator:
    def __init__(self, results_dir: Path, outcomes: dict) -> None:
        self.results_dir = results_dir
        self.outcomes = outcomes
        self.seen: list[str] = []

    async def evaluate_task(self, task, task_index, total_tasks):
        self.seen.append(task.instance_id)
        resolved, patch = self.outcomes[task.instance_id]
        traces = self.results_dir / "traces"
        traces.mkdir(parents=True, exist_ok=True)
        (traces / f"trace_{task.instance_id}.json").write_text(json.dumps(ATIF))
        await asyncio.sleep(0)
        return SimpleNamespace(agent_patch=patch, resolved=resolved, test_exit_code=0 if resolved else 1,
                               tool_calls=7, duration_seconds=90.0)


class RunSamplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def args(self, **kw) -> argparse.Namespace:
        base = dict(agent_dir=str(ROOT / "agent"), work_dir=str(self.tmp / "work"), data_dir=str(self.tmp),
                    model_root=str(self.tmp), model_version="2", n=2, temperature=0.7, seed=1000,
                    concurrency=2, max_time_minutes=0, stop_after_hours=1.0, sandbox="subprocess")
        base.update(kw)
        return argparse.Namespace(**base)

    def run_once(self, tasks, outcomes, out, calls) -> None:
        def fake_eval(sample_dir, results_dir, *a, **k):
            ev = FakeEvaluator(results_dir, outcomes)
            calls.append(ev)
            return ev

        with mock.patch.object(ct, "start_server", return_value=(None, None, None)), \
             mock.patch.object(ct, "make_evaluator", side_effect=fake_eval):
            asyncio.run(ct.run_samples(self.args(), tasks, out))

    def test_end_to_end_with_fake_harness_and_resume(self) -> None:
        import yaml

        tasks = [SimpleNamespace(instance_id=t, repo="r", problem_statement=f"issue {t}") for t in ("t1", "t2")]
        outcomes = {"t1": (True, P_A), "t2": (False, P_B)}
        out = self.tmp / "out"
        out.mkdir()
        calls: list = []
        self.run_once(tasks, outcomes, out, calls)
        raw = ct.load_jsonl(out / ct.RAW_FILE)
        self.assertEqual(len(raw), 4)
        self.assertTrue(all(r["messages"] for r in raw))
        cfg = yaml.safe_load((self.tmp / "work" / "agent_s1" / "configs" / "sampling.yaml").read_text())
        self.assertEqual((cfg["temperature"], cfg["seed"]), (0.7, 1001))
        self.assertEqual(cfg["max_output_tokens"], 16384)

        self.run_once(tasks, outcomes, out, calls)  # resume: nothing new
        self.assertEqual(len(ct.load_jsonl(out / ct.RAW_FILE)), 4)

        summary = ct.finalize(out)
        self.assertEqual((summary["gold_tasks"], summary["silver_tasks"]), (1, 0))
        gold = ct.load_jsonl(out / "sft_gold.jsonl")
        self.assertEqual({r["task_id"] for r in gold}, {"t1"})
        self.assertEqual(gold[0]["issue"], "issue t1")


    def test_shuffled_order_and_partial_resume(self) -> None:
        ids = [f"t{i}" for i in range(8)]
        tasks = [SimpleNamespace(instance_id=t, repo="r", problem_statement=t) for t in ids]
        outcomes = {t: (False, "") for t in ids}
        out = self.tmp / "out"
        out.mkdir()
        ct.append_jsonl(out / ct.RAW_FILE, rec("t3", 0, False, ""))
        calls: list = []
        self.run_once(tasks, outcomes, out, calls)
        s0, s1 = calls
        self.assertNotEqual(s0.seen, s1.seen)  # samples walk different orders
        self.assertEqual(s0.seen, [t.instance_id for t in ct.sample_order(tasks, 1000) if t.instance_id != "t3"])
        self.assertEqual(sorted(s1.seen), ids)
        raw = ct.load_jsonl(out / ct.RAW_FILE)
        self.assertEqual(len(raw), 16)
        self.assertEqual(ct.done_keys(raw), {(t, i) for t in ids for i in (0, 1)})


if __name__ == "__main__":
    unittest.main()
