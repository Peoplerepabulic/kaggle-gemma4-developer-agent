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


class VllmKnobTests(unittest.TestCase):
    STARTER = {  # starter Cell 4 VllmConfig kwargs for 12B, no adapters, 4 GPUs
        "port": 8000, "host": "127.0.0.1", "tool_call_parser": "gemma4", "reasoning_parser": "gemma4",
        "max_model_len": 32768, "dtype": "bfloat16", "gpu_memory_utilization": 0.90,
        "enable_auto_tool_choice": True, "enable_lora": False, "max_loras": 1, "max_lora_rank": 128,
        "scheduler_reserve_full_isl": False, "tensor_parallel_size": 4, "startup_timeout": 1200,
        "extra_args": ("--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'),
    }

    def run_start(self, cfg_cls, **kw) -> dict:
        import dataclasses

        captured = {}

        def record(**kwargs):
            captured.update(kwargs)
            return cfg_cls(**{k: v for k, v in kwargs.items() if k in {f.name for f in dataclasses.fields(cfg_cls)}})

        server = mock.MagicMock()
        fake = {
            "litellm": SimpleNamespace(drop_params=False),
            "torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4, is_available=lambda: True,
                                                          is_bf16_supported=lambda: True)),
            "adk_submission": SimpleNamespace(VllmConfig=mock.MagicMock(side_effect=record),
                                              VllmServer=mock.MagicMock(return_value=server),
                                              discover_adapters=lambda *a, **k: SimpleNamespace(adapters={})),
            "adk_submission.discovery": SimpleNamespace(
                discover_declared_models=lambda d: {"gemma-4-12b-it-qat-w4a16-ct"}),
            "swegemma": SimpleNamespace(), "swegemma.config": SimpleNamespace(ALLOWED_ADAPTER_EXTENSIONS=(".safetensors",)),
        }
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, fake), \
                mock.patch.object(ct, "_config_fields", side_effect=lambda c: {f.name for f in dataclasses.fields(cfg_cls)}):
            (Path(d) / "gemma-4-12b-it-qat-w4a16-ct" / "2").mkdir(parents=True)
            ct.start_server(Path(d), Path(d), "2", **kw)
        captured.pop("model")
        return captured

    def cfg(self, with_eager: bool):
        import dataclasses

        fields = [(n, object, None) for n in self.STARTER] + [("model", object, None)]
        if with_eager:
            fields.append(("enforce_eager", bool, False))
        return dataclasses.make_dataclass("VllmConfig", fields)

    def test_defaults_identical_to_starter(self) -> None:
        for with_eager in (True, False):
            self.assertEqual(self.run_start(self.cfg(with_eager)), self.STARTER)

    def test_enforce_eager_first_class_field(self) -> None:
        got = self.run_start(self.cfg(True), enforce_eager=True, gpu_memory_utilization=0.8)
        self.assertEqual(got, {**self.STARTER, "enforce_eager": True, "gpu_memory_utilization": 0.8})

    def test_enforce_eager_via_extra_args(self) -> None:
        got = self.run_start(self.cfg(False), enforce_eager=True)
        self.assertEqual(got["extra_args"], self.STARTER["extra_args"] + ("--enforce-eager",))
        self.assertNotIn("enforce_eager", got)

    def test_config_fields_detection(self) -> None:
        import dataclasses

        self.assertIn("enforce_eager", ct._config_fields(self.cfg(True)))
        self.assertNotIn("enforce_eager", ct._config_fields(self.cfg(False)))

        class Plain:
            def __init__(self, model, enforce_eager=False):
                pass

        self.assertIn("enforce_eager", ct._config_fields(Plain))
        self.assertTrue(dataclasses.is_dataclass(self.cfg(True)))
