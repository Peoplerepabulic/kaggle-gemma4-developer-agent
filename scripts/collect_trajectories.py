#!/usr/bin/env python3
"""Collect patch trajectories on the public dev tasks and build SFT data.

Runs the official harness exactly like starter Cells 4-5 (vLLM server +
swegemma Evaluator: Phase 1 agent run, Phase 2 hidden-test verification), n
independent samples per task (each sample gets its own copy of agent/ with a
distinct sampling seed and temperature), then classifies (ThinkPRM
multipath-consistent filter):
- gold:   >=2 passing samples converge to an equivalent patch (patch_signature)
- silver: exactly 1 passing sample
- drop:   everything else (incl. >=2 passing but divergent; counted in summary)

A deterministic held-out split (hash of instance_id) is excluded by default so
the P2 A/B can be measured on tasks the LoRA never saw.

Every finished sample is appended to <out>/raw_samples.jsonl at once, so a run
can be stopped and resumed (finished (task, sample) pairs are skipped).

Kaggle usage (L4x4, internet off): run starter Cell 1 (wheelhouse) only, then
  !python scripts/collect_trajectories.py --limit 30
Do not start the starter's vLLM server in the same session (GPU memory).
Smoke test first: --limit 1 --n 1, and check that messages are non-empty.
Rebuild gold/silver from an existing raw log without GPU: --classify-only.

Output rows (sft_gold.jsonl / sft_silver.jsonl):
{"task_id", "repo", "issue", "messages", "patch", "tests_passed",
 "sample_idx", "signature", "diff_size", "n_samples", "n_passed"}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_WORKING = Path("/kaggle/working")
DEFAULT_DATA_DIR = Path("/kaggle/input/competitions/gemma-4-developer-agent")
DEFAULT_MODEL_ROOT = Path("/kaggle/input/models/google/gemma-4/other")
RAW_FILE = "raw_samples.jsonl"
PATCH_TOOLS = {"edit_file", "write_file", "submit_patch"}


# ---------------------------------------------------------------- pure helpers

def patch_signature(patch: str) -> str:
    """Crude equivalence: collapse whitespace, drop hunk/blob headers, hash the sorted body.

    `index <blob>..<blob>` lines are dropped because blob hashes differ whenever
    whitespace does; blank +/- lines are ignored.
    """
    lines = []
    for l in patch.splitlines():
        if l.startswith(("+++", "---", "@@", "index ")):
            continue
        norm = " ".join(l.split())
        if norm not in ("", "+", "-"):
            lines.append(norm)
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()[:16]


def diff_size(patch: str) -> int:
    return sum(1 for l in patch.splitlines()
               if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))


def dev_split(instance_id: str, holdout_mod: int = 4) -> str:
    if holdout_mod <= 1:
        return "train"
    h = int(hashlib.sha256(instance_id.encode()).hexdigest(), 16)
    return "holdout" if h % holdout_mod == 0 else "train"


def sample_order(tasks: list[Any], seed: int) -> list[Any]:
    """Per-sample task order: concurrent samples start on different tasks (spreads sandbox load)."""
    import random

    order = list(tasks)
    random.Random(seed).shuffle(order)
    return order


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn last line from an interrupted run
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_keys(records: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(r["task_id"], int(r["sample_idx"])) for r in records}


def _text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_text(i) for i in x)
    if isinstance(x, dict):
        return _text(x.get("text", x.get("content", json.dumps(x, ensure_ascii=False))))
    return str(x)


def trace_to_messages(trace: Any) -> list[dict[str, Any]]:
    """Best-effort ATIF SessionTrace -> chat messages (system/user/assistant/tool).

    Assistant messages carry `patch_producing` (an edit_file / write_file /
    submit_patch call) for P2 label masking. Returns [] if the shape is unknown;
    the raw trace path is kept in the record either way.
    """
    if isinstance(trace, dict):
        steps = trace.get("steps") or trace.get("trajectory") or trace.get("events")
    elif isinstance(trace, list):
        steps = trace
    else:
        return []
    if not isinstance(steps, list):
        return []
    msgs: list[dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        src = str(st.get("source") or st.get("role") or "").lower()
        content = _text(st.get("message", st.get("content")))
        if src in {"system", "user"}:
            msgs.append({"role": src, "content": content})
            continue
        if src not in {"agent", "assistant", "model"}:
            continue
        calls = []
        for i, tc in enumerate(st.get("tool_calls") or []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = tc.get("function_name") or tc.get("name") or fn.get("name") or ""
            args = tc.get("arguments", fn.get("arguments", {}))
            calls.append({
                "id": str(tc.get("tool_call_id") or tc.get("id") or f"call_{len(msgs)}_{i}"),
                "type": "function",
                "function": {"name": name,
                             "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)},
            })
        m: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning = st.get("reasoning_content") or st.get("reasoning") or st.get("thought")
        if reasoning:
            m["reasoning"] = _text(reasoning)
        if calls:
            m["tool_calls"] = calls
        m["patch_producing"] = any(c["function"]["name"] in PATCH_TOOLS for c in calls)
        msgs.append(m)
        obs = st.get("observation") or {}
        results = obs.get("results", []) if isinstance(obs, dict) else obs if isinstance(obs, list) else []
        for j, res in enumerate(results):
            if isinstance(res, dict):
                cid = res.get("source_call_id") or res.get("tool_call_id")
                if cid is None and j < len(calls):
                    cid = calls[j]["id"]
                msgs.append({"role": "tool", "tool_call_id": str(cid), "content": _text(res.get("content"))})
    return msgs if any(m["role"] == "assistant" for m in msgs) else []


def classify(records: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict[str, Any]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
    gold: list[dict] = []
    silver: list[dict] = []
    divergent = 0
    for tid in sorted(by_task):
        runs = sorted(by_task[tid], key=lambda r: int(r["sample_idx"]))
        passing = [r for r in runs if r.get("resolved") and (r.get("patch") or "").strip()]
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in passing:
            groups[r["signature"]].append(r)
        best = min(groups.values(), key=lambda g: (-len(g), min(x["diff_size"] for x in g)), default=[])
        meta = {"n_samples": len(runs), "n_passed": len(passing)}
        if len(best) >= 2:
            gold.extend(_sft_row(r, meta) for r in best)
        elif len(passing) == 1:
            silver.append(_sft_row(passing[0], meta))
        elif len(passing) >= 2:
            divergent += 1
    return gold, silver, {"divergent_passing_tasks": divergent}


def _sft_row(r: dict[str, Any], meta: dict[str, int]) -> dict[str, Any]:
    return {"task_id": r["task_id"], "repo": r.get("repo"), "issue": r.get("issue"),
            "messages": r.get("messages") or [], "patch": r["patch"], "tests_passed": True,
            "sample_idx": r["sample_idx"], "signature": r["signature"],
            "diff_size": r["diff_size"], **meta}


def summarize(records: list[dict], gold: list[dict], silver: list[dict], extra: dict) -> dict[str, Any]:
    tasks = {r["task_id"] for r in records}
    passed = [r for r in records if r.get("resolved")]
    solved = {r["task_id"] for r in passed}
    kept = gold + silver
    durs = [r["duration_s"] for r in records if isinstance(r.get("duration_s"), (int, float))]
    return {
        "tasks_covered": len(tasks),
        "samples": len(records),
        "sample_pass_rate": round(len(passed) / len(records), 4) if records else 0.0,
        "task_pass_at_n": round(len(solved) / len(tasks), 4) if tasks else 0.0,
        "gold_tasks": len({r["task_id"] for r in gold}),
        "gold_rows": len(gold),
        "silver_tasks": len(silver),
        "dropped_divergent_tasks": extra.get("divergent_passing_tasks", 0),
        "avg_diff_size_kept": round(sum(r["diff_size"] for r in kept) / len(kept), 2) if kept else 0.0,
        "avg_minutes_per_sample": round(sum(durs) / len(durs) / 60, 2) if durs else 0.0,
        "kept_rows_missing_messages": sum(1 for r in kept if not r["messages"]),
    }


def finalize(out_dir: Path) -> dict[str, Any]:
    records = load_jsonl(out_dir / RAW_FILE)
    gold, silver, extra = classify(records)
    write_jsonl(out_dir / "sft_gold.jsonl", gold)
    write_jsonl(out_dir / "sft_silver.jsonl", silver)
    summary = summarize(records, gold, silver, extra)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, indent=2))
    if summary["kept_rows_missing_messages"]:
        print("WARN: some kept rows have no messages — check trace capture before P2")
    return summary


# ------------------------------------------------------------ harness wrapper

def make_sample_dir(agent_dir: Path, dest: Path, temperature: float, seed: int) -> Path:
    import yaml

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(agent_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
    cfg = dest / "configs" / "sampling.yaml"
    data = (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}) if cfg.is_file() else {}
    data.update({"temperature": temperature, "seed": seed})
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return dest


def read_eval_budget(agent_dir: Path) -> dict[str, Any]:
    import yaml

    for name in ("eval_config.yaml", "eval_config.yml"):
        p = agent_dir / name
        if p.is_file():
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            return raw.get("evaluation", raw)
    return {}


def _to_jsonable(obj: Any) -> Any:
    for attr in ("model_dump", "to_dict", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return obj


def find_trace(result: Any, results_dir: Path, task_id: str) -> tuple[Any, str | None]:
    for attr in ("trace", "session_trace", "trajectory"):
        val = getattr(result, attr, None)
        if val:
            return _to_jsonable(val), f"result.{attr}"
    for p in sorted(results_dir.rglob(f"trace_{task_id}*.json")):
        try:
            return json.loads(p.read_text(encoding="utf-8")), str(p)
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def start_server(agent_dir: Path, model_root: Path, model_version: str) -> tuple[Any, Any, Any]:
    """Starter Cell 4, unchanged in substance."""
    import re
    import litellm
    import torch
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    from adk_submission.discovery import discover_declared_models
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

    litellm.drop_params = True
    extra = {"gemma-4-12b-it-qat-w4a16-ct": (
        "--hf-overrides", '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}')}
    declared = discover_declared_models(agent_dir)
    assert len(declared) == 1, f"Expected exactly 1 declared model, got {sorted(declared)}"
    model = next(iter(declared))
    model_path = model_root / model / model_version
    assert model_path.exists(), f"Model weights not found at {model_path} — attach the model input"
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    amap = getattr(adapters, "adapters", None) or {}
    for name in amap:
        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name), f"Unsafe adapter name {name!r}"
    ranks = [r for info in amap.values() if (r := getattr(info, "rank", None)) is not None and r > 0]
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    cfg = VllmConfig(
        model=str(model_path), port=8000, host="127.0.0.1",
        tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768,
        dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "auto",
        gpu_memory_utilization=0.90, enable_auto_tool_choice=True,
        enable_lora=bool(amap), max_loras=max(1, len(amap)),
        max_lora_rank=max(ranks) if ranks else 128, scheduler_reserve_full_isl=False,
        tensor_parallel_size=4 if gpus >= 4 else (2 if gpus >= 2 else 1),
        startup_timeout=60 * 20, extra_args=extra.get(model, ()),
    )
    server = VllmServer(cfg, adapter_manifest=adapters)
    server.start()
    print(f"vLLM server started on {server.base_url} (model={model})")
    models = server.create_model_registry(
        aliases=[model], model_prefix="openai/", api_key=os.environ.get("LOCAL_API_KEY", "EMPTY"))
    return server, models, adapters


def make_evaluator(sample_dir: Path, results_dir: Path, data_dir: Path, models: Any,
                   adapters: Any, budget: dict[str, Any], sandbox: str) -> Any:
    """Starter Cell 5 EvalConfig, one Evaluator per sample copy of agent/."""
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from swegemma.config import EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator

    limits, gen_constraints = build_submission_limits()
    turns = budget.get("max_turns", budget.get("max_llm_calls"))
    cfg = EvalConfig(
        tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots",
        results_dir=results_dir, submission_dir=sample_dir, models=models, sandbox=sandbox,
        timeout_seconds=int(budget.get("timeout_seconds", 300)),
        verification_timeout_seconds=int(budget.get("verification_timeout_seconds", 300)),
        max_time_minutes=float(budget.get("max_time_minutes", 60.0)),
        max_tool_calls=int(budget.get("max_tool_calls", 100)),
        max_turns=int(turns) if turns is not None else None,
        limits=limits, generation_constraints=gen_constraints, adapter_manifest=adapters,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(data_dir / "graphs"), embeddings_dir=str(data_dir / "embeddings"),
        wheels_dir=data_dir / "wheels", verbose=False,
    )
    return Evaluator(cfg)


async def call_evaluate(ev: Any, task: Any, k: int, total: int) -> Any:
    import asyncio
    import inspect

    fn = ev.evaluate_task
    kw = {"task": task, "task_index": k, "total_tasks": total}
    res = await fn(**kw) if inspect.iscoroutinefunction(fn) else await asyncio.to_thread(fn, **kw)
    return await res if inspect.isawaitable(res) else res


async def run_samples(args: argparse.Namespace, tasks: list[Any], out_dir: Path) -> None:
    import asyncio

    raw_path = out_dir / RAW_FILE
    done = done_keys(load_jsonl(raw_path))
    agent_dir, work = Path(args.agent_dir), Path(args.work_dir)
    budget = read_eval_budget(agent_dir)
    if args.max_time_minutes:
        budget["max_time_minutes"] = args.max_time_minutes
    _server, models, adapters = start_server(agent_dir, Path(args.model_root), args.model_version)
    deadline = time.monotonic() + args.stop_after_hours * 3600
    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()
    todo = sum(1 for t in tasks for i in range(args.n) if (t.instance_id, i) not in done)
    print(f"{len(tasks)} tasks x {args.n} samples, {todo} to run, budget {budget}")

    async def worker(idx: int) -> None:
        sample_dir = make_sample_dir(agent_dir, work / f"agent_s{idx}", args.temperature, args.seed + idx)
        results_dir = work / "results" / f"s{idx}"
        ev = make_evaluator(sample_dir, results_dir, Path(args.data_dir), models, adapters,
                            budget, args.sandbox)
        for k, task in enumerate(sample_order(tasks, args.seed + idx), start=1):
            if (task.instance_id, idx) in done:
                continue
            if time.monotonic() > deadline:
                print(f"[s{idx}] stop-after-hours reached, stopping")
                return
            async with sem:
                t0 = time.monotonic()
                try:
                    res = await call_evaluate(ev, task, k, len(tasks))
                except Exception as e:  # one bad task must not kill the run
                    print(f"[s{idx}] {task.instance_id} ERROR {type(e).__name__}: {e}")
                    continue
            patch = (getattr(res, "agent_patch", "") or "").strip()
            trace, trace_src = find_trace(res, results_dir, task.instance_id)
            rec = {
                "task_id": task.instance_id, "repo": task.repo,
                "issue": getattr(task, "problem_statement", ""), "sample_idx": idx,
                "resolved": bool(getattr(res, "resolved", False)),
                "test_exit_code": getattr(res, "test_exit_code", None),
                "patch": patch, "signature": patch_signature(patch) if patch else "",
                "diff_size": diff_size(patch), "tool_calls": getattr(res, "tool_calls", None),
                "duration_s": getattr(res, "duration_seconds", time.monotonic() - t0),
                "messages": trace_to_messages(trace), "trace_source": trace_src,
                "temperature": args.temperature, "seed": args.seed + idx,
            }
            async with write_lock:
                append_jsonl(raw_path, rec)
            print(f"[s{idx}] {task.instance_id} resolved={rec['resolved']} diff={rec['diff_size']} "
                  f"msgs={len(rec['messages'])} {rec['duration_s']:.0f}s")

    await asyncio.gather(*(worker(i) for i in range(args.n)))


def select_tasks(tasks: list[Any], args: argparse.Namespace) -> list[Any]:
    if args.task_ids:
        wanted = set(args.task_ids)
        return [t for t in tasks if t.instance_id in wanted]
    sel = [t for t in tasks if args.split == "all" or dev_split(t.instance_id, args.holdout_mod) == args.split]
    sel = [t for i, t in enumerate(sel) if i % args.num_shards == args.shard_index]
    return sel[: args.limit] if args.limit else sel


def main() -> None:
    out_default = (KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else ROOT) / "data"
    work_default = (KAGGLE_WORKING if KAGGLE_WORKING.is_dir() else ROOT / "data") / "collect"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="competition dataset root")
    ap.add_argument("--agent-dir", default=str(ROOT / "agent"))
    ap.add_argument("--out", default=str(out_default))
    ap.add_argument("--work-dir", default=str(work_default), help="per-sample agent copies + harness results")
    ap.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    ap.add_argument("--model-version", default="2")
    ap.add_argument("--n", type=int, default=4, help="samples per task")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1000, help="sample i uses seed+i")
    ap.add_argument("--concurrency", type=int, default=4, help="parallel agent sessions (vLLM batches them)")
    ap.add_argument("--split", choices=["train", "holdout", "all"], default="train")
    ap.add_argument("--holdout-mod", type=int, default=4, help="holdout = sha256(id) %% mod == 0")
    ap.add_argument("--limit", type=int, default=0, help="first N selected tasks (0 = all)")
    ap.add_argument("--task-ids", nargs="*")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-time-minutes", type=float, default=0, help="override eval_config per-task cap")
    ap.add_argument("--stop-after-hours", type=float, default=10.5, help="no new samples after this")
    ap.add_argument("--sandbox", default="subprocess")
    ap.add_argument("--classify-only", action="store_true", help="rebuild gold/silver from raw log")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.classify_only:
        import asyncio

        for k, v in {"TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
                     "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                     "LITELLM_TELEMETRY": "False", "OTEL_SDK_DISABLED": "true",
                     "LOCAL_API_KEY": "EMPTY", "OPENAI_API_KEY": "EMPTY"}.items():
            os.environ.setdefault(k, v)
        from swegemma.models import load_tasks

        all_tasks = load_tasks(Path(args.data_dir) / "tasks.jsonl")
        tasks = select_tasks(all_tasks, args)
        holdout = [t.instance_id for t in all_tasks if dev_split(t.instance_id, args.holdout_mod) == "holdout"]
        (out_dir / "holdout_ids.txt").write_text("\n".join(sorted(holdout)) + "\n")
        asyncio.run(run_samples(args, tasks, out_dir))
    finalize(out_dir)


if __name__ == "__main__":
    main()
