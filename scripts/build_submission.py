#!/usr/bin/env python3
"""Validate agent/ and package it as submission.zip.

Mirrors the starter notebook (Cells 4 + 6) and HARNESS_README §2-3, §7:
- exactly one root config (agent.yaml); no symlinks; no path escaping the root
- allowed extensions only: .yaml .yml .md .txt .py .json .safetensors
- total UNPACKED size <= 3 GiB (adapters included)
- YAML parses; !include targets exist inside agent/ (.md/.txt as text, .yaml recursive)
- agent configs (agent.yaml + every config_path + sub_agents/*.yaml):
  known keys only, one allowed base model, harness tool names only,
  config_path / adapter / skills references resolve,
  generate_content_config fields + token ceilings, instruction placeholders
- eval_config.yaml: `evaluation:` mapping with known numeric keys
- skills: SKILL.md frontmatter with `name`; adapters: complete + model size matches
Run: python scripts/build_submission.py [--agent-dir agent] [--out submission.zip]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "agent"
MAX_BYTES = 3 * 1024**3
ALLOWED_EXTS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
MODEL_PREFIXES = ("openai/", "google/", "hosted_vllm/", "custom/")
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
HARNESS_TOOLS = {
    "run_command", "submit_patch", "get_status", "read_file", "edit_file", "write_file",
    "get_code_neighbors", "search_similar_code", "get_code_subgraph",
    "load_skill", "load_skill_resource", "run_skill_script",
}
AGENT_CLASSES = {"LlmAgent", "SequentialAgent", "ParallelAgent", "LoopAgent"}
AGENT_KEYS = {
    "agent_class", "name", "model", "adapter", "description", "instruction",
    "global_instruction", "tools", "skills", "sub_agents", "output_key", "include_contents",
    "disallow_transfer_to_parent", "disallow_transfer_to_peers", "generate_content_config",
    "max_iterations",
    "before_agent_callbacks", "after_agent_callbacks", "before_model_callbacks",
    "after_model_callbacks", "before_tool_callbacks", "after_tool_callbacks",
}
GEN_KEYS = {
    "temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty",
    "frequency_penalty", "stop_sequences", "response_mime_type", "seed", "thinking_config",
}
THINKING_KEYS = {"thinking_budget", "thinking_level", "include_thoughts"}
MAX_TOKENS = 32768
EVAL_KEYS = {"timeout_seconds", "verification_timeout_seconds", "max_tool_calls",
             "max_time_minutes", "max_turns"}
STATE_VARS = {"problem_description"}  # `hints` is only set when non-empty: use {hints?}
ADAPTER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")
SIZE_RE = re.compile(r"gemma-?4-(31b|12b|e4b|e2b)", re.IGNORECASE)
SKIP_NAMES = {"__pycache__", ".DS_Store"}
MAX_INCLUDE_DEPTH = 10
HIDDEN_TASKS = 120
TIME_BUDGET_MIN = 12 * 60


class Include:
    def __init__(self, target: str) -> None:
        self.target = target


def _yaml() -> Any:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda ld, node: Include(ld.construct_scalar(node)))
    return yaml, Loader


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _iter_files(agent_dir: Path) -> list[Path]:
    out = []
    for p in sorted(agent_dir.rglob("*")):
        if any(part in SKIP_NAMES for part in p.relative_to(agent_dir).parts):
            continue
        if p.is_symlink() or p.is_file():
            out.append(p)
    return out


def _model_size(name: str) -> str | None:
    m = SIZE_RE.search(name)
    return m.group(1).lower() if m else None


def normalize_model(name: str) -> str:
    for pre in MODEL_PREFIXES:
        if name.startswith(pre):
            return name[len(pre):]
    return name


class _Ctx:
    def __init__(self, agent_dir: Path) -> None:
        self.root = agent_dir
        self.yaml, self.loader = _yaml()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.models: dict[str, str] = {}
        self.adapters_used: dict[str, str] = {}
        self.visited: set[Path] = set()
        self.skills_used: set[Path] = set()

    def rel(self, p: Path) -> str:
        return p.resolve().relative_to(self.root.resolve()).as_posix()

    def find(self, target: str, base: Path, where: str, what: str) -> Path | None:
        t = Path(target)
        if t.is_absolute() or ".." in t.parts or "\x00" in target:
            self.errors.append(f"{where}: {what} escapes submission root: {target}")
            return None
        hit = next((c for c in (base / t, self.root / t) if c.exists()), None)
        if hit is None:
            self.errors.append(f"{where}: {what} not found: {target}")
        elif not _within(hit, self.root):
            self.errors.append(f"{where}: {what} escapes submission root: {target}")
            return None
        return hit

    def load(self, path: Path, depth: int = 0, chain: tuple[Path, ...] = ()) -> Any:
        where = self.rel(path)
        try:
            doc = self.yaml.load(path.read_text(encoding="utf-8"), Loader=self.loader)
        except self.yaml.YAMLError as e:
            self.errors.append(f"YAML parse error in {where}: {e}")
            return None
        return self._resolve(doc, path, depth, chain + (path.resolve(),))

    def _resolve(self, obj: Any, path: Path, depth: int, chain: tuple[Path, ...]) -> Any:
        if isinstance(obj, Include):
            where = self.rel(path)
            hit = self.find(obj.target, path.parent, where, "!include target")
            if hit is None:
                return None
            if hit.resolve() in chain or depth >= MAX_INCLUDE_DEPTH:
                self.errors.append(f"{where}: !include cycle or depth > {MAX_INCLUDE_DEPTH}: {obj.target}")
                return None
            if hit.suffix.lower() in {".yaml", ".yml"}:
                return self.load(hit, depth + 1, chain)
            if hit.suffix.lower() in {".md", ".txt"}:
                return hit.read_text(encoding="utf-8")
            self.errors.append(f"{where}: !include must target .md/.txt/.yaml/.yml: {obj.target}")
            return None
        if isinstance(obj, dict):
            return {k: self._resolve(v, path, depth, chain) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve(v, path, depth, chain) for v in obj]
        return obj

    def agent_file(self, path: Path) -> None:
        if path.resolve() in self.visited:
            return
        self.visited.add(path.resolve())
        doc = self.load(path)
        if doc is not None:
            self.agent(doc, path, self.rel(path))

    def agent(self, doc: Any, path: Path, where: str) -> None:
        if not isinstance(doc, dict):
            self.errors.append(f"{where}: agent config must be a mapping")
            return
        for k in sorted(set(doc) - AGENT_KEYS):
            self.errors.append(f"{where}: unknown agent key `{k}`")
        if not doc.get("name"):
            self.errors.append(f"{where}: `name` required")
        cls = doc.get("agent_class", "LlmAgent")
        if cls not in AGENT_CLASSES:
            self.errors.append(f"{where}: unknown agent_class {cls!r}")
        if isinstance(doc.get("model"), str):
            self.models[where] = normalize_model(doc["model"])
        elif cls == "LlmAgent":
            self.errors.append(f"{where}: LlmAgent needs `model`")
        if doc.get("adapter") is not None:
            self.adapters_used[where] = str(doc["adapter"])
        for key in ("instruction", "global_instruction"):
            if isinstance(doc.get(key), str):
                for var in PLACEHOLDER_RE.findall(doc[key]):
                    if var not in STATE_VARS:
                        self.errors.append(
                            f"{where}: {key} placeholder {{{var}}} is not a harness state variable "
                            f"(ADK would raise on it; use {{{var}?}} if optional, or remove braces)")
        for i, tool in enumerate(doc.get("tools") or []):
            if isinstance(tool, str):
                if tool not in HARNESS_TOOLS:
                    self.errors.append(f"{where}: tool not provided by harness: {tool}")
            elif isinstance(tool, dict) and set(tool) == {"agent_tool"}:
                spec = tool["agent_tool"] or {}
                cp = spec.get("config_path") if isinstance(spec, dict) else None
                if not cp:
                    self.errors.append(f"{where}: tools[{i}].agent_tool needs config_path")
                elif (hit := self.find(cp, path.parent, where, "agent_tool config_path")):
                    self.agent_file(hit)
            elif isinstance(tool, dict) and "name" in tool:
                self.agent(tool, path, f"{where}:tools[{i}]")
            else:
                self.errors.append(f"{where}: tools[{i}] is neither a tool name, agent_tool nor inline agent")
        for i, sub in enumerate(doc.get("sub_agents") or []):
            cp = sub.get("config_path") if isinstance(sub, dict) else None
            if not cp:
                self.errors.append(f"{where}: sub_agents[{i}] needs config_path")
            elif (hit := self.find(cp, path.parent, where, "sub_agents config_path")):
                self.agent_file(hit)
        for s in doc.get("skills") or []:
            hit = self.find(str(s), path.parent, where, "skill directory")
            if hit is not None:
                if not (hit / "SKILL.md").is_file():
                    self.errors.append(f"{where}: skill has no SKILL.md: {s}")
                self.skills_used.add(hit.resolve())
        gcc = doc.get("generate_content_config")
        if gcc is not None:
            self.gen_config(gcc, where)

    def gen_config(self, gcc: Any, where: str) -> None:
        if not isinstance(gcc, dict):
            self.errors.append(f"{where}: generate_content_config must be a mapping")
            return
        for k in sorted(set(gcc) - GEN_KEYS):
            self.errors.append(f"{where}: generate_content_config field not allowed: `{k}`")
        mot = gcc.get("max_output_tokens")
        if mot is not None and not (isinstance(mot, int) and 1 <= mot <= MAX_TOKENS):
            self.errors.append(f"{where}: max_output_tokens must be an int in 1..{MAX_TOKENS}")
        tc = gcc.get("thinking_config")
        if tc is not None:
            if not isinstance(tc, dict):
                self.errors.append(f"{where}: thinking_config must be a mapping")
                return
            for k in sorted(set(tc) - THINKING_KEYS):
                self.errors.append(f"{where}: thinking_config field not allowed: `{k}`")
            tb = tc.get("thinking_budget")
            if tb is not None and not (isinstance(tb, int) and 0 <= tb <= MAX_TOKENS):
                self.errors.append(f"{where}: thinking_budget must be an int in 0..{MAX_TOKENS}")
            if "thinking_level" in tc:
                self.warnings.append(f"{where}: thinking_level is dropped under vLLM; use thinking_budget")


def _check_eval_config(ctx: _Ctx) -> None:
    files = [ctx.root / n for n in ("eval_config.yaml", "eval_config.yml") if (ctx.root / n).is_file()]
    if len(files) > 1:
        ctx.errors.append("both eval_config.yaml and eval_config.yml present")
    if not files:
        ctx.warnings.append("no eval_config.yaml: harness defaults apply (60 min, 100 calls per task)")
        return
    doc = ctx.load(files[0])
    where = files[0].name
    if not isinstance(doc, dict) or not isinstance(doc.get("evaluation"), dict):
        ctx.errors.append(f"{where}: budgets must sit under an `evaluation:` mapping")
        return
    for k in sorted(set(doc) - {"evaluation"}):
        ctx.errors.append(f"{where}: top-level key `{k}` is ignored by the harness")
    ev = doc["evaluation"]
    for k in sorted(set(ev) - EVAL_KEYS):
        ctx.errors.append(f"{where}: unknown evaluation key `{k}`")
    for k in EVAL_KEYS & set(ev):
        v = ev[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            ctx.errors.append(f"{where}: evaluation.{k} must be a positive number")
    mtm = ev.get("max_time_minutes")
    if isinstance(mtm, (int, float)) and mtm * HIDDEN_TASKS > TIME_BUDGET_MIN * 0.9:
        ctx.warnings.append(
            f"{where}: max_time_minutes={mtm} x {HIDDEN_TASKS} tasks = {mtm * HIDDEN_TASKS / 60:.1f} h "
            f"of agent time alone, before setup (budget {TIME_BUDGET_MIN / 60:.0f} h)")


def _check_skills(ctx: _Ctx) -> None:
    skills_dir = ctx.root / "skills"
    if not skills_dir.is_dir():
        return
    for d in sorted(x for x in skills_dir.iterdir() if x.is_dir() and x.name not in SKIP_NAMES):
        skill_md = d / "SKILL.md"
        if not skill_md.is_file():
            ctx.errors.append(f"skills/{d.name}: SKILL.md missing")
            continue
        parts = skill_md.read_text(encoding="utf-8").split("---", 2)
        meta = None
        if len(parts) == 3 and not parts[0].strip():
            try:
                meta = ctx.yaml.safe_load(parts[1])
            except ctx.yaml.YAMLError:
                meta = None
        if not isinstance(meta, dict) or not meta.get("name"):
            ctx.errors.append(f"skills/{d.name}/SKILL.md: frontmatter with `name` required")
        if d.resolve() not in ctx.skills_used:
            ctx.warnings.append(f"skills/{d.name} is not listed in any agent's `skills:` (shipped but unused)")


def _check_adapters(ctx: _Ctx, root_size: str | None) -> None:
    adapters_dir = ctx.root / "adapters"
    present: set[str] = set()
    if adapters_dir.is_dir():
        for d in sorted(x for x in adapters_dir.iterdir() if x.name not in SKIP_NAMES):
            if not d.is_dir():
                ctx.errors.append(f"adapters/{d.name}: only adapter folders belong in adapters/")
                continue
            present.add(d.name)
            if not ADAPTER_NAME_RE.fullmatch(d.name):
                ctx.errors.append(f"adapters/{d.name}: name must match [A-Za-z0-9][A-Za-z0-9._-]*")
            for req in ("adapter_config.json", "adapter_model.safetensors"):
                if not (d / req).is_file():
                    ctx.errors.append(f"adapters/{d.name}: {req} missing")
            cfg_path = d / "adapter_config.json"
            if cfg_path.is_file():
                try:
                    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e:
                    ctx.errors.append(f"adapters/{d.name}/adapter_config.json: invalid JSON ({e})")
                    continue
                size = _model_size(cfg.get("base_model_name_or_path") or "")
                if root_size and size and size != root_size:
                    ctx.errors.append(
                        f"adapters/{d.name}: trained for {size.upper()} but agent model is {root_size.upper()}")
                if isinstance(cfg.get("r"), int) and cfg["r"] > 128:
                    ctx.errors.append(f"adapters/{d.name}: rank {cfg['r']} > max_lora_rank 128")
    for where, name in ctx.adapters_used.items():
        if name not in present:
            ctx.errors.append(f"{where}: adapter `{name}` has no folder under adapters/")


def validate_all(agent_dir: Path) -> tuple[list[str], list[str]]:
    roots = [n for n in ROOT_CONFIG_NAMES if (agent_dir / n).is_file()]
    if "agent.yaml" not in roots:
        return ["agent.yaml missing (must sit at zip root)"], []
    errors: list[str] = []
    if len(roots) > 1:
        errors.append(f"multiple root configs: {roots}")

    files = _iter_files(agent_dir)
    total = 0
    for p in files:
        rel = p.relative_to(agent_dir)
        if p.is_symlink():
            errors.append(f"symlink not allowed: {rel}")
        elif ".." in rel.parts or not _within(p, agent_dir):
            errors.append(f"path escapes submission root: {rel}")
        elif p.suffix.lower() not in ALLOWED_EXTS:
            errors.append(f"disallowed extension: {rel}")
        else:
            total += p.stat().st_size
    if total > MAX_BYTES:
        errors.append(f"unpacked size {total / 1024**3:.2f} GiB exceeds 3 GiB")

    try:
        ctx = _Ctx(agent_dir)
    except ImportError:
        return errors + ["PyYAML not installed — cannot validate YAML files"], []
    ctx.errors = errors

    for p in files:
        if not p.is_symlink() and p.suffix.lower() in {".yaml", ".yml"}:
            ctx.load(p)  # surfaces parse/include errors in every YAML file
    ctx.errors = list(dict.fromkeys(ctx.errors))
    ctx.agent_file(agent_dir / "agent.yaml")
    sub_dir = agent_dir / "sub_agents"
    if sub_dir.is_dir():
        for p in sorted(sub_dir.glob("*.y*ml")):
            if p.resolve() not in ctx.visited:
                ctx.warnings.append(f"{ctx.rel(p)} is not referenced by any agent (still checked)")
                ctx.agent_file(p)

    root_model = ctx.models.get("agent.yaml")
    for where, name in ctx.models.items():
        if name not in ALLOWED_MODELS:
            ctx.errors.append(f"{where}: model not in allowed list: {name}")
    if len(set(ctx.models.values())) > 1:
        ctx.errors.append(f"more than one base model in submission: {sorted(set(ctx.models.values()))}")

    _check_eval_config(ctx)
    _check_skills(ctx)
    _check_adapters(ctx, _model_size(root_model or ""))
    return list(dict.fromkeys(ctx.errors)), ctx.warnings


def validate(agent_dir: Path) -> list[str]:
    return validate_all(agent_dir)[0]


def build(agent_dir: Path, out: Path) -> int:
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in _iter_files(agent_dir):
            zf.write(p, p.relative_to(agent_dir).as_posix())
            total += p.stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-dir", default=str(AGENT_DIR))
    ap.add_argument("--out", default=str(ROOT / "submission.zip"))
    args = ap.parse_args()
    agent_dir, out = Path(args.agent_dir), Path(args.out)

    errors, warnings = validate_all(agent_dir)
    for w in warnings:
        print("WARN:", w)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    total = build(agent_dir, out)
    size = os.path.getsize(out)
    print(f"wrote {out}  uncompressed={total/1e6:.1f} MB  zip={size/1e6:.1f} MB")
    print("OK: root config, paths, extensions, size, YAML + includes, agent schema keys, tools, "
          "single allowed model, references, generation + eval config, skills, adapters")
    return 0


if __name__ == "__main__":
    sys.exit(main())
