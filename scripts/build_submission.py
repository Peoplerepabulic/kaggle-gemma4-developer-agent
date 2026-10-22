#!/usr/bin/env python3
"""Validate agent/ and package it as submission.zip.

Checks (mirror the starter notebook's validation plus CLAUDE.md hard rules):
- agent.yaml at the zip root
- no symlinks, no path escaping the submission root ('../' or !include targets)
- only allowed file extensions (sync ALLOWED_EXTS with the starter notebook)
- every YAML parses (the harness's !include tag is accepted, target must exist)
- exactly one base model across agent.yaml and sub_agents/, from the allowed list
- each skill's SKILL.md has YAML frontmatter with `name`
- each adapter folder has adapter_config.json + adapter_model.safetensors,
  and its base model size matches the agent's model (catches the starter's
  31B-shaped dummy adapters)
- total archive size <= 3 GiB
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
ALLOWED_EXTS = {".yaml", ".yml", ".md", ".json", ".safetensors", ".txt", ".py"}
ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}
SKIP_NAMES = {"__pycache__", ".DS_Store"}
SIZE_RE = re.compile(r"gemma-?4-(31b|12b|e4b|e2b)", re.IGNORECASE)


class Include:
    def __init__(self, target: str) -> None:
        self.target = target


def _yaml_loader() -> Any:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include", lambda ld, node: Include(ld.construct_scalar(node)))
    return Loader


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _iter_files(agent_dir: Path) -> list[Path]:
    out = []
    for p in sorted(agent_dir.rglob("*")):
        rel = p.relative_to(agent_dir)
        if any(part in SKIP_NAMES for part in rel.parts):
            continue
        if p.is_symlink() or p.is_file():
            out.append(p)
    return out


def _walk_includes(obj: Any) -> list[str]:
    if isinstance(obj, Include):
        return [obj.target]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _walk_includes(v)]
    if isinstance(obj, list):
        return [t for v in obj for t in _walk_includes(v)]
    return []


def _model_size(name: str) -> str | None:
    m = SIZE_RE.search(name)
    return m.group(1).lower() if m else None


def validate(agent_dir: Path) -> list[str]:
    errors: list[str] = []
    if not (agent_dir / "agent.yaml").is_file():
        return ["agent.yaml missing (must sit at zip root)"]

    files = _iter_files(agent_dir)
    for p in files:
        rel = p.relative_to(agent_dir)
        if p.is_symlink():
            errors.append(f"symlink not allowed: {rel}")
        elif ".." in rel.parts or not _within(p, agent_dir):
            errors.append(f"path escapes submission root: {rel}")
        elif p.suffix.lower() not in ALLOWED_EXTS:
            errors.append(f"disallowed extension: {rel}")

    try:
        loader = _yaml_loader()
        import yaml
    except ImportError:
        errors.append("PyYAML not installed — cannot validate YAML files")
        return errors

    models: dict[str, str] = {}
    for p in files:
        if p.is_symlink() or p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        rel = p.relative_to(agent_dir)
        try:
            doc = yaml.load(p.read_text(encoding="utf-8"), Loader=loader)
        except yaml.YAMLError as e:
            errors.append(f"YAML parse error in {rel}: {e}")
            continue
        for target in _walk_includes(doc):
            cands = [p.parent / target, agent_dir / target]
            hit = next((c for c in cands if c.is_file()), None)
            if hit is None:
                errors.append(f"{rel}: !include target not found: {target}")
            elif ".." in Path(target).parts or not _within(hit, agent_dir):
                errors.append(f"{rel}: !include escapes submission root: {target}")
        if isinstance(doc, dict) and isinstance(doc.get("model"), str):
            models[str(rel)] = doc["model"]

    if "agent.yaml" not in models:
        errors.append("agent.yaml has no `model:` key")
    for rel, name in models.items():
        if name.split("/")[-1] not in ALLOWED_MODELS:
            errors.append(f"{rel}: model not in allowed list: {name}")
    distinct = {n.split("/")[-1] for n in models.values()}
    if len(distinct) > 1:
        errors.append(f"more than one base model in submission: {sorted(distinct)}")
    root_size = _model_size(models.get("agent.yaml", ""))

    skills_dir = agent_dir / "skills"
    if skills_dir.is_dir():
        for d in sorted(x for x in skills_dir.iterdir() if x.is_dir() and x.name not in SKIP_NAMES):
            skill_md = d / "SKILL.md"
            if not skill_md.is_file():
                errors.append(f"skills/{d.name}: SKILL.md missing")
                continue
            parts = skill_md.read_text(encoding="utf-8").split("---", 2)
            meta = None
            if len(parts) == 3 and not parts[0].strip():
                try:
                    meta = yaml.safe_load(parts[1])
                except yaml.YAMLError:
                    meta = None
            if not isinstance(meta, dict) or not meta.get("name"):
                errors.append(f"skills/{d.name}/SKILL.md: frontmatter with `name` required")

    adapters_dir = agent_dir / "adapters"
    if adapters_dir.is_dir():
        for d in sorted(x for x in adapters_dir.iterdir() if x.is_dir() and x.name not in SKIP_NAMES):
            for req in ("adapter_config.json", "adapter_model.safetensors"):
                if not (d / req).is_file():
                    errors.append(f"adapters/{d.name}: {req} missing")
            cfg_path = d / "adapter_config.json"
            if cfg_path.is_file():
                try:
                    base = json.loads(cfg_path.read_text(encoding="utf-8")).get("base_model_name_or_path") or ""
                except json.JSONDecodeError as e:
                    errors.append(f"adapters/{d.name}/adapter_config.json: invalid JSON ({e})")
                    continue
                size = _model_size(base)
                if root_size and size and size != root_size:
                    errors.append(f"adapters/{d.name}: trained for {size.upper()} but agent model is {root_size.upper()}")
    return errors


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

    errors = validate(agent_dir)
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    total = build(agent_dir, out)
    size = os.path.getsize(out)
    print(f"wrote {out}  uncompressed={total/1e6:.1f} MB  zip={size/1e6:.1f} MB")
    if size > MAX_BYTES:
        print("ERROR: zip exceeds 3 GiB")
        return 1
    print("OK: agent.yaml at root, no symlinks/escapes, extensions allowed, YAML + includes valid, "
          "single allowed model, skills/adapters well-formed, size within limit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
