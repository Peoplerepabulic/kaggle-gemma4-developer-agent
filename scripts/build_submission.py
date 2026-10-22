#!/usr/bin/env python3
"""Validate agent/ and package it as submission.zip.

Checks mirror the starter notebook's validation:
- agent.yaml at the zip root
- no symlinks, no '../' traversal
- only allowed file extensions
- total archive size <= 3 GiB
Run: python scripts/build_submission.py [--out submission.zip]
"""
import argparse, os, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = ROOT / "agent"
MAX_BYTES = 3 * 1024**3
ALLOWED_EXTS = {".yaml", ".yml", ".md", ".json", ".safetensors", ".txt", ".py"}

def validate(agent_dir: Path) -> list[str]:
    errors = []
    if not (agent_dir / "agent.yaml").is_file():
        errors.append("agent/agent.yaml missing (must sit at zip root)")
    for p in agent_dir.rglob("*"):
        if p.is_symlink():
            errors.append(f"symlink not allowed: {p.relative_to(agent_dir)}")
        elif p.is_file() and p.suffix.lower() not in ALLOWED_EXTS:
            errors.append(f"disallowed extension: {p.relative_to(agent_dir)}")
    return errors

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "submission.zip"))
    args = ap.parse_args()
    errors = validate(AGENT_DIR)
    if errors:
        print("VALIDATION FAILED:"); [print(" -", e) for e in errors]; return 1
    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(AGENT_DIR.rglob("*")):
            if p.is_file() and not p.is_symlink():
                zf.write(p, p.relative_to(AGENT_DIR)); total += p.stat().st_size
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}  uncompressed={total/1e6:.1f} MB  zip={size/1e6:.1f} MB")
    if size > MAX_BYTES:
        print("ERROR: zip exceeds 3 GiB"); return 1
    print("OK: agent.yaml at root, no symlinks, extensions allowed, size within limit")
    return 0

if __name__ == "__main__":
    sys.exit(main())
