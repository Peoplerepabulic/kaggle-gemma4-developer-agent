"""Run: python -m unittest discover -s tests   (or: pytest tests)"""
import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission as bs  # noqa: E402


class ValidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.agent = self.tmp / "agent"
        shutil.copytree(ROOT / "agent", self.agent)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def errs(self) -> list[str]:
        return bs.validate(self.agent)

    def assertErr(self, needle: str) -> None:
        errs = self.errs()
        self.assertTrue(any(needle in e for e in errs), f"{needle!r} not in {errs}")

    def add_adapter(self, base: str, weights: bool = True) -> None:
        d = self.agent / "adapters" / "a1"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
        if weights:
            (d / "adapter_model.safetensors").write_bytes(b"")

    def test_repo_template_is_valid(self) -> None:
        self.assertEqual(self.errs(), [])

    def test_build_puts_agent_yaml_at_root(self) -> None:
        out = self.tmp / "s.zip"
        bs.build(self.agent, out)
        names = zipfile.ZipFile(out).namelist()
        self.assertIn("agent.yaml", names)
        self.assertFalse(any(n.startswith(("/", "..")) or "\\" in n for n in names))

    def test_missing_agent_yaml(self) -> None:
        (self.agent / "agent.yaml").unlink()
        self.assertErr("agent.yaml missing")

    def test_symlink(self) -> None:
        os.symlink(self.agent / "eval_config.yaml", self.agent / "link.yaml")
        self.assertErr("symlink not allowed")

    def test_bad_extension(self) -> None:
        (self.agent / "run.sh").write_text("echo hi")
        self.assertErr("disallowed extension")

    def test_pycache_skipped(self) -> None:
        (self.agent / "skills" / "debug-loop" / "__pycache__").mkdir()
        (self.agent / "skills" / "debug-loop" / "__pycache__" / "x.pyc").write_bytes(b"")
        self.assertEqual(self.errs(), [])

    def test_broken_include(self) -> None:
        (self.agent / "prompts" / "system.md").unlink()
        self.assertErr("!include target not found")

    def test_include_escape(self) -> None:
        (self.tmp / "outside.md").write_text("x")
        (self.agent / "configs" / "bad.yaml").write_text("x: !include ../../outside.md\n")
        self.assertErr("escapes submission root")

    def test_yaml_parse_error(self) -> None:
        (self.agent / "configs" / "bad.yaml").write_text("a: [unclosed\n")
        self.assertErr("YAML parse error")

    def test_disallowed_model(self) -> None:
        p = self.agent / "agent.yaml"
        p.write_text(p.read_text().replace("gemma-4-12b-it-qat-w4a16-ct", "gemma-3-27b-it"))
        self.assertErr("model not in allowed list")

    def test_mixed_models(self) -> None:
        (self.agent / "sub_agents").mkdir(exist_ok=True)
        (self.agent / "sub_agents" / "t.yaml").write_text("name: t\nmodel: gemma-4-e4b-it-qat-w4a16-ct\n")
        self.assertErr("more than one base model")

    def test_skill_without_name(self) -> None:
        (self.agent / "skills" / "x").mkdir()
        (self.agent / "skills" / "x" / "SKILL.md").write_text("---\ndescription: d\n---\nbody\n")
        self.assertErr("frontmatter with `name` required")

    def test_adapter_missing_weights(self) -> None:
        self.add_adapter("google/gemma-4-12b-it", weights=False)
        self.assertErr("adapter_model.safetensors missing")

    def test_adapter_wrong_size(self) -> None:
        self.add_adapter("google/gemma-4-31b-it")
        self.assertErr("trained for 31B but agent model is 12B")

    def test_adapter_matching_size(self) -> None:
        self.add_adapter("google/gemma-4-12b-it")
        self.assertEqual(self.errs(), [])


if __name__ == "__main__":
    unittest.main()
