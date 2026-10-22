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

    # --- schema checks (official sample_submission + HARNESS_README) ---

    def set_root(self, text: str) -> None:
        (self.agent / "agent.yaml").write_text(text)

    def test_official_starter_agent_yaml_passes_schema(self) -> None:
        shutil.copy(ROOT / "tests" / "fixtures" / "starter_agent.yaml", self.agent / "agent.yaml")
        p = self.agent / "sub_agents" / "code_analyzer.yaml"
        p.write_text(p.read_text().replace("12b", "31b"))
        self.add_adapter("google/gemma-4-31b-it")
        (self.agent / "adapters" / "a1").rename(self.agent / "adapters" / "main_lora")
        self.assertEqual(self.errs(), [])

    def test_provider_prefix_normalized(self) -> None:
        p = self.agent / "agent.yaml"
        p.write_text(p.read_text().replace("model: gemma", "model: google/gemma"))
        self.assertEqual(self.errs(), [])

    def test_unknown_agent_key(self) -> None:
        p = self.agent / "agent.yaml"
        p.write_text(p.read_text() + "adapters:\n  - path: adapters/sft_v1\n")
        self.assertErr("unknown agent key `adapters`")

    def test_unknown_tool(self) -> None:
        p = self.agent / "agent.yaml"
        p.write_text(p.read_text().replace("  - get_status\n", "  - get_status\n  - web_search\n"))
        self.assertErr("tool not provided by harness: web_search")

    def test_dangling_agent_tool(self) -> None:
        (self.agent / "sub_agents" / "code_analyzer.yaml").unlink()
        self.assertErr("agent_tool config_path not found")

    def test_adapter_reference_without_folder(self) -> None:
        p = self.agent / "agent.yaml"
        p.write_text(p.read_text().replace("instruction:", "adapter: sft_v1\ninstruction:"))
        self.assertErr("adapter `sft_v1` has no folder")

    def test_stray_file_in_adapters(self) -> None:
        (self.agent / "adapters").mkdir(exist_ok=True)
        (self.agent / "adapters" / "README.md").write_text("x")
        self.assertErr("only adapter folders belong in adapters/")

    def test_sampling_wrapper_key(self) -> None:
        (self.agent / "configs" / "sampling.yaml").write_text("generation:\n  temperature: 0.2\n  max_tokens: 10\n")
        self.assertErr("generate_content_config field not allowed: `generation`")

    def test_max_output_tokens_ceiling(self) -> None:
        (self.agent / "configs" / "sampling.yaml").write_text("max_output_tokens: 40000\n")
        self.assertErr("max_output_tokens must be an int in 1..32768")

    def test_eval_config_old_layout(self) -> None:
        (self.agent / "eval_config.yaml").write_text("defaults:\n  max_time_minutes: 5\n")
        self.assertErr("budgets must sit under an `evaluation:` mapping")

    def test_eval_config_extra_keys(self) -> None:
        (self.agent / "eval_config.yaml").write_text(
            "evaluation:\n  max_time_minutes: 4\n  max_repair_rounds: 2\nretry:\n  enabled: true\n")
        self.assertErr("unknown evaluation key `max_repair_rounds`")
        self.assertErr("top-level key `retry` is ignored")

    def test_eval_config_bad_value(self) -> None:
        (self.agent / "eval_config.yaml").write_text("evaluation:\n  max_tool_calls: -1\n")
        self.assertErr("evaluation.max_tool_calls must be a positive number")

    def test_instruction_placeholders(self) -> None:
        p = self.agent / "prompts" / "system.md"
        p.write_text("Fix {problem_description}. {hints?} Use {{literal}}.\n")
        self.assertEqual(self.errs(), [])
        p.write_text("Use the {repo_name} repository.\n")
        self.assertErr("placeholder {repo_name} is not a harness state variable")

    def test_multiple_root_configs(self) -> None:
        shutil.copy(self.agent / "agent.yaml", self.agent / "root_agent.yaml")
        self.assertErr("multiple root configs")

    def test_include_cycle(self) -> None:
        (self.agent / "configs" / "a.yaml").write_text("x: !include configs/b.yaml\n")
        (self.agent / "configs" / "b.yaml").write_text("y: !include configs/a.yaml\n")
        self.assertErr("!include cycle")


if __name__ == "__main__":
    unittest.main()
