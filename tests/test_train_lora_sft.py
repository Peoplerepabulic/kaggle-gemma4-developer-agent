"""Run: python -m unittest discover -s tests

fixtures/gemma4_chat_template.jinja is the Gemma 4 chat template as vendored by
llama.cpp (models/templates/google-gemma-4-31B-it.jinja), verbatim. Tests that
need transformers/tokenizers skip when they are not installed.
"""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_lora_sft as t  # noqa: E402

FIX = ROOT / "tests" / "fixtures"
ROW = json.loads((FIX / "sft_row.json").read_text())
TEMPLATE = (FIX / "gemma4_chat_template.jinja").read_text()
SPECIALS = ["<bos>", "<|turn>", "<turn|>", "<|think|>", "<|channel>", "<channel|>", "<|tool_call>",
            "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>']

EDIT_SPAN = ('<|tool_call>call:edit_file{filepath:<|"|>fastapi/sse.py<|"|>,new_string:<|"|>    return v.strip()'
             '<|"|>,old_string:<|"|>    return v<|"|>}<tool_call|><|tool_response>')
SUBMIT_SPAN = "<|channel>thought\nSubmit.\n<channel|><|tool_call>call:submit_patch{}<tool_call|><|tool_response>"


class CharTokenizer:
    """Char-level tokenizer with atomic special tokens and offsets (no deps)."""
    bos_token = "<bos>"
    chat_template = None
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def _id(self, s: str) -> int:
        if s not in self.vocab:
            self.vocab[s] = len(self.vocab) + 1
            self.inv[self.vocab[s]] = s
        return self.vocab[s]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offs, i = [], [], 0
        while i < len(text):
            sp = next((s for s in SPECIALS if text.startswith(s, i)), None)
            piece = sp or text[i]
            ids.append(self._id(piece))
            offs.append((i, i + len(piece)))
            i += len(piece)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.inv[i] for i in ids)


def jinja_render(messages, enable_thinking=True):
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    return env.from_string(TEMPLATE).render(messages=messages, bos_token="<bos>",
                                            enable_thinking=enable_thinking, add_generation_prompt=False)


def supervised_chunks(tok, ids, labels):
    chunks, cur = [], []
    for i, l in zip(ids, labels):
        if l != t.IGNORE:
            cur.append(i)
        elif cur:
            chunks.append(tok.decode(cur))
            cur = []
    if cur:
        chunks.append(tok.decode(cur))
    return chunks


class RenderTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("jinja2"), "jinja2 not installed")
    def test_port_matches_official_template(self) -> None:
        for system in (None, "You are an agent.\n"):
            for think in (True, False):
                msgs = t.normalize_messages(ROW["messages"])
                if system:
                    msgs = [{"role": "system", "content": system}] + msgs
                with self.subTest(system=bool(system), think=think):
                    self.assertEqual(t.render_segments(msgs, "<bos>", think).text, jinja_render(msgs, think))

    def test_tool_call_args_rendered_natively(self) -> None:
        text = t.render_segments(t.normalize_messages(ROW["messages"])).text
        self.assertIn('call:read_file{end_line:40,filepath:<|"|>fastapi/sse.py<|"|>,start_line:1}', text)
        self.assertNotIn('{"filepath"', text.split("<|tool_response>")[0])

    def test_format_argument(self) -> None:
        self.assertEqual(t.format_argument({"B": [1, True], "a": None}, escape_keys=False), "{a:None,B:[1,true]}")


class MaskTests(unittest.TestCase):
    def test_supervised_spans_are_exactly_patch_turns(self) -> None:
        tok = CharTokenizer()
        ex = t.build_example(ROW, tok, 8192, "<bos>", True, system_prompt="SYS")
        self.assertEqual(supervised_chunks(tok, ex.input_ids, ex.labels), [EDIT_SPAN, SUBMIT_SPAN])
        self.assertFalse(ex.truncated)

    def test_spans_match_rendered_patch_messages(self) -> None:
        tok = CharTokenizer()
        msgs = t.normalize_messages(ROW["messages"])
        r = t.render_segments(msgs)
        chunks = [r.text[s:e] for s, e in r.train_spans()]
        patch_names = [m["tool_calls"][0]["function"]["name"] for m in msgs if m.get("patch_producing")]
        self.assertEqual(len(chunks), len(patch_names))
        for chunk, name in zip(chunks, patch_names):
            self.assertIn(f"call:{name}", chunk)
            self.assertNotIn("response:", chunk)  # tool output never supervised
        ids, labels = t.encode(tok, r)
        self.assertEqual(supervised_chunks(tok, ids, labels), chunks)

    def test_no_patch_turns_means_no_example(self) -> None:
        row = copy.deepcopy(ROW)
        for m in row["messages"]:
            m["patch_producing"] = False
        self.assertIsNone(t.build_example(row, CharTokenizer(), 8192, "<bos>", True))

    def test_missing_flag_inferred_from_tool_name(self) -> None:
        row = copy.deepcopy(ROW)
        for m in row["messages"]:
            m.pop("patch_producing", None)
        tok = CharTokenizer()
        ex = t.build_example(row, tok, 8192, "<bos>", True)
        self.assertEqual(supervised_chunks(tok, ex.input_ids, ex.labels), [EDIT_SPAN, SUBMIT_SPAN])

    def test_segment_fallback_without_offsets(self) -> None:
        class NoOffsets(CharTokenizer):
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                if return_offsets_mapping:
                    raise NotImplementedError
                return super().__call__(text)

        tok = NoOffsets()
        ex = t.build_example(ROW, tok, 8192, "<bos>", True)
        self.assertEqual(supervised_chunks(tok, ex.input_ids, ex.labels), [EDIT_SPAN, SUBMIT_SPAN])


class TruncationTests(unittest.TestCase):
    def long_row(self):
        row = copy.deepcopy(ROW)
        filler = [{"role": "assistant", "content": "", "patch_producing": False,
                   "tool_calls": [{"id": f"f{i}", "type": "function",
                                   "function": {"name": "run_command", "arguments": json.dumps({"command": "ls"})}}]}
                  for i in range(6)]
        tools = [{"role": "tool", "tool_call_id": f"f{i}", "content": "x" * 400} for i in range(6)]
        steps = [m for pair in zip(filler, tools) for m in pair]
        row["messages"] = row["messages"][:1] + steps + row["messages"][1:]
        return row

    def test_left_truncation_keeps_head_and_patch_tail(self) -> None:
        tok = CharTokenizer()
        row = self.long_row()
        full = t.build_example(row, tok, 100000, "<bos>", True)
        ex = t.build_example(row, tok, len(full.input_ids) - 1000, "<bos>", True)
        self.assertTrue(ex.truncated)
        self.assertGreater(ex.dropped_steps, 0)
        self.assertLessEqual(len(ex.input_ids), len(full.input_ids) - 1000)
        text = tok.decode(ex.input_ids)
        self.assertIn("Problem Statement:\nSSE event ids", text)
        self.assertEqual(supervised_chunks(tok, ex.input_ids, ex.labels), [EDIT_SPAN, SUBMIT_SPAN])

    def test_dataset_counts_truncated_and_dropped(self) -> None:
        tok = CharTokenizer()
        early_patch = copy.deepcopy(self.long_row())
        msgs = early_patch["messages"]
        for m in msgs:
            m["patch_producing"] = False
        msgs[1]["patch_producing"] = True  # only the first body step is a patch turn
        msgs[1]["tool_calls"][0]["function"]["name"] = "edit_file"
        full_len = len(t.build_example(self.long_row(), tok, 100000, "<bos>", True).input_ids)
        args = SimpleNamespace(max_len=full_len - 1000, no_thinking=False)
        examples, stats = t.build_dataset([ROW, self.long_row(), early_patch], tok, args, None)
        self.assertEqual(stats["rows"], 3)
        self.assertEqual(stats["examples"], 2)
        self.assertEqual(stats["dropped_no_supervision"], 1)
        self.assertEqual(stats["truncated_rows"], 1)
        self.assertGreater(stats["supervised_pct"], 0)

    def test_huge_output_in_last_patch_step_is_shortened_not_dropped(self) -> None:
        tok = CharTokenizer()
        row = copy.deepcopy(ROW)
        row["messages"][8]["content"] = "y" * 5000  # submit_patch tool output, last patch step
        ex = t.build_example(row, tok, 2000, "<bos>", True)
        self.assertLessEqual(len(ex.input_ids), 2000)
        self.assertTrue(ex.truncated)
        self.assertIn("chars truncated", tok.decode(ex.input_ids))
        self.assertIn(SUBMIT_SPAN, supervised_chunks(tok, ex.input_ids, ex.labels))

    def test_early_patch_step_dropped_when_later_patch_remains(self) -> None:
        tok = CharTokenizer()
        row = copy.deepcopy(ROW)
        row["messages"][4]["content"] = "y" * 5000  # edit_file tool output
        ex = t.build_example(row, tok, 2000, "<bos>", True)
        self.assertEqual(supervised_chunks(tok, ex.input_ids, ex.labels), [SUBMIT_SPAN])
        self.assertEqual(ex.dropped_steps, 2)


class ExportHelperTests(unittest.TestCase):
    def test_infer_base_name(self) -> None:
        self.assertEqual(t.infer_base_name("/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b-it/1", None),
                         "google/gemma-4-12b-it")
        self.assertEqual(t.infer_base_name("/x/y", "google/gemma-4-e4b-it"), "google/gemma-4-e4b-it")
        with self.assertRaises(SystemExit):
            t.infer_base_name("/x/1", None)

    def test_validator_reads_base_name_as_12b(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import build_submission as bs
        self.assertEqual(bs._model_size(t.infer_base_name("/m/gemma-4-12b-it/1", None)), "12b")

    def test_target_regex_language_model_only(self) -> None:
        import re
        rx = re.compile(t.target_regex(t.DEFAULT_TARGETS.split(",")))
        for name in ("model.language_model.layers.3.self_attn.q_proj", "model.layers.0.mlp.down_proj",
                     "model.language_model.layers.11.self_attn.o_proj"):
            self.assertTrue(rx.fullmatch(name), name)
        for name in ("model.vision_tower.encoder.layers.0.self_attn.q_proj",
                     "model.audio_tower.layers.0.self_attn.q_proj",
                     "model.language_model.layers.0.per_layer_input_gate", "lm_head"):
            self.assertFalse(rx.fullmatch(name), name)
        only_attn = re.compile(t.target_regex(["q_proj", "o_proj"]))
        self.assertFalse(only_attn.fullmatch("model.layers.0.mlp.up_proj"))

    def test_rekey(self) -> None:
        k = "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
        self.assertEqual(t.rekey_name(k, "text"), "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight")
        self.assertEqual(t.rekey_name(k, "as-trained"), k)


def hf_char_tokenizer(texts):
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    chars = sorted({c for s in texts for c in s})
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<bos>", unk_token="[UNK]", pad_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": [s for s in SPECIALS if s != "<bos>"]})
    tok.chat_template = TEMPLATE
    return tok


@unittest.skipUnless(importlib.util.find_spec("transformers"), "transformers not installed")
class HFTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        sysmd = (ROOT / "agent" / "prompts" / "system.md").read_text()
        self.tok = hf_char_tokenizer([json.dumps(ROW), sysmd, TEMPLATE])

    def test_template_check_and_offsets(self) -> None:
        self.assertIsNone(t.check_template(self.tok, ROW, True, "SYS"))
        ex = t.build_example(ROW, self.tok, 8192, "<bos>", True, "SYS")
        chunks = supervised_chunks(self.tok, ex.input_ids, ex.labels)
        self.assertEqual([c.replace(" ", "") for c in chunks], [EDIT_SPAN.replace(" ", ""), SUBMIT_SPAN.replace(" ", "")])

    def test_dry_run_cli(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.tok.save_pretrained(d)
            data = Path(d) / "gold.jsonl"
            data.write_text(json.dumps(ROW) + "\n" + json.dumps(ROW) + "\n")
            res = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_lora_sft.py"), "--dry-run",
                                  "--tokenizer", d, "--data", str(data)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("chat template check: port == tokenizer.apply_chat_template on all 2 rows", res.stdout)
            stats = json.loads(res.stdout.split("STATS ", 1)[1].splitlines()[0])
            self.assertEqual((stats["examples"], stats["truncated_rows"]), (2, 0))
            self.assertTrue(1 <= stats["supervised_pct"] <= 40, stats)
            self.assertIn("⟦MASKED:", res.stdout)


if __name__ == "__main__":
    unittest.main()
