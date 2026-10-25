#!/usr/bin/env python3
"""LoRA SFT on filtered passing trajectories (PEFT -> .safetensors).

Scope of this version: pipeline validation (tokenize -> train -> export ->
package -> serve in harness), not a quality run.

Rendering: rows are rendered in the Gemma 4 chat format by `render_segments`,
a line-by-line port of the official chat template (tokenizer chat_template /
llama.cpp models/templates/google-gemma-4-31B-it.jinja). When the tokenizer
ships a chat template, every row's port output is checked against
`tokenizer.apply_chat_template` and a mismatch aborts (override:
--allow-template-mismatch). The port exists because label masking needs
character spans per message, which the template cannot report.

Labels: loss only on assistant messages with patch_producing=true: their
thought channel, tool calls, the `<|tool_response>` stop token the model emits
after its calls, and final content + `<turn|>`. Everything else is -100.

Truncation: rows over --max-len lose whole steps (assistant msg + its tool
results) from the left of the body, never the last patch-producing step; the
head (system + first user) and the tail are kept. If still too long, the
longest (masked) tool outputs are cut in the middle, then tokens from the left
of the body. A row with no supervised token left is dropped.

Training: LoRA r16/a32/d0.05 on the language model's q,k,v,o,gate,up,down,
bf16, gradient checkpointing, batch 1, cosine lr 1e-4, local weights only.
Export: agent/adapters/sft_v1/{adapter_config.json, adapter_model.safetensors},
keys renamed to the text-only layout (model.layers.*) that vLLM's
Gemma4ForCausalLM override serves; --key-style as-trained keeps PEFT's names.

  python scripts/train_lora_sft.py --dry-run --base /path/to/gemma-4-12b-it
  python scripts/train_lora_sft.py --base /kaggle/input/.../gemma-4-12b-it/1
  python scripts/train_lora_sft.py --rekey agent/adapters/sft_v1 --key-style as-trained
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PATCH_TOOLS = {"edit_file", "write_file", "submit_patch"}
DEFAULT_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
SIZE_RE = re.compile(r"gemma-?4-(31b|12b|e4b|e2b)(-it)?", re.IGNORECASE)
Q = '<|"|>'
IGNORE = -100


# ------------------------------------------------------------------ rendering

@dataclass
class Rendered:
    segments: list[tuple[str, bool]] = field(default_factory=list)  # (text, supervised)

    def add(self, text: str, train: bool = False) -> None:
        if text:
            if self.segments and self.segments[-1][1] == train:
                self.segments[-1] = (self.segments[-1][0] + text, train)
            else:
                self.segments.append((text, train))

    @property
    def text(self) -> str:
        return "".join(t for t, _ in self.segments)

    def train_spans(self) -> list[tuple[int, int]]:
        spans, pos = [], 0
        for t, train in self.segments:
            if train:
                spans.append((pos, pos + len(t)))
            pos += len(t)
        return spans


def _dictsort(d: dict) -> list[tuple[Any, Any]]:
    return sorted(d.items(), key=lambda kv: kv[0].lower() if isinstance(kv[0], str) else kv[0])


def format_argument(arg: Any, escape_keys: bool = True) -> str:
    if isinstance(arg, str):
        return Q + arg + Q
    if isinstance(arg, bool):
        return "true" if arg else "false"
    if isinstance(arg, dict):
        items = [(Q + k + Q if escape_keys else str(k)) + ":" + format_argument(v, escape_keys)
                 for k, v in _dictsort(arg)]
        return "{" + ",".join(items) + "}"
    if isinstance(arg, (list, tuple)):
        return "[" + ",".join(format_argument(i, escape_keys) for i in arg) + "]"
    return str(arg)


def strip_thinking(text: str) -> str:
    out = ""
    for part in text.split("<channel|>"):
        out += part.split("<|channel>")[0] if "<|channel>" in part else part
    return out.strip()


def tool_response_block(name: str, response: Any) -> str:
    if isinstance(response, dict):
        body = "response:" + name + "{" + ",".join(
            f"{k}:{format_argument(v, escape_keys=False)}" for k, v in _dictsort(response)) + "}"
    else:
        body = "response:" + name + "{value:" + format_argument(response, escape_keys=False) + "}"
    return "<|tool_response>" + body + "<tool_response|>"


def _content_text(content: Any, role: str) -> str:
    if isinstance(content, str):
        return strip_thinking(content) if role == "model" else content.strip()
    if isinstance(content, list):
        out = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                t = item.get("text", "")
                out += strip_thinking(t) if role == "model" else t.strip()
        return out
    return ""


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Parse JSON-string tool-call arguments into mappings, as vLLM does before templating."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.setdefault("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        parsed = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                        if isinstance(parsed, dict):
                            fn["arguments"] = parsed
                    except json.JSONDecodeError:
                        pass
                calls.append(tc)
            m["tool_calls"] = calls
        if m.get("role") == "assistant" and "patch_producing" not in m:
            m["patch_producing"] = any((c.get("function") or {}).get("name") in PATCH_TOOLS
                                       for c in m.get("tool_calls") or [])
        out.append(m)
    return out


def render_segments(messages: list[dict], bos: str = "<bos>", enable_thinking: bool = True) -> Rendered:
    """Port of the Gemma 4 chat template (no tool declarations, no generation prompt)."""
    r = Rendered()
    r.add(bos)
    loop = messages
    prev_type = None
    if enable_thinking or (messages and messages[0]["role"] in ("system", "developer")):
        r.add("<|turn>system\n")
        if enable_thinking:
            r.add("<|think|>\n")
        if messages and messages[0]["role"] in ("system", "developer"):
            r.add((messages[0].get("content") or "").strip())
            loop = messages[1:]
        r.add("<turn|>\n")
    last_user = max((i for i, m in enumerate(loop) if m["role"] == "user"), default=-1)

    for i, m in enumerate(loop):
        if m["role"] == "tool":
            continue
        prev_type = None
        role = "model" if m["role"] == "assistant" else m["role"]
        train = role == "model" and bool(m.get("patch_producing"))
        prev_nt = next((loop[j]["role"] for j in range(i - 1, -1, -1) if loop[j]["role"] != "tool"), None)
        if not (role == "model" and prev_nt == "assistant"):
            r.add("<|turn>" + role + "\n")
        thinking = m.get("reasoning") or m.get("reasoning_content")
        calls = m.get("tool_calls") or []
        if thinking and i > last_user and calls:
            r.add("<|channel>thought\n" + thinking + "\n<channel|>", train)
        for tc in calls:
            fn = tc["function"]
            args = fn.get("arguments")
            if isinstance(args, dict):
                body = ",".join(f"{k}:{format_argument(v, escape_keys=False)}" for k, v in _dictsort(args))
            else:
                body = args if isinstance(args, str) else ""
            r.add("<|tool_call>call:" + fn["name"] + "{" + body + "}<tool_call|>", train)
        if calls:
            prev_type = "tool_call"

        responded = False
        if m.get("tool_responses"):
            for tr in m["tool_responses"]:
                _add_response(r, tool_response_block(tr.get("name", "unknown"), tr.get("response")),
                              train and not responded)
                responded = True
            prev_type = "tool_response"
        elif calls:
            for k in range(i + 1, len(loop)):
                follow = loop[k]
                if follow["role"] != "tool":
                    break
                name = follow.get("name") or "unknown"
                for tc in calls:
                    if tc.get("id") == follow.get("tool_call_id"):
                        name = tc["function"]["name"]
                body = follow.get("content")
                if isinstance(body, list):
                    body = "".join(p.get("text", "") for p in body if isinstance(p, dict) and p.get("type") == "text")
                _add_response(r, tool_response_block(name, body), train and not responded)
                responded = True
                prev_type = "tool_response"

        content = m.get("content")
        r.add(_content_text(content, role), train)
        if prev_type == "tool_call" and not responded:
            r.add("<|tool_response>", train)
        elif not (responded and not content):
            r.add("<turn|>", train)
            r.add("\n")
    return r


def _add_response(r: Rendered, block: str, train_opener: bool) -> None:
    opener = "<|tool_response>"
    r.add(opener, train_opener)
    r.add(block[len(opener):])


# ------------------------------------------------------------- tokenization

def encode(tokenizer: Any, rendered: Rendered) -> tuple[list[int], list[int]]:
    """Token ids + labels (supervised tokens keep their id, others IGNORE)."""
    text = rendered.text
    spans = rendered.train_spans()
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
    except (NotImplementedError, TypeError, ValueError, KeyError):
        ids, labels = [], []
        for seg, train in rendered.segments:
            seg_ids = list(tokenizer(seg, add_special_tokens=False)["input_ids"])
            ids += seg_ids
            labels += seg_ids if train else [IGNORE] * len(seg_ids)
        return ids, labels
    labels, si = [], 0
    for tok, (s, e) in zip(ids, offsets):
        while si < len(spans) and spans[si][1] <= s:
            si += 1
        inside = si < len(spans) and spans[si][0] <= s < spans[si][1] and e > s
        labels.append(tok if inside else IGNORE)
    return ids, labels


def split_head_body(messages: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), -1)
    head, rest = messages[: first_user + 1], messages[first_user + 1:]
    steps: list[list[dict]] = []
    for m in rest:
        if m["role"] == "tool" and steps:
            steps[-1].append(m)
        else:
            steps.append([m])
    return head, steps


@dataclass
class Example:
    task_id: str
    sample_idx: Any
    input_ids: list[int]
    labels: list[int]
    rendered: Rendered
    full_tokens: int
    truncated: bool
    dropped_steps: int

    @property
    def supervised(self) -> int:
        return sum(1 for l in self.labels if l != IGNORE)


def build_example(row: dict, tokenizer: Any, max_len: int, bos: str, enable_thinking: bool,
                  system_prompt: str | None = None) -> Example | None:
    msgs = normalize_messages(row.get("messages") or [])
    if system_prompt and not (msgs and msgs[0]["role"] in ("system", "developer")):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    head, steps = split_head_body(msgs)
    dropped = 0
    rendered = render_segments(head + [m for s in steps for m in s], bos, enable_thinking)
    ids, labels = encode(tokenizer, rendered)
    full = len(ids)
    def has_patch(ss: list[list[dict]]) -> bool:
        return any(m.get("patch_producing") for s in ss for m in s)

    while len(ids) > max_len and len(steps) > 1 and (has_patch(steps[1:]) or not has_patch(steps)):
        steps = steps[1:]
        dropped += 1
        rendered = render_segments(head + [m for s in steps for m in s], bos, enable_thinking)
        ids, labels = encode(tokenizer, rendered)
    body = [dict(m) for s in steps for m in s]
    while len(ids) > max_len:  # still too long: shorten the longest tool output (masked) in the middle
        tools = [m for m in body if m["role"] == "tool" and isinstance(m.get("content"), str)]
        longest = max(tools, key=lambda m: len(m["content"]), default=None)
        if longest is None or len(longest["content"]) <= 400:
            break
        c = longest["content"]
        keep = len(c) // 4
        longest["content"] = c[:keep] + f"\n...[{len(c) - 2 * keep} chars truncated]...\n" + c[-keep:]
        rendered = render_segments(head + body, bos, enable_thinking)
        ids, labels = encode(tokenizer, rendered)
    if len(ids) > max_len:  # last resort: cut tokens from the left of the body
        head_len = len(encode(tokenizer, render_segments(head, bos, enable_thinking))[0])
        head_len = min(head_len, max_len // 2)
        keep = max_len - head_len
        ids, labels = ids[:head_len] + ids[-keep:], labels[:head_len] + labels[-keep:]
    ex = Example(row.get("task_id", "?"), row.get("sample_idx"), ids, labels, rendered,
                 full, full > max_len, dropped)
    return ex if ex.supervised > 0 else None


def check_template(tokenizer: Any, row: dict, enable_thinking: bool, system_prompt: str | None) -> str | None:
    """None if the port matches tokenizer.apply_chat_template for this row, else a diff snippet."""
    if not getattr(tokenizer, "chat_template", None):
        return "tokenizer has no chat_template (port used unchecked)"
    msgs = normalize_messages(row.get("messages") or [])
    if system_prompt and not (msgs and msgs[0]["role"] in ("system", "developer")):
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    ours = render_segments(msgs, tokenizer.bos_token or "", enable_thinking).text
    theirs = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                           enable_thinking=enable_thinking)
    if ours == theirs:
        return None
    i = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b), min(len(ours), len(theirs)))
    return f"first diff at char {i}: port={ours[i-80:i+80]!r} template={theirs[i-80:i+80]!r}"


# ------------------------------------------------------------------ reports

def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def show_masked(tokenizer: Any, ex: Example, elide: int = 300) -> str:
    out, i, ids, labels = [], 0, ex.input_ids, ex.labels
    while i < len(ids):
        train = labels[i] != IGNORE
        j = i
        while j < len(ids) and (labels[j] != IGNORE) == train:
            j += 1
        text = tokenizer.decode(ids[i:j], skip_special_tokens=False)
        if train:
            out.append(text)
        else:
            if elide and len(text) > 2 * elide:
                text = f"{text[:elide]} …[{len(text) - 2 * elide} chars]… {text[-elide:]}"
            out.append(f"⟦MASKED: {text}⟧")
        i = j
    return "".join(out)


def build_dataset(rows: list[dict], tokenizer: Any, args: argparse.Namespace,
                  system_prompt: str | None) -> tuple[list[Example], dict[str, Any]]:
    bos = tokenizer.bos_token or ""
    examples, dropped = [], []
    for row in rows:
        ex = build_example(row, tokenizer, args.max_len, bos, not args.no_thinking, system_prompt)
        (examples.append(ex) if ex else dropped.append(row.get("task_id")))
    sup = sum(e.supervised for e in examples)
    tot = sum(len(e.input_ids) for e in examples)
    stats = {"rows": len(rows), "examples": len(examples), "dropped_no_supervision": len(dropped),
             "truncated_rows": sum(e.truncated for e in examples), "tokens": tot,
             "supervised_tokens": sup, "supervised_pct": round(100 * sup / tot, 2) if tot else 0.0}
    return examples, stats


def print_report(examples: list[Example], stats: dict, tokenizer: Any, show_row: int, elide: int) -> None:
    print(f"{'task_id':<28} {'idx':>3} {'full':>7} {'tokens':>7} {'superv':>7} {'%':>6}  trunc")
    for e in examples:
        pct = 100 * e.supervised / len(e.input_ids)
        print(f"{e.task_id:<28} {str(e.sample_idx):>3} {e.full_tokens:>7} {len(e.input_ids):>7} "
              f"{e.supervised:>7} {pct:>6.2f}  {'yes (-%d steps)' % e.dropped_steps if e.truncated else ''}")
    print("STATS " + json.dumps(stats))
    bad = [e.task_id for e in examples if not 1 <= 100 * e.supervised / len(e.input_ids) <= 40]
    if bad:
        print(f"WARN: supervised % outside 1-40 for {bad}")
    if examples:
        ex = examples[min(show_row, len(examples) - 1)]
        print(f"\n===== sample {ex.task_id} #{ex.sample_idx} (⟦MASKED: …⟧ = no loss; rest = trained) =====")
        print(show_masked(tokenizer, ex, elide))


# ------------------------------------------------------------ train/export

def infer_base_name(base: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    m = SIZE_RE.search(base)
    if not m:
        raise SystemExit("cannot infer model size from --base path; pass --base-name google/gemma-4-12b-it")
    return f"google/gemma-4-{m.group(1).lower()}-it"


def target_regex(names: list[str]) -> str:
    attn = [n for n in names if n in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    mlp = [n for n in names if n in {"gate_proj", "up_proj", "down_proj"}]
    alts = ([rf"self_attn\.({'|'.join(attn)})"] if attn else []) + ([rf"mlp\.({'|'.join(mlp)})"] if mlp else [])
    return rf"model\.(language_model\.)?layers\.\d+\.({'|'.join(alts)})"


def rekey_name(key: str, style: str) -> str:
    return key.replace(".language_model.", ".") if style == "text" else key


def finalize_adapter(out: Path, base_name: str, targets: list[str], style: str) -> float:
    """Plain target list + base name in adapter_config, keys in the chosen layout, drop PEFT README."""
    from safetensors.torch import load_file, save_file

    cfg_path = out / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["base_model_name_or_path"] = base_name
    cfg["target_modules"] = targets
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    w = out / "adapter_model.safetensors"
    tensors = load_file(str(w))
    renamed = {rekey_name(k, style): v.contiguous() for k, v in tensors.items()}
    if len(renamed) != len(tensors):
        raise SystemExit("rekey produced duplicate tensor names")
    save_file(renamed, str(w), metadata={"format": "pt"})
    for extra in ("README.md",):
        (out / extra).unlink(missing_ok=True)
    print(f"adapter keys e.g. {next(iter(renamed))}  ({len(renamed)} tensors)")
    return w.stat().st_size / 1e6


def train(args: argparse.Namespace, examples: list[Example]) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
    base_name = infer_base_name(args.base, args.base_name)
    import transformers
    dtype_kw = "dtype" if tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"
    load_kw = {dtype_kw: torch.bfloat16, "local_files_only": True, "device_map": args.device_map,
               "attn_implementation": args.attn_impl}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, **load_kw)
    except ValueError:  # multimodal checkpoint without a CausalLM mapping
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.base, **load_kw)
    model.config.use_cache = False
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      target_modules=target_regex(targets), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    class DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, i: int) -> dict:
            return {"input_ids": examples[i].input_ids, "labels": examples[i].labels}

    pad = getattr(args, "pad_id", 0)

    def collate(batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = [b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]
        lab = [b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch]
        att = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}

    targs = TrainingArguments(
        output_dir=args.ckpt_dir, per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=1,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=args.optim, logging_steps=1, save_strategy="no", report_to="none",
        remove_unused_columns=False, seed=args.seed, dataloader_num_workers=0, use_cpu=args.cpu,
    )
    Trainer(model=model, args=targs, train_dataset=DS(), data_collator=collate).train()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    mb = finalize_adapter(out, base_name, targets, args.key_style)
    print(f"saved adapter to {out}  ({mb:.1f} MB, base {base_name}, keys {args.key_style})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "sft_gold.jsonl"))
    ap.add_argument("--base", help="LOCAL path to the unquantized twin (e.g. gemma-4-12b-it); never downloaded")
    ap.add_argument("--tokenizer", help="local tokenizer path (default: --base)")
    ap.add_argument("--base-name", help="written to adapter_config (default: inferred from --base path)")
    ap.add_argument("--out", default=str(ROOT / "agent" / "adapters" / "sft_v1"))
    ap.add_argument("--ckpt-dir", default=str(ROOT / "data" / "sft_ckpt"), help="Trainer scratch (no checkpoints saved)")
    ap.add_argument("--system-prompt", default=str(ROOT / "agent" / "prompts" / "system.md"),
                    help="prepended when a row has no system message; '' to disable")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--no-thinking", action="store_true", help="render without <|think|> (scored config thinks)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--optim", default="adamw_torch", help="e.g. adamw_torch, adamw_torch_fused, adafactor")
    ap.add_argument("--target-modules", default=DEFAULT_TARGETS)
    ap.add_argument("--key-style", choices=["text", "as-trained"], default="text",
                    help="text: model.layers.* (vLLM Gemma4ForCausalLM); as-trained: PEFT names unchanged")
    ap.add_argument("--device-map", default="auto", help="'auto' shards 12B bf16 across the 4 L4s")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true", help="CPU wiring test on a tiny model only")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer only: stats + one masked sample")
    ap.add_argument("--show-row", type=int, default=0)
    ap.add_argument("--elide", type=int, default=300, help="masked spans longer than 2x this are elided (0 = full)")
    ap.add_argument("--allow-template-mismatch", action="store_true")
    ap.add_argument("--rekey", metavar="ADAPTER_DIR", help="only re-export an existing adapter with --key-style")
    args = ap.parse_args()

    if args.rekey:
        if args.key_style != "text":
            ap.error("--rekey converts an as-trained adapter to --key-style text; the reverse needs re-training")
        out = Path(args.rekey)
        cfg = json.loads((out / "adapter_config.json").read_text())
        targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
        mb = finalize_adapter(out, args.base_name or cfg.get("base_model_name_or_path", ""), targets, args.key_style)
        print(f"rekeyed {out} ({mb:.1f} MB)")
        return 0

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok_path = args.tokenizer or args.base
    if not tok_path:
        ap.error("--base (or --tokenizer for --dry-run) is required")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
    rows = load_jsonl(args.data)
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8") if args.system_prompt else None
    if not rows:
        raise SystemExit(f"no rows in {args.data}")

    mismatches = [(r.get("task_id"), d) for r in rows
                  if (d := check_template(tokenizer, r, not args.no_thinking, system_prompt))]
    if not mismatches:
        print(f"chat template check: port == tokenizer.apply_chat_template on all {len(rows)} rows")
    for tid, d in mismatches[:3]:
        print(f"TEMPLATE {tid}: {d}")
    if mismatches and getattr(tokenizer, "chat_template", None) and not args.allow_template_mismatch:
        raise SystemExit("rendering does not match the tokenizer's chat template; "
                         "fix the port or pass --allow-template-mismatch")

    examples, stats = build_dataset(rows, tokenizer, args, system_prompt)
    print_report(examples, stats, tokenizer, args.show_row, args.elide)
    if args.dry_run:
        return 0
    if not args.base:
        ap.error("--base is required for training")
    args.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train(args, examples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
