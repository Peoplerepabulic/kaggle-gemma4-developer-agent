# Adapters

Trained PEFT LoRA adapters go in `agent/adapters/`, one folder per adapter. Reference one from an agent YAML with `adapter: <folder name>` (name must match `[A-Za-z0-9][A-Za-z0-9._-]*`):

```
adapters/sft_v1/adapter_config.json
adapters/sft_v1/adapter_model.safetensors
```

Train on the unquantized twin of the chosen QAT base model. If the base model is not 31B, do NOT reuse the starter notebook's dummy adapters (they are 31B-shaped).
