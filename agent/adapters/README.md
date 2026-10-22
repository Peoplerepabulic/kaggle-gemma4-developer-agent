# Adapters

Trained PEFT LoRA adapters go here, one folder per adapter:

```
adapters/sft_v1/adapter_config.json
adapters/sft_v1/adapter_model.safetensors
```

Train on the unquantized twin of the chosen QAT base model. If the base model is not 31B, do NOT reuse the starter notebook's dummy adapters (they are 31B-shaped).
