# Kaggle notebook workflow

The scored artifact is produced by a Kaggle notebook, not from this repo directly.

1. Copy the official **Getting Started** notebook (Kaggle Staff) → Copy & Edit.
2. In the notebook: attach the competition dataset + the chosen Gemma 4 model, accelerator = L4x4, internet = OFF.
3. Run unchanged once → baseline submission.zip at `/kaggle/working/submission.zip` → Submit.
4. For our versions: upload `agent/` as a Kaggle Dataset (or paste the configs), replace the starter's packaging cell with the contents of `../scripts/build_submission.py` logic, and re-run.
5. Keep one notebook version per submission; name them `v1-baseline`, `v2-harness`, `v3-lora`, `v4-verifier`… and log each in `../docs/experiments.md`.
