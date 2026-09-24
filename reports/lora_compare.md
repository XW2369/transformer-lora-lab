# LoRA fine-tuning report

- Base model: `bert-base-chinese` (total params 102,564,098)
- LoRA r=8, alpha=16; replaced 24 Linear layers (query/value)
- Trainable params: **294,912** (0.2875%)
- Data: train 1000 / test 300 / max_len 48 / epochs 3

| metric | value |
|---|---|
| accuracy before | 0.4200 |
| accuracy after | 0.8367 |
| delta | +0.4167 |
| elapsed | 8.2 min |
