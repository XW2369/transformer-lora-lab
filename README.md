transformer-lora-lab
Day5：手写 Transformer + LoRA 微调（CPU 环境，bert-base-chinese + ChnSentiCorp 中文情感二分类）

项目结构
src/transformerlab/attention.py — 手写 SDPA、MultiHeadAttention、因果 mask、正弦位置编码
src/transformerlab/transformer.py — Pre-LN Transformer Encoder（FFN + 残差 + LayerNorm）
src/transformerlab/lora.py — 手写 LoRALinear（低秩 A/B 分解）+ 注入 + 基座冻结
scripts/finetune_lora.py — 中文情感分类微调主脚本
scripts/plot_lora_curve.py — 结果绘图
tests/ — 13 条 pytest 用例（含与 nn.MultiheadAttention 对拍 + 负向对照）
环境
本项目使用 .venv 虚拟环境（不执行 activate，直接用解释器绝对路径调用）：

.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install transformers datasets scikit-learn matplotlib pytest ruff
.\.venv\Scripts\python.exe -m pip install -e .
复现
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe scripts\finetune_lora.py --n-train 1000 --n-test 300 --max-len 48 --epochs 3
.\.venv\Scripts\python.exe scripts\plot_lora_curve.py
结果
手写 MHA 与 nn.MultiheadAttention 最大误差：0.000e+00（d_model=16）/ 9.6857548e-08（d_model=32，即 float32 机器精度 eps≈1.19e-07）
LoRA（r=8, alpha=16，注入 query/value）：可训练参数 294,912 / 102,564,098 = 0.2875%，替换 24 个 Linear 层
ChnSentiCorp：微调前 acc 0.4200 → 微调后 0.8367（+41.67 pp）
逐轮曲线：acc 0.6233 → 0.6800 → 0.8367；train loss 0.7016 → 0.6363 → 0.4848
CPU 训练总耗时 8.2 min（1000 train / 300 test / max_len 48 / 3 epoch）
数据说明
seamew/ChnSentiCorp 为 script 数据集，datasets>=5.0 已不支持。本项目改用其 refs/convert/parquet 转换分支，已固化为本地 parquet 文件读取。