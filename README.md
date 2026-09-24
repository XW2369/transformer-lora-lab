# transformer-lora-lab

Day5: 手写 Transformer + LoRA 微调

### 最终完整更新后的README全文（你直接覆盖）
```markdown
# 手搓多头注意力（MHA）

本项目基于PyTorch手写MultiHeadAttention，包含SDPA、split_heads/combine_heads、位置编码，并且和官方nn.MultiheadAttention做结果对拍校验。

## 环境安装
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch pytest
