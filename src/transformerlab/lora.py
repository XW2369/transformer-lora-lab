"""手写 LoRA：W + B @ A，基座冻结，只训 A/B。"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout_p: float = 0.0):
        super().__init__()
        assert isinstance(base, nn.Linear), "base 必须是 nn.Linear"
        self.base = base
        # ★ 基座冻结：不再参与梯度
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r          # ★ 缩放因子 = α/r
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.reset_parameters()

    def reset_parameters(self):
        # ★ A 随机、B 全零 —— 保证训练开始前 ΔW = B @ A = 0，不扰动预训练权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        residual = self.base(x)                                    # [*, out]
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return residual + self.scaling * delta

    @torch.no_grad()
    def merged_weight(self):
        """推理时可合并：W' = W + scaling * B @ A，零额外延迟。"""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def _get_child(container, key: str):
    if isinstance(container, (nn.ModuleList, nn.Sequential)):
        return container[int(key)]
    if isinstance(container, nn.ModuleDict):
        return container[key]
    return getattr(container, key)


def _set_child(container, key: str, value):
    if isinstance(container, (nn.ModuleList, nn.Sequential)):
        container[int(key)] = value
    elif isinstance(container, nn.ModuleDict):
        container[key] = value
    else:
        setattr(container, key, value)


def inject_lora(model, target_names=("query", "value"), r=8, alpha=16, dropout_p=0.0):
    targets = [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and n.split(".")[-1] in target_names
    ]
    for name, old in targets:                 # ★ 先收集完再替换，避免边遍历边改
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = _get_child(parent, p)
        _set_child(parent, parts[-1], LoRALinear(old, r=r, alpha=alpha, dropout_p=dropout_p))
    return model, len(targets)


def mark_only_lora_as_trainable(model):
    for n, p in model.named_parameters():
        p.requires_grad = ("lora_A" in n) or ("lora_B" in n)
    return model


def lora_param_ratio(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / total
