import torch
import torch.nn.functional as F
from torch import nn

from transformerlab.lora import (
    LoRALinear,
    inject_lora,
    lora_param_ratio,
    mark_only_lora_as_trainable,
)


def test_b_zero_init_gives_zero_delta():
    """B 全零 -> 初始 ΔW = 0，输出与原层完全一致。"""
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    layer = LoRALinear(base, r=4, alpha=8)
    layer.eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        assert (layer(x) - base(x)).abs().max().item() < 1e-6


def test_merge_weight_equals_forward():
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    layer = LoRALinear(base, r=4, alpha=8)
    nn.init.normal_(layer.lora_A, std=0.02)
    nn.init.normal_(layer.lora_B, std=0.02)
    layer.eval()
    x = torch.randn(3, 16)
    with torch.no_grad():
        via_module = layer(x)
        via_merge = F.linear(x, layer.merged_weight(), base.bias)
    assert (via_module - via_merge).abs().max().item() < 1e-5


def test_inject_lora_replaces_only_targets_and_freezes_base():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(8, 8)
            self.value = nn.Linear(8, 8)
            self.other = nn.Linear(8, 8)

    model, cnt = inject_lora(Tiny(), target_names=("query", "value"), r=2, alpha=4)
    mark_only_lora_as_trainable(model)

    assert cnt == 2
    assert isinstance(model.query, LoRALinear)
    assert isinstance(model.other, nn.Linear)
    assert model.other.weight.requires_grad is False
    trainable, total, ratio = lora_param_ratio(model)
    assert trainable == 64
    assert total == 280
    assert 0 < ratio < 1

    assert trainable == 64
    assert 0 < ratio < 1


def test_scaling_equals_alpha_over_r():
    layer = LoRALinear(nn.Linear(8, 8), r=4, alpha=8)
    assert abs(layer.scaling - 2.0) < 1e-12
