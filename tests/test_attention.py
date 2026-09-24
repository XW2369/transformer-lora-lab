import torch

from transformerlab.attention import (
    MultiHeadAttention,
    build_causal_mask,
    combine_heads,
    scaled_dot_product_attention,
    split_heads,
)
from transformerlab.transformer import (
    PositionwiseFFN,
    TransformerEncoder,
    TransformerEncoderBlock,
)


def test_split_heads():
    B, N, d_model = 2, 5, 32
    x = torch.randn(B, N, d_model)
    out = split_heads(x, n_heads=4)
    assert out.shape == (B, 4, N, 8)


def test_combine_heads():
    B, N, h, dk = 2, 5, 4, 8
    x = torch.randn(B, h, N, dk)
    out = combine_heads(x)
    assert out.shape == (B, N, h * dk)


def test_sdpa_shape():
    B, h, N, dk = 2, 4, 5, 8
    q, k, v = torch.randn(B, h, N, dk), torch.randn(B, h, N, dk), torch.randn(B, h, N, dk)
    mask = build_causal_mask(N)
    attn_out, _ = scaled_dot_product_attention(q, k, v, mask)
    assert attn_out.shape == q.shape


def test_mha_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    B, N = 2, 5
    x = torch.randn(B, N, 32)
    out = mha(x)  # ★ return_weights=False -> 返回张量，不是元组
    assert out.shape == (B, N, 32)


def test_mha_against_torch_ref():
    """对拍：手写 MHA 与 torch 官方 MHA 校验，误差 < 1e-5"""
    d_model, n_heads = 32, 4
    ref = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
    mine = MultiHeadAttention(d_model, n_heads, dropout_p=0.0).load_from_torch_mha(ref)
    mine.eval()
    ref.eval()

    B, N = 2, 5
    x = torch.randn(B, N, d_model)
    mask = build_causal_mask(N)
    ref_mask = (mask == 0).bool()  # 官方语义：True = 屏蔽

    out_ref, _ = ref(x, x, x, attn_mask=ref_mask)
    out_mine = mine(x, mask=mask)
    diff = torch.max(torch.abs(out_ref - out_mine))
    assert diff < 1e-5, f"误差太大：{diff.item()}"


def test_mha_negative_control_mismatched_weights():
    """负向对照：不搬权重时误差必须很大，证明上面那条对拍不是永远为真。"""
    torch.manual_seed(0)
    d_model, n_heads = 32, 4
    ref = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.0)
    mine = MultiHeadAttention(d_model, n_heads, dropout_p=0.0)
    ref.eval()
    mine.eval()
    x = torch.randn(2, 5, d_model)
    out_ref, _ = ref(x, x, x)
    assert (out_ref - mine(x)).abs().max().item() > 1e-3


def test_transformer_encoder_shape():
    B, L, D = 2, 10, 32
    model = TransformerEncoder(dim=D, n_heads=4, ffn_hidden_dim=64, num_layers=2)
    x = torch.randn(B, L, D)
    out = model(x)
    assert out.shape == (B, L, D)


def test_encoder_block_shape():
    B, L, D = 2, 10, 32
    blk = TransformerEncoderBlock(dim=D, n_heads=4, ffn_hidden_dim=64)
    x = torch.randn(B, L, D)
    o = blk(x)
    assert o.shape == (B, L, D)


def test_ffn_shape():
    B, L, D = 2, 10, 32
    ffn = PositionwiseFFN(dim=D, hidden_dim=64)
    x = torch.randn(B, L, D)
    o = ffn(x)
    assert o.shape == (B, L, D)
