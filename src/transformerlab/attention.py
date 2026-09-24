import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    缩放点积注意力
    q,k,v: [batch, n_head, seq_len, d_k]
    mask: float矩阵，1=可见，0=屏蔽；屏蔽位置要填 -1e9
    """
    d_k = q.size(-1)
    attn_score = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
    attn_weight = F.softmax(attn_score, dim=-1)
    if dropout_p > 0:
        attn_weight = F.dropout(attn_weight, p=dropout_p)
    out = torch.matmul(attn_weight, v)
    return out, attn_weight

if __name__ == "__main__":
    # 测试 scaled_dot_product_attention
    b, h, s, d = 2,1,4,16
    q = torch.randn(b, h, s, d)
    k = torch.randn(b, h, s, d)
    v = torch.randn(b, h, s, d)
    out, w = scaled_dot_product_attention(q,k,v)
    print("output shape:", out.shape)
    print("attn weight shape:", w.shape)
    assert out.shape == (b,h,s,d)
    print("✅ scaled_dot_product_attention 测试通过！")
def build_causal_mask(seq_len):
    """
    返回float类型因果mask，1=可见，0=屏蔽
    shape [seq_len, seq_len]，上三角=0（看不到未来token），下三角+对角线=1
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.float32), diagonal=0)
    return mask
def split_heads(x, n_heads):
    B, N, d_model = x.shape
    d_k = d_model // n_heads
    x = x.reshape(B, N, n_heads, d_k)
    x = x.permute(0,2,1,3)   # [B,N,h,dk] → [B,h,N,dk]
    return x

def combine_heads(x):
    """
    输入 x: [B, h, N, d_k]
    返回: [B, N, d_model]
    ✨重点：permute之后 .contiguous()
    """
    B, h, N, d_k = x.shape
    x = x.permute(0,2,1,3)  # [B, h, N, d_k] → [B, N, h, d_k]
    x = x.contiguous()       # 【必加！】解决 view/reshape 内存不连续报错
    x = x.reshape(B, N, h * d_k)
    return x
class MultiHeadAttention(nn.Module):
    """手写多头自注意力。接口对齐 nn.MultiheadAttention，便于对拍。"""

    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def _split(self, t: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        return t.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None, return_weights: bool = False):
        batch, seq_len, _ = x.shape
        q = self._split(self.q_proj(x), batch, seq_len)
        k = self._split(self.k_proj(x), batch, seq_len)
        v = self._split(self.v_proj(x), batch, seq_len)

        out, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.o_proj(out)
        return (out, attn) if return_weights else out

    @torch.no_grad()
    def load_from_torch_mha(self, ref: nn.MultiheadAttention) -> "MultiHeadAttention":
        """把官方权重搬进来，用于对拍。顺序是 q;k;v，不能塞反。"""
        q_w, k_w, v_w = ref.in_proj_weight.chunk(3, dim=0)
        q_b, k_b, v_b = ref.in_proj_bias.chunk(3, dim=0)
        self.q_proj.weight.copy_(q_w)
        self.q_proj.bias.copy_(q_b)
        self.k_proj.weight.copy_(k_w)
        self.k_proj.bias.copy_(k_b)
        self.v_proj.weight.copy_(v_w)
        self.v_proj.bias.copy_(v_b)
        self.o_proj.weight.copy_(ref.out_proj.weight)
        self.o_proj.bias.copy_(ref.out_proj.bias)
        return self

if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, n_heads, seq_len, batch = 16, 4, 5, 2
    x = torch.randn(batch, seq_len, d_model)

    ref = nn.MultiheadAttention(
        embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=0.0
    )
    mine = MultiHeadAttention(d_model, n_heads, dropout_p=0.0)
    mine.load_from_torch_mha(ref)
    ref.eval()
    mine.eval()

    o_ref, _ = ref(x, x, x)
    o_mine = mine(x)
    diff = (o_ref - o_mine).abs().max().item()

    print("ref  shape:", tuple(o_ref.shape))
    print("mine shape:", tuple(o_mine.shape))
    print(f"max abs diff: {diff:.3e}")
    assert diff < 1e-5, f"对拍未通过: {diff}"
    print("[OK] 手写 MHA 与 nn.MultiheadAttention 对拍通过")
