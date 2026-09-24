# transformer.py

from torch import nn

from .attention import MultiHeadAttention


class PositionwiseFFN(nn.Module):
    """fc1 → GELU → Dropout → fc2"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Pre-LN: x + drop(attn(norm1(x))) → x + drop(ffn(norm2(x)))"""
    def __init__(self, dim: int, n_heads: int, ffn_hidden_dim: int, dropout: float =0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = PositionwiseFFN(dim, ffn_hidden_dim, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # multi-head attention residual
        attn_out = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop1(attn_out)

        # ffn residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop2(ffn_out)
        return x


class TransformerEncoder(nn.Module):
    """ModuleList堆叠 + 尾部LayerNorm"""
    def __init__(self, dim:int, n_heads:int, ffn_hidden_dim:int, num_layers:int, dropout:float=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(dim, n_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        for blk in self.layers:
            x = blk(x, mask=mask)
        x = self.final_norm(x)
        return x
