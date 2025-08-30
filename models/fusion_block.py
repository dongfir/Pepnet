import torch
import torch.nn as nn

class ResidualFFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))


class GateFusion(nn.Module):
    """
    门控融合模块：对 x1 与 x2 执行自适应融合。
    输入：两个 [N, D] 的特征向量
    输出：融合后的 [N, D] 向量
    """
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)  # [N, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [N, 1]
        return gate * x1 + (1 - gate) * x2


class FusionBlock(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.1, fusion_type="cross_attention", bidirectional=False):
        super().__init__()
        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            from models.crossmodal_module import CrossModalAttention
            self.cross_attn = CrossModalAttention(
                hidden_dim=dim,
                heads=heads,
                n_layers=2,
                dropout=dropout,
                residual=True
            )
            self.gate_fusion = GateFusion(dim)
            self.ffn = ResidualFFN(dim, dropout)

        elif fusion_type == "add":
            self.norm = nn.LayerNorm(dim)

        elif fusion_type == "concat":
            self.linear = nn.Linear(dim * 2, dim)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x_seq, x_struct):
        if self.fusion_type == "cross_attention":
            fused = self.cross_attn(x_seq, x_struct)
            fused = self.gate_fusion(x_seq, fused)
            return self.ffn(fused)

        elif self.fusion_type == "add":
            return self.norm(x_seq + x_struct)

        elif self.fusion_type == "concat":
            fused = torch.cat([x_seq, x_struct], dim=-1)
            return self.norm(self.linear(fused))
