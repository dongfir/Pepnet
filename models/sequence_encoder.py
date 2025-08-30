import torch
import torch.nn as nn

class MultiScaleConv(nn.Module):
    def __init__(self, dim, kernel_sizes=(3, 5, 7), dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=k, padding=k//2, groups=dim) for k in kernel_sizes
        ])
        self.pointwise = nn.Conv1d(dim * len(kernel_sizes), dim, kernel_size=1)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        conv_outs = [conv(x) for conv in self.convs]
        x_cat = torch.cat(conv_outs, dim=1)  # [B, D * K, N]
        x_out = self.pointwise(x_cat)  # [B, D, N]
        x_out = x_out.transpose(1, 2)  # [B, N, D]
        return self.norm(x_out)

class LocalGateAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x_raw, x_attn):
        gate_input = torch.cat([x_raw, x_attn], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, N, 1]
        return gate * x_raw + (1 - gate) * x_attn

class SequenceEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 4,
                                       dropout=dropout, batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.multi_scale_conv = MultiScaleConv(hidden_dim, dropout=dropout)
        self.gate = LocalGateAttention(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x).unsqueeze(0)  # [1, N, D]
        for layer in self.transformers:
            x = layer(x)  # → [1, N, D]
        x_conv = self.multi_scale_conv(x)  # → [1, N, D]
        x = self.gate(x, x_conv)  # 门控融合
        x = self.norm(x)
        return x.squeeze(0)
