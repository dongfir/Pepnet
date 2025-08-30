import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class StructureFFN(nn.Module):
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

class ResidualGATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim // heads, heads=heads)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, x, edge_index):
        residual = x
        x = F.elu(self.gat(x, edge_index))
        x = self.dropout(x)

        if self.res_proj is not None:
            residual = self.res_proj(residual)

        return self.norm(x + residual)

class StructureGNNEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, heads=4, dropout=0.1, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(ResidualGATBlock(in_dim, hidden_dim, heads, dropout))

        self.ffn = StructureFFN(hidden_dim, dropout)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.ffn(x)
        return x  # [N, hidden_dim]
