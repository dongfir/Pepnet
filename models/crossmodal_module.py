import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalCrossAttentionLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.prot_to_pep = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.pep_to_prot = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        self.proj_gate = nn.Linear(dim * 2, 1)
        self.norm_prot = nn.LayerNorm(dim)
        self.norm_pep = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_prot, x_pep):
        # Protein ← Peptide
        out1, _ = self.prot_to_pep(query=x_prot, key=x_pep, value=x_pep)
        gate1 = torch.sigmoid(self.proj_gate(torch.cat([x_prot, out1], dim=-1)))
        x_prot_fused = self.norm_prot(x_prot + gate1 * self.dropout(out1))

        # Peptide ← Protein
        out2, _ = self.pep_to_prot(query=x_pep, key=x_prot, value=x_prot)
        gate2 = torch.sigmoid(self.proj_gate(torch.cat([x_pep, out2], dim=-1)))
        x_pep_fused = self.norm_pep(x_pep + gate2 * self.dropout(out2))

        return x_prot_fused, x_pep_fused


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim=256, heads=4, n_layers=2, dropout=0.1, residual=True):
        super().__init__()
        self.layers = nn.ModuleList([
            BidirectionalCrossAttentionLayer(hidden_dim, heads, dropout)
            for _ in range(n_layers)
        ])
        self.residual = residual
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_prot, x_pep):
        """
        x_prot: [N, D] - structural features
        x_pep:  [N, D] - sequence features
        """
        x1, x2 = x_prot.unsqueeze(0), x_pep.unsqueeze(0)  # [1, N, D]

        for layer in self.layers:
            x1, x2 = layer(x1, x2)
            x1 = F.gelu(x1)
            x2 = F.gelu(x2)

        fused = (x1 + x2) / 2  # Bidirectional fusion
        return self.final_norm(fused.squeeze(0))
