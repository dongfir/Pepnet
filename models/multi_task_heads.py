import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool

class NodeClassifierHead(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.mlp(x).squeeze(-1)  # [N]

class GlobalClassifierHead(nn.Module):
    def __init__(self, hidden_dim=256, pool_type='mean'):
        super().__init__()
        self.pool_type = pool_type
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, batch):
        if self.pool_type == 'mean':
            pooled = global_mean_pool(x, batch)
        else:
            pooled = global_max_pool(x, batch)
        return self.linear(pooled).squeeze(-1)  # [B]

class MaskDecoderHead(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.decoder = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.decoder(x)  # 用于掩码节点重建

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for classification loss
        self.beta = beta  # Weight for regression loss
        self.class_loss_fn = nn.BCEWithLogitsLoss()
        self.reg_loss_fn = nn.MSELoss()

    def forward(self, cls_logits, cls_targets, reg_logits, reg_targets):
        cls_loss = self.class_loss_fn(cls_logits, cls_targets)
        reg_loss = self.reg_loss_fn(reg_logits, reg_targets)
        return self.alpha * cls_loss + self.beta * reg_loss
