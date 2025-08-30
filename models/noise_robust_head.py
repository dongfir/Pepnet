# models/noise_robust_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .heads_cosine import CosineMarginHead, two_class_to_binary_logit

class NoiseRobustClassifierHead(nn.Module):
    """
    三路特征（x_seq, x_gnn, x_fused）→
      - node_logits：最终二分类 logit（自适应融合）
      - 训练辅助：seq_logits/gnn_logits/fused_logits
      - sample_weight：样本级权重（降噪；mean≈1）
      - agreement：模态一致性分数（0..1）
    支持可调：
      * 余弦头超参：s, m
      * 融合权重：fused_base, fused_gain, aux_gain  （final = fused*(fused_base+fused_gain*r) + avg_aux*(aux_gain*r)）
      * 样本权重幂：weight_gamma（>1 更抑制“低一致性”脏样本）
    """
    def __init__(
        self,
        hidden_dim=256,
        head_type="cosine",
        s: float = 30.0,
        m: float = 0.20,
        gate_dim: int = 64,
        fused_base: float = 0.70,
        fused_gain: float = 0.30,
        aux_gain: float = 0.30,
        weight_gamma: float = 1.00,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_type = head_type
        self.fused_base = float(fused_base)
        self.fused_gain = float(fused_gain)
        self.aux_gain   = float(aux_gain)
        self.weight_gamma = float(weight_gamma)

        # 单模态分支
        self.seq_head = nn.Linear(hidden_dim, 1)
        self.gnn_head = nn.Linear(hidden_dim, 1)

        # 融合分支：cosine 或 linear
        if head_type == "cosine":
            self.fused_head = CosineMarginHead(feat_dim=hidden_dim, s=s, m=m, mode="cosface")
            self.use_cosine = True
        else:
            self.fused_head = nn.Linear(hidden_dim, 1)
            self.use_cosine = False

        # gate 输入（模态/融合投影 + 差分/乘积 + 余弦相似 + 概率一致性）
        self.proj_s = nn.Linear(hidden_dim, gate_dim)
        self.proj_g = nn.Linear(hidden_dim, gate_dim)
        self.proj_f = nn.Linear(hidden_dim, gate_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_dim * 5 + 2, gate_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_dim, 1)
        )

    def forward(self, x_seq, x_gnn, x_fused, labels=None, return_aux=True):
        # 分支 logits
        l_seq = self.seq_head(x_seq).squeeze(-1)
        l_gnn = self.gnn_head(x_gnn).squeeze(-1)

        if self.use_cosine:
            two = self.fused_head(x_fused, labels) if (self.training and labels is not None) else self.fused_head(x_fused)
            l_fused = two_class_to_binary_logit(two)
        else:
            l_fused = self.fused_head(x_fused).squeeze(-1)

        # 概率与一致性
        p_seq = torch.sigmoid(l_seq)
        p_gnn = torch.sigmoid(l_gnn)
        agreement = 1.0 - (p_seq - p_gnn).abs()  # (0..1)

        # gate r（0..1）
        zs = F.normalize(self.proj_s(x_seq), dim=-1)
        zg = F.normalize(self.proj_g(x_gnn), dim=-1)
        zf = F.normalize(self.proj_f(x_fused), dim=-1)

        z_abs = (zs - zg).abs()
        z_mul = zs * zg
        cos_sg = (zs * zg).sum(-1, keepdim=True).clamp(-1, 1)
        agr = agreement.unsqueeze(-1)

        gate_in = torch.cat([zf, zs, zg, z_abs, z_mul, cos_sg, agr], dim=-1)
        r = torch.sigmoid(self.gate_mlp(gate_in)).squeeze(-1)

        # 自适应融合（参数可调）
        avg_aux = 0.5 * (l_seq + l_gnn)
        node_logits = l_fused * (self.fused_base + self.fused_gain * r) + avg_aux * (self.aux_gain * r)

        # 样本权重：w_raw = 0.5*r + 0.5*agreement  → 幂次增强 → 归一（mean≈1）
        with torch.no_grad():
            w_raw = 0.5 * r + 0.5 * agreement
            w = torch.pow(w_raw.clamp_min(1e-6), self.weight_gamma)
            w = w / (w.mean() + 1e-8)

        if not return_aux:
            return node_logits

        aux = {
            "seq_logits": l_seq,
            "gnn_logits": l_gnn,
            "fused_logits": l_fused,
            "agreement": agreement.detach(),
            "sample_weight": w.detach(),
        }
        return node_logits, aux
