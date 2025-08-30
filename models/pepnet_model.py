import torch
import torch.nn as nn
from config import Config as cfg
from models.sequence_encoder import SequenceEncoder
from models.structure_gnn_encoder import StructureGNNEncoder
from models.multi_task_heads import GlobalClassifierHead, MultiTaskLoss
from models.fusion_block import FusionBlock
from models.crossmodal_module import CrossModalAttention
from models.noise_robust_head import NoiseRobustClassifierHead

class PepNet(nn.Module):
    """
    v6-lite：噪声鲁棒版 PepNet

    组件：
      - 编码器：SequenceEncoder（序列） + StructureGNNEncoder（结构）
      - 跨模态：CrossModalAttention -> 残差融合 FusionBlock
      - 头部：NoiseRobustClassifierHead（cosine margin + gate 自适应融合 + 样本权重）

    forward 返回：
      {
        "node_logits":   (N,),   # 主二分类 logit（用于 BCE + 阈扫）
        "seq_logits":    (N,),   # 辅助：序列分支 logit
        "gnn_logits":    (N,),   # 辅助：结构分支 logit
        "fused_logits":  (N,),   # 辅助：融合分支 logit
        "sample_weight": (N,),   # 样本权重（mean≈1），用于降噪加权 BCE
        "agreement":     (N,),   # 模态一致性分数（0..1）
        "global_logits": (B,)    # 图级二分类（若 use_global_task=True）
      }
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_global_task: bool = False,
        use_mask_task: bool = False,
        dropout: float = 0.1,
        head_type: str = "cosine",
        head_s: float = None,
        head_m: float = None,
        gate_dim: int = None,
        fused_base: float = None,
        fused_gain: float = None,
        aux_gain: float = None,
        weight_gamma: float = None,
    ):
        super().__init__()
        self.use_global = use_global_task
        self.use_mask = use_mask_task
        self.hidden_dim = hidden_dim

        # === 模态编码器 ===
        self.seq_encoder = SequenceEncoder(input_dim=in_channels, hidden_dim=hidden_dim)

        # 兼容不同签名的 StructureGNNEncoder
        try:
            self.struct_encoder = StructureGNNEncoder(
                input_dim=in_channels, hidden_dim=hidden_dim,
                num_layers=num_layers, dropout=dropout
            )
        except TypeError:
            self.struct_encoder = StructureGNNEncoder(
                input_dim=in_channels, hidden_dim=hidden_dim
            )

        # === 跨模态注意力 + 残差融合 ===
        self.cross_attention = CrossModalAttention(
            hidden_dim=hidden_dim, heads=4, n_layers=2, dropout=dropout, residual=True
        )
        self.residual_fusion = FusionBlock(hidden_dim)

        # === 噪声鲁棒头：余弦间隔 + gate 自适应融合 + 样本权重 ===
        self.node_head = NoiseRobustClassifierHead(
            hidden_dim=hidden_dim,
            head_type=head_type,
            s=head_s if head_s is not None else cfg.head_s,
            m=head_m if head_m is not None else cfg.head_m,
            gate_dim=gate_dim if gate_dim is not None else 64,
            fused_base=fused_base if fused_base is not None else cfg.gate_fused_base,
            fused_gain=fused_gain if fused_gain is not None else cfg.gate_fused_gain,
            aux_gain=aux_gain if aux_gain is not None else cfg.gate_aux_gain,
            weight_gamma=weight_gamma if weight_gamma is not None else cfg.weight_gamma,
        )

        # === 图级任务 ===
        if self.use_global:
            self.global_head = GlobalClassifierHead(hidden_dim)

        self.multi_task_loss_fn = MultiTaskLoss(alpha=0.7, beta=0.3)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = getattr(
            data, "batch",
            torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        )

        # 1) 双模态编码
        x_seq = self.seq_encoder(x)                 # (N, D)
        x_gnn = self.struct_encoder(x, edge_index)  # (N, D)

        # 2) 跨模态注意力 + 残差融合
        x_cross = self.cross_attention(x_seq, x_gnn)    # (N, D)
        x_fused = self.residual_fusion(x_seq, x_cross)  # (N, D)

        # 3) 噪声鲁棒头（训练时传 labels 触发余弦 margin；评估不传）
        labels = data.y if (self.training and hasattr(data, "y")) else None
        node_logits, aux = self.node_head(
            x_seq, x_gnn, x_fused, labels=labels, return_aux=True
        )

        outputs = {
            "node_logits":   node_logits,
            "seq_logits":    aux.get("seq_logits"),
            "gnn_logits":    aux.get("gnn_logits"),
            "fused_logits":  aux.get("fused_logits"),
            "sample_weight": aux.get("sample_weight"),
            "agreement":     aux.get("agreement"),
        }

        # 4) 图级任务
        if self.use_global:
            outputs["global_logits"] = self.global_head(x_fused, batch)

        return outputs
