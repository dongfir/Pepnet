import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineMarginHead(nn.Module):
    """
    二分类 CosFace/ArcFace 头：
      训练：对真类通道加 margin（余弦空间）
      推理：不加 margin（仅缩放 s）
    注意：forward(labels=None) 用于推理；训练时需传 labels。
    返回形状：[N, 2]（两类 logit），你可以用 logit_pos - logit_neg 得到二分类 logit。
    """
    def __init__(self, feat_dim=256, s=30.0, m=0.20, mode="cosface"):
        super().__init__()
        self.s = s
        self.m = m
        self.mode = mode  # "cosface" or "arcface"
        self.weight = nn.Parameter(torch.empty(2, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels=None):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)          # [2, D]
        cos = torch.matmul(x, w.t())                   # [N, 2]

        if labels is None:
            return cos * self.s

        one_hot = F.one_hot(labels.long(), num_classes=2).float()
        if self.mode == "cosface":
            cos_m = cos - one_hot * self.m
        else:
            theta = torch.acos(cos.clamp(-1+1e-7, 1-1e-7))
            theta_m = theta + one_hot * self.m
            cos_m = torch.cos(theta_m)
        return cos_m * self.s

def two_class_to_binary_logit(two_logits: torch.Tensor) -> torch.Tensor:
    """把 [N,2] 转成二分类 logit：logit_pos - logit_neg"""
    return (two_logits[:, 1] - two_logits[:, 0]).contiguous()
