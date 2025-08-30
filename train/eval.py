import os
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, confusion_matrix
)

from config import Config as cfg  # 引入配置文件
from models.pepnet_model import PepNet
from calibration import (
    collect_logits_targets, fit_temp_bias, apply_temp_bias,
    fit_isotonic, apply_isotonic
)
from thresholds import best_threshold_mcc


@torch.no_grad()
def evaluate_mcc_style(
    model,
    loader,
    device,
    use_isotonic: bool = True,
    use_temp_bias: bool = False,
    t_min: float = 0.28,
    t_max: float = 0.55,
    t_step: float = 0.002,
    sen_floor=None,
    spe_floor=None,
):
    """
    与训练期一致：可选 Isotonic/Temp 校准 + 仅按 MCC 搜阈值。
    只返回 7 个指标（不打印/不保存阈值）。
    """
    model.eval()
    logits, targets = collect_logits_targets(model, loader, device)

    if use_isotonic:
        ir = fit_isotonic(logits, targets)
        probs = apply_isotonic(logits, ir).cpu().numpy().astype(float)
    elif use_temp_bias:
        T, b = fit_temp_bias(logits, targets)
        probs = apply_temp_bias(logits, T, b).cpu().numpy().astype(float)
    else:
        probs = torch.sigmoid(logits).cpu().numpy().astype(float)

    y_true = targets.cpu().numpy().astype(int)

    # 只按 MCC 找阈值（内部可选 SEN/SPE 下限）
    th, _ = best_threshold_mcc(
        y_true, probs,
        t_min=t_min, t_max=t_max, t_step=t_step,
        sen_floor=sen_floor, spe_floor=spe_floor
    )
    y_pred = (probs > th).astype(int)

    # 7 指标
    has_both = len(set(y_true.tolist())) > 1
    auc = roc_auc_score(y_true, probs) if has_both else 0.0
    pr_auc = average_precision_score(y_true, probs) if has_both else 0.0
    f1 = f1_score(y_true, y_pred) if has_both else 0.0
    acc = float((y_pred == y_true).mean())
    mcc = matthews_corrcoef(y_true, y_pred) if has_both else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)

    return {
        "ACC": acc, "AUC": auc, "PR-AUC": pr_auc,
        "F1": f1, "MCC": mcc, "SEN": sen, "SPE": spe
    }


def inject_head_runtime_hparams(model):
    """
    与 train.py 保持一致：把 Config 里的头/门控超参注入到模型实例。
    这些不在 state_dict 里，加载权重后务必调用一次。
    """
    model.node_head.fused_base    = float(getattr(cfg, "gate_fused_base", 0.70))
    model.node_head.fused_gain    = float(getattr(cfg, "gate_fused_gain", 0.30))
    model.node_head.aux_gain      = float(getattr(cfg, "gate_aux_gain",   0.30))
    model.node_head.weight_gamma  = float(getattr(cfg, "weight_gamma",    1.00))
    if hasattr(model.node_head, "fused_head"):
        if hasattr(model.node_head.fused_head, "s"):
            model.node_head.fused_head.s = float(getattr(cfg, "head_s", 30.0))
        if hasattr(model.node_head.fused_head, "m"):
            model.node_head.fused_head.m = float(getattr(cfg, "head_m", 0.20))

    # 可选：打印确认，防止训练/评估不一致
    try:
        s_val = getattr(model.node_head.fused_head, "s", -1.0)
        m_val = getattr(model.node_head.fused_head, "m", -1.0)
    except Exception:
        s_val, m_val = -1.0, -1.0
    print("[HeadCfg] s=%.2f m=%.2f | base=%.2f gain=%.2f aux=%.2f | w_gamma=%.2f" %
          (s_val, m_val, model.node_head.fused_base, model.node_head.fused_gain,
           model.node_head.aux_gain, model.node_head.weight_gamma))


def main():
    device = torch.device(cfg.device)

    # 1) 构建模型（head_type 与训练一致）
    model = PepNet(
        in_channels=cfg.in_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
        use_global_task=cfg.use_global, use_mask_task=cfg.use_mask,
        dropout=cfg.dropout, head_type=getattr(cfg, "head_type", "cosine")
    ).to(device)

    # 2) 优先加载 EMA 权重（若存在）
    ema_path = cfg.model_save_path.replace(".pt", "_ema.pt")
    ckpt_path = ema_path if os.path.exists(ema_path) else cfg.model_save_path
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state["model"])
    print(f"[Load] Loaded checkpoint: {ckpt_path}")

    # 3) 注入与训练一致的头/门控运行时超参
    inject_head_runtime_hparams(model)

    # 4) 划分验证集（与训练一致：尾部 val_ratio）
    dataset = torch.load(cfg.data_path, weights_only=False)
    split = int((1 - getattr(cfg, "val_ratio", 0.1)) * len(dataset))
    val_dataset = dataset[split:]
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    # 5) 评估（保持与训练一致的校准/阈扫配置；只输出 7 指标）
    metrics = evaluate_mcc_style(
        model, val_loader, device,
        use_isotonic=getattr(cfg, "use_isotonic", True),
        use_temp_bias=getattr(cfg, "use_temp_bias", False),
        t_min=getattr(cfg, "th_search_min", 0.28),
        t_max=getattr(cfg, "th_search_max", 0.55),
        t_step=getattr(cfg, "th_search_step", 0.002),
        sen_floor=getattr(cfg, "sen_floor", None),
        spe_floor=getattr(cfg, "spe_floor", None),
    )

    print("\n[Evaluation Result]")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
