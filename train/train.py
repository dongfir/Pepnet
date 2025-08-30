import math, copy, numpy as np, torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, confusion_matrix)
from tqdm import tqdm

from config import Config as cfg  # 引入配置文件
from models.pepnet_model import PepNet
from calibration import collect_logits_targets, fit_temp_bias, apply_temp_bias, fit_isotonic, apply_isotonic
from thresholds import best_threshold_mcc
from ema import EMA


# ---------------- utils ----------------
def estimate_pos_weight(dataset):
    pos, neg = 0, 0
    for d in dataset:
        y = (d.y > 0.5).long().view(-1)
        pos += int(y.sum()); neg += int((1 - y).sum())
    return float(neg / max(1, pos)) if pos > 0 else 1.0


def compute_loss(outputs, data, pos_weight: float | None, epoch: int,
                 cons_weight: float = 0.05, cons_warmup_epochs: int = 3):
    """
    - 主损失：逐样本 BCE（可选 pos_weight），乘以模型给的 sample_weight 后求均值
    - 轻量一致性正则：MSE(sigmoid(seq_logits), sigmoid(gnn_logits))
      * 前 cons_warmup_epochs 轮对 seq/gnn 分支做 detach，防止早期互拉
    """
    logits = torch.clamp(outputs["node_logits"].squeeze(), -30.0, 30.0)
    targets = data.y.float()

    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    else:
        bce_fn = nn.BCEWithLogitsLoss(reduction="none")

    loss_vec = bce_fn(logits, targets)

    # 样本级降噪权重（mean≈1，由模型里 gate 估计）
    w = outputs.get("sample_weight", None)
    if w is not None:
        loss_vec = loss_vec * w

    loss = loss_vec.mean()

    # —— 轻量模态一致性正则 —— #
    if ("seq_logits" in outputs) and ("gnn_logits" in outputs):
        seq_log = outputs["seq_logits"]
        gnn_log = outputs["gnn_logits"]
        if epoch <= max(0, cons_warmup_epochs):
            seq_log = seq_log.detach()
            gnn_log = gnn_log.detach()
        p_s = torch.sigmoid(torch.clamp(seq_log, -30, 30))
        p_g = torch.sigmoid(torch.clamp(gnn_log, -30, 30))
        loss_cons = nn.functional.mse_loss(p_s, p_g)
        loss = loss + float(cons_weight) * loss_cons

    return loss


def any_param_nonfinite(model):
    for p in model.parameters():
        if p is not None and p.data is not None and not torch.isfinite(p.data).all():
            return True
    return False


# ---------------- train / eval ----------------
def train_one_epoch(model, loader, optimizer, device,
                    scaler: GradScaler | None, grad_clip: float, scheduler: LambdaLR | None,
                    ema_obj: EMA | None, epoch: int, pos_weight: float | None):
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None and scaler.is_enabled()
    cons_weight = float(getattr(cfg, "consistency_weight", 0.025))
    cons_warmup = int(getattr(cfg, "consistency_warmup_epochs", 3))

    for data in tqdm(loader, desc="Training", unit="batch"):
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=use_amp):
            outputs = model(data)
            loss = compute_loss(outputs, data, pos_weight=pos_weight,
                                epoch=epoch, cons_weight=cons_weight,
                                cons_warmup_epochs=cons_warmup)

        if not torch.isfinite(loss):
            if scheduler is not None: scheduler.step()
            if use_amp: scaler.update()
            continue

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema_obj is not None:
            ema_obj.update(model)

        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_mcc_epoch(model, loader, device,
                       use_ema: bool = False, ema_obj: EMA | None = None,
                       use_isotonic: bool = True, use_temp_bias: bool = False,
                       t_min: float = 0.28, t_max: float = 0.55, t_step: float = 0.002,
                       sen_floor=None, spe_floor=None):
    # 临时套用 EMA 权重做验证
    if use_ema and ema_obj is not None:
        ema_obj.apply(model)
    logits, targets = collect_logits_targets(model, loader, device)
    if use_ema and ema_obj is not None:
        ema_obj.restore(model)

    if use_isotonic:
        ir = fit_isotonic(logits, targets)
        probs = apply_isotonic(logits, ir).cpu().numpy().astype(float)
    elif use_temp_bias:
        T, b = fit_temp_bias(logits, targets)
        probs = apply_temp_bias(logits, T, b).cpu().numpy().astype(float)
    else:
        probs = torch.sigmoid(logits).cpu().numpy().astype(float)

    y_true = targets.cpu().numpy().astype(int)
    th, _ = best_threshold_mcc(
        y_true, probs, t_min=t_min, t_max=t_max, t_step=t_step,
        sen_floor=sen_floor, spe_floor=spe_floor
    )
    y_pred = (probs > th).astype(int)

    has_both = len(set(y_true.tolist())) > 1
    auc = roc_auc_score(y_true, probs) if has_both else 0.0
    pr  = average_precision_score(y_true, probs) if has_both else 0.0
    f1  = f1_score(y_true, y_pred) if has_both else 0.0
    acc = float((y_pred == y_true).mean())
    mcc = matthews_corrcoef(y_true, y_pred) if has_both else 0.0
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sen = tp / (tp + fn + 1e-8); spe = tn / (tn + fp + 1e-8)
    return auc, pr, f1, acc, mcc, sen, spe


# ---------------- main ----------------
def main():
    # 数据加载
    dataset = torch.load(cfg.data_path, weights_only=False)
    split = int((1 - getattr(cfg, "val_ratio", 0.1)) * len(dataset))
    train_dataset, val_dataset = dataset[:split], dataset[split:]
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=cfg.batch_size)

    device = torch.device(cfg.device)
    model = PepNet(
        cfg.in_dim, cfg.hidden_dim, cfg.num_layers,
        cfg.use_global, cfg.use_mask, cfg.dropout,
        head_type=getattr(cfg, "head_type", "cosine"),
    ).to(device)

    # 超参数注入（从配置文件获取）
    model.node_head.fused_base = float(getattr(cfg, "gate_fused_base", 0.62))
    model.node_head.fused_gain = float(getattr(cfg, "gate_fused_gain", 0.38))
    model.node_head.aux_gain = float(getattr(cfg, "gate_aux_gain", 0.38))
    model.node_head.weight_gamma = float(getattr(cfg, "weight_gamma", 1.30))

    # 优化器与调度器设置
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(getattr(cfg, "warmup_ratio", 0.1) * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = LambdaLR(optimizer, lr_lambda)

    use_amp = getattr(cfg, "use_amp", True) and torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    use_ema = bool(getattr(cfg, "use_ema", True))
    ema_decay = float(getattr(cfg, "ema_decay", 0.999))
    ema_obj = EMA(model, decay=ema_decay) if use_ema else None

    # 训练与验证
    best_mcc, best_state = -1.0, None
    patience = int(getattr(cfg, "early_stop_patience", 10))
    since_best = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            scaler=scaler, grad_clip=getattr(cfg, "grad_clip", 0.5),
            scheduler=scheduler, ema_obj=ema_obj, epoch=epoch, pos_weight=None
        )

        # 验证
        auc, pr, f1, acc, mcc, sen, spe = evaluate_mcc_epoch(
            model, val_loader, device,
            use_ema=use_ema, ema_obj=ema_obj,
            use_isotonic=True, use_temp_bias=False,
            t_min=0.28, t_max=0.55, t_step=0.002
        )

        # 保存最优模型
        improved = mcc > best_mcc
        if improved:
            best_mcc = mcc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, cfg.model_save_path)
            if use_ema and ema_obj is not None:
                ema_obj.apply(model)
                ema_path = cfg.model_save_path.replace(".pt", "_ema.pt")
                torch.save(model.state_dict(), ema_path)
                ema_obj.restore(model)
            since_best = 0
        else:
            since_best += 1

        # 输出结果
        print(f"[Epoch {epoch:03d}] Loss: {train_loss:.4f} | ACC: {acc:.4f} | AUC: {auc:.4f} | PR-AUC: {pr:.4f} | "
              f"F1: {f1:.4f} | MCC: {mcc:.4f} | SEN: {sen:.4f} | SPE: {spe:.4f}")

        if since_best >= patience:
            print(f"[EarlyStop] No improvement for {patience} epochs. Stop.")
            break


if __name__ == "__main__":
    main()
