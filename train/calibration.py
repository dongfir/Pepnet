# calibration.py
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression

@torch.no_grad()
def collect_logits_targets(model, loader, device):
    model.eval()
    L, Y = [], []
    for data in loader:
        data = data.to(device)
        logits = model(data)["node_logits"].squeeze()
        L.append(torch.clamp(logits, -30.0, 30.0).detach().float().cpu())
        Y.append(data.y.detach().float().cpu())
    return torch.cat(L), torch.cat(Y)

def fit_temp_bias(logits, targets, max_iter=150):
    device = logits.device
    T = torch.tensor(1.0, device=device, requires_grad=True)
    b = torch.tensor(0.0, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T, b], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        z = (logits - b) / (T.abs() + 1e-6)
        loss = F.binary_cross_entropy_with_logits(z, targets)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        T.data = T.abs()
    return float(T.item()), float(b.item())

@torch.no_grad()
def apply_temp_bias(logits, T, b):
    z = (logits - b) / max(T, 1e-6)
    p = torch.sigmoid(z)
    return torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)

def fit_isotonic(logits, targets):
    """在概率空间上做单调拟合（更强校准）"""
    p = torch.sigmoid(logits).cpu().numpy().astype(float)
    y = targets.cpu().numpy().astype(float)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(p, y)
    return ir

@torch.no_grad()
def apply_isotonic(logits, ir: IsotonicRegression):
    p = torch.sigmoid(logits).cpu().numpy().astype(float)
    p_cal = ir.transform(p)
    return torch.tensor(p_cal, dtype=torch.float32)
