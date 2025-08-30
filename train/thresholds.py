# thresholds.py
import numpy as np
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def best_threshold_mcc(y_true, y_prob, t_min=0.28, t_max=0.55, t_step=0.002,
                       sen_floor=None, spe_floor=None):
    ts = np.arange(t_min, t_max + 1e-12, t_step, dtype=float)
    best_t, best_mcc = 0.5, -1e9
    for t in ts:
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sen = tp / (tp + fn + 1e-8)
        spe = tn / (tn + fp + 1e-8)
        if sen_floor is not None and sen < sen_floor:
            continue
        if spe_floor is not None and spe < spe_floor:
            continue
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc, best_t = mcc, float(t)
    return best_t, best_mcc
