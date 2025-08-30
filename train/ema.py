# ema.py
import copy
import torch

class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            assert n in self.shadow
            self.shadow[n].mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def apply(self, model):
        self._backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self._backup[n] = p.data.clone()
            p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self._backup[n])
        self._backup = {}
