"""StableAdamW optimiser — AdamW with RMS-based learning-rate scaling.

Forked from the Dinomaly reference implementation which itself is derived from
the official PyTorch AdamW.  The key difference is per-step gradient clipping
via an RMS-based ``clip_threshold`` that stabilises training with large learning
rates.

Reference: https://github.com/guojiajeremy/Dinomaly/blob/master/optimizers/StableAdamW.py
"""

from __future__ import annotations

import math

import torch
from torch.optim.optimizer import Optimizer


class StableAdamW(Optimizer):
    r"""AdamW variant with RMS-based learning-rate scaling.

    Parameters
    ----------
    params:
        Iterable of parameters or parameter-group dicts.
    lr:
        Learning rate (default: 1e-3).
    betas:
        Coefficients for running averages of gradient/squared-gradient
        (default: (0.9, 0.999)).
    eps:
        Numerical-stability term (default: 1e-8).
    weight_decay:
        Decoupled weight-decay coefficient (default: 1e-2).
    amsgrad:
        Whether to use the AMSGrad variant (default: False).
    clip_threshold:
        RMS clip threshold for the learning-rate scale (default: 1.0).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        clip_threshold: float = 1.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            clip_threshold=clip_threshold,
        )
        super().__init__(params, defaults)

    def _rms(self, tensor: torch.Tensor) -> float:
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Decoupled weight decay
                p.data.mul_(1.0 - group["lr"] * group["weight_decay"])

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("StableAdamW does not support sparse gradients.")

                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialisation
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                # Decay running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                # RMS-based learning-rate scaling
                lr_scale = grad / denom
                lr_scale = max(1.0, self._rms(lr_scale) / group["clip_threshold"])

                step_size = group["lr"] / bias_correction1 / lr_scale
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
