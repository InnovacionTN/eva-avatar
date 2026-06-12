"""One-euro filter for jitter reduction on driving motion parameters.

Operates elementwise on torch tensors (any shape), so one instance can smooth
a scalar (yaw) or a whole expression tensor. dt comes from real timestamps.
"""
import math

import torch


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x: torch.Tensor, t: float) -> torch.Tensor:
        if self.x_prev is None:
            self.x_prev = x.clone()
            self.dx_prev = torch.zeros_like(x)
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # derivative estimate, lowpassed
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # adaptive cutoff: faster motion -> higher cutoff (less lag)
        cutoff = self.min_cutoff + self.beta * dx_hat.abs()
        a = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))

        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
