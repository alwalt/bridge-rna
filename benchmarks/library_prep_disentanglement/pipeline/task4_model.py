"""Small scMEDAL-inspired heads for frozen BridgeRNA embeddings."""
from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function


class _GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.strength * grad_output, None


def gradient_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(x, strength)


class Disentangler(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, latent_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        def encoder():
            return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim))
        self.fe = encoder(); self.re = encoder()
        self.decoder = nn.Sequential(nn.Linear(2 * latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))
        self.re_classifier = nn.Linear(latent_dim, 2)
        self.fe_adversary = nn.Linear(latent_dim, 2)

    def forward(self, z: torch.Tensor, grl_strength: float = 1.0) -> dict[str, torch.Tensor]:
        fe, re = self.fe(z), self.re(z)
        return {"fe": fe, "re": re, "reconstructed": self.decoder(torch.cat([fe, re], dim=1)),
                "re_logits": self.re_classifier(re), "fe_logits": self.fe_adversary(gradient_reverse(fe, grl_strength))}


class LinearResidualizer:
    """Remove the training-set difference between Ribo and PolyA centroids."""
    def fit(self, z, y):
        import numpy as np
        direction = np.asarray(z)[np.asarray(y) == 1].mean(0) - np.asarray(z)[np.asarray(y) == 0].mean(0)
        norm = np.linalg.norm(direction)
        if norm == 0: raise ValueError("Zero library direction")
        self.direction_ = direction / norm
        return self

    def transform(self, z):
        import numpy as np
        z = np.asarray(z); return z - (z @ self.direction_)[:, None] * self.direction_[None, :]
