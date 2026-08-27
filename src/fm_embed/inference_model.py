"""Repository-local architecture for the archived Flash ExpressionPerformer."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryExpressionEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 100.0, mask_token_id: float = -10.0):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.register_buffer(
            "inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x == self.mask_token_id).nonzero(as_tuple=False)
        frequencies = torch.einsum("bi,j->bij", x, self.inv_freq)
        embedding = torch.cat([frequencies.sin(), frequencies.cos()], dim=-1)
        if len(mask):
            embedding[mask[:, 0], mask[:, 1], :] = 0
        return embedding


class FlashTransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, n_heads: int):
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.hidden_dim, self.n_heads = hidden_dim, n_heads
        self.head_dim = hidden_dim // n_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(x)
        batch, sequence, _ = hidden.shape
        def project(layer):
            return layer(hidden).view(
                batch, sequence, self.n_heads, self.head_dim
            ).transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            project(self.q_proj), project(self.k_proj), project(self.v_proj),
            is_causal=False,
        )
        attention = attention.transpose(1, 2).contiguous().view(
            batch, sequence, self.hidden_dim
        )
        x = x + self.out_proj(attention)
        return x + self.ffn(self.norm2(x))


class ExpressionPerformer(nn.Module):
    def __init__(
        self, num_genes: int, hidden_dim: int, n_heads: int, n_layers: int,
        ffn_dim: int, ree_base: float, mask_token_id: float, feature_type: str,
        compute_type: str, include_species_embedding: bool, num_species: int,
    ):
        super().__init__()
        if str(feature_type).lower() != "flash":
            raise ValueError("Only the archived Flash checkpoint is supported for inference")
        self.include_species_embedding = include_species_embedding
        self.gene_embedding = nn.Embedding(num_genes, hidden_dim)
        self.ree = RotaryExpressionEmbedding(hidden_dim, ree_base, mask_token_id)
        if include_species_embedding:
            self.species_embedding = nn.Embedding(num_species, hidden_dim)
        self.layers = nn.ModuleList([
            FlashTransformerLayer(hidden_dim, ffn_dim, n_heads) for _ in range(n_layers)
        ])
        self.output_map = nn.Linear(hidden_dim, 1)

    def _encode_hidden(self, x: torch.Tensor, species_ids: Optional[torch.Tensor] = None):
        batch, genes = x.shape
        ids = torch.arange(genes, device=x.device)
        hidden = self.gene_embedding(ids).unsqueeze(0) + self.ree(x)
        if self.include_species_embedding:
            if species_ids is None:
                species_ids = torch.zeros(batch, dtype=torch.long, device=x.device)
            hidden = hidden + self.species_embedding(species_ids.long()).unsqueeze(1)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden

    def forward(self, x: torch.Tensor, species_ids: Optional[torch.Tensor] = None):
        return self.output_map(self._encode_hidden(x, species_ids)).squeeze(-1)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, species_ids=None, normalize: bool = False):
        embedding = self._encode_hidden(x, species_ids).mean(dim=1)
        return F.normalize(embedding, dim=-1) if normalize else embedding


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict
