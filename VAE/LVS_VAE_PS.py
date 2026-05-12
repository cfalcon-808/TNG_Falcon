# ============================================================
#  Project    : Tigers & Goats
#  Module     : Placing-Survival LVS-VAE
#  File       : LVS_VAE_PS.py
#
#  Purpose / Goal:
#    Define a separate VAE for opening/placing survival shaping only.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .LVS_VAE import (
        DEFAULT_HIDDEN_DIMS,
        DEFAULT_INPUT_DIM,
        DEFAULT_LATENT_DIM,
        DEFAULT_VALUE_HIDDEN_DIM,
        _make_mlp,
    )
except ImportError:  # Allows direct execution when VAE/ is on sys.path.
    from LVS_VAE import (
        DEFAULT_HIDDEN_DIMS,
        DEFAULT_INPUT_DIM,
        DEFAULT_LATENT_DIM,
        DEFAULT_VALUE_HIDDEN_DIM,
        _make_mlp,
    )


@dataclass(frozen=True)
class LVSVAEPSLoss:
    """Container for placing-survival VAE loss terms."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    placing_survival: Tensor


class LVSVAEPS(nn.Module):
    """VAE with only a placing-survival head."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        value_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = _make_mlp(input_dim, hidden_dims)
        encoder_output_dim = hidden_dims[-1]
        self.mu = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar = nn.Linear(encoder_output_dim, latent_dim)

        decoder_hidden_dims = tuple(reversed(hidden_dims))
        self.decoder = _make_mlp(latent_dim, decoder_hidden_dims)
        self.reconstruction_head = nn.Linear(decoder_hidden_dims[-1], input_dim)

        self.placing_survival_head = nn.Sequential(
            nn.Linear(latent_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

        self.register_buffer("state_scale", self._build_state_scale(input_dim))

    @staticmethod
    def _build_state_scale(input_dim: int) -> Tensor:
        if input_dim == DEFAULT_INPUT_DIM:
            return torch.tensor([2.0] * 23 + [6.0, 1.0], dtype=torch.float32)
        return torch.ones(input_dim, dtype=torch.float32)

    def normalize_states(self, states: Tensor) -> Tensor:
        return states.float() / self.state_scale.clamp_min(1.0)

    def denormalize_states(self, states: Tensor) -> Tensor:
        return states * self.state_scale.clamp_min(1.0)

    def encode(self, states: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(self.normalize_states(states))
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        hidden = self.decoder(z)
        return torch.sigmoid(self.reconstruction_head(hidden))

    def predict_placing_survival_from_latent(self, mu: Tensor) -> Tensor:
        return torch.sigmoid(self.placing_survival_head(mu))

    def forward(self, states: Tensor) -> dict[str, Tensor]:
        mu, logvar = self.encode(states)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decode(z),
            "placing_survival": self.predict_placing_survival_from_latent(mu),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def predict_placing_survival(self, states: Tensor) -> Tensor:
        mu, _ = self.encode(states)
        return self.predict_placing_survival_from_latent(mu)


def lvs_vae_ps_loss(
    model: LVSVAEPS,
    states: Tensor,
    value_labels: Tensor,
    outputs: dict[str, Tensor],
    beta: float = 0.001,
    placing_survival_weight: float = 1.0,
    placing_survival_labels: Tensor | None = None,
    placing_survival_mask: Tensor | None = None,
) -> LVSVAEPSLoss:
    """Compute reconstruction, KL, and masked placing-survival loss."""
    normalized_states = model.normalize_states(states)
    value_labels = value_labels.float().view(-1, 1)

    reconstruction_loss = F.mse_loss(
        outputs["reconstruction"],
        normalized_states,
        reduction="mean",
    )
    if placing_survival_labels is None:
        placing_survival_labels = (value_labels > 0.0).float()
    else:
        placing_survival_labels = placing_survival_labels.float().view(-1, 1)

    if placing_survival_mask is None:
        placing_survival_mask = states[:, 24].view(-1, 1) == 0
    else:
        placing_survival_mask = placing_survival_mask.bool().view(-1, 1)

    if placing_survival_mask.any():
        placing_survival_loss = F.mse_loss(
            outputs["placing_survival"][placing_survival_mask],
            placing_survival_labels[placing_survival_mask],
            reduction="mean",
        )
    else:
        placing_survival_loss = outputs["placing_survival"].sum() * 0.0

    kl_loss = -0.5 * torch.mean(
        1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
    )
    total_loss = (
        reconstruction_loss
        + beta * kl_loss
        + placing_survival_weight * placing_survival_loss
    )

    return LVSVAEPSLoss(
        total=total_loss,
        reconstruction=reconstruction_loss,
        kl=kl_loss,
        placing_survival=placing_survival_loss,
    )
