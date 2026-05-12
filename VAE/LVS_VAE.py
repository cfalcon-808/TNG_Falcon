# ============================================================
#  Project    : Tigers & Goats
#  Module     : Latent Value Shaping VAE
#  File       : LVS_VAE.py
#
#  Purpose / Goal:
#    Define the value-shaping VAE only: reconstruction + KL + outcome value.
#    Placing-survival models live in LVS_VAE_PS.py.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_INPUT_DIM = 25
DEFAULT_LATENT_DIM = 8
DEFAULT_HIDDEN_DIMS = (64, 64)
DEFAULT_VALUE_HIDDEN_DIM = 32


@dataclass(frozen=True)
class LVSVAELoss:
    """Container for the value LVS-VAE loss terms."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    value: Tensor


def _make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a simple feed-forward MLP body."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    return nn.Sequential(*layers)


class LVSVAE(nn.Module):
    """Small VAE with a supervised outcome value head."""

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

        self.value_head = nn.Sequential(
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
        """Convert raw int-like state features to float values on [0, 1]."""
        return states.float() / self.state_scale.clamp_min(1.0)

    def denormalize_states(self, states: Tensor) -> Tensor:
        """Convert normalized reconstructed states back to raw feature scale."""
        return states * self.state_scale.clamp_min(1.0)

    def encode(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Encode raw states into latent mean and log variance."""
        hidden = self.encoder(self.normalize_states(states))
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z using the standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent samples into normalized state reconstructions."""
        hidden = self.decoder(z)
        return torch.sigmoid(self.reconstruction_head(hidden))

    def predict_value_from_latent(self, mu: Tensor) -> Tensor:
        """Predict goat-favorable value from the latent mean."""
        return torch.sigmoid(self.value_head(mu))

    def forward(self, states: Tensor) -> dict[str, Tensor]:
        """Run VAE reconstruction and value prediction."""
        mu, logvar = self.encode(states)
        z = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decode(z),
            "value": self.predict_value_from_latent(mu),
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    @torch.no_grad()
    def predict_value(self, states: Tensor) -> Tensor:
        """Predict value for frozen-model reward shaping/evaluation."""
        mu, _ = self.encode(states)
        return self.predict_value_from_latent(mu)


def lvs_vae_loss(
    model: LVSVAE,
    states: Tensor,
    value_labels: Tensor,
    outputs: dict[str, Tensor],
    beta: float = 0.001,
    value_weight: float = 1.0,
) -> LVSVAELoss:
    """Compute reconstruction, KL, value, and total value-VAE loss."""
    normalized_states = model.normalize_states(states)
    value_labels = value_labels.float().view(-1, 1)

    reconstruction_loss = F.mse_loss(
        outputs["reconstruction"],
        normalized_states,
        reduction="mean",
    )
    value_loss = F.mse_loss(
        outputs["value"],
        value_labels,
        reduction="mean",
    )
    kl_loss = -0.5 * torch.mean(
        1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
    )
    total_loss = reconstruction_loss + beta * kl_loss + value_weight * value_loss

    return LVSVAELoss(
        total=total_loss,
        reconstruction=reconstruction_loss,
        kl=kl_loss,
        value=value_loss,
    )
