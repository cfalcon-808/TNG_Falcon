# ============================================================
#  Project    : Tigers & Goats
#  Module     : Latent Value Shaping VAE
#  File       : LVS_VAE.py
#  Version    : lvs_vae1.0
#
#  Purpose / Goal:
#    Define the offline Variational Autoencoder used to learn compact
#    latent board-state representations and supervised goat-favorability
#    values for reward shaping.
#
#  Overview:
#    - Input: 25 raw state features
#        * 23 board cells: 0=empty, 1=goat, 2=tiger
#        * goats_eaten_state
#        * phase_state
#    - Encoder maps normalized state features to latent mu/logvar
#    - Decoder reconstructs the normalized state from sampled latent z
#    - Value head predicts final-outcome value from latent mu
#    - Loss combines reconstruction, KL regularization, and value MSE
#
#  Workflow:
#    1) Generate dataset with VAE/DATASET_GENERATOR.ipynb
#    2) Load states and value labels in TRAIN_VAE.py
#    3) Train LVSVAE with lvs_vae_loss(...)
#    4) Freeze encoder + value head for PPO reward shaping
#
#  Quick Use:
#    model = LVSVAE()
#    outputs = model(states)
#    losses = lvs_vae_loss(model, states, value_labels, outputs)
#    losses.total.backward()
#
#  Integration points:
#    - VAE/DATASET_GENERATOR.ipynb
#    - VAE/TRAIN_VAE.py
#    - PPO reward shaping wrapper / callback
# ============================================================
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Default User configs
DEFAULT_INPUT_DIM = 25
DEFAULT_LATENT_DIM = 8
DEFAULT_HIDDEN_DIMS = (64, 64)
DEFAULT_VALUE_HIDDEN_DIM = 32


# Create a dataclass to hold the loss terms, where member is a tensor
@dataclass(frozen=True)
class LVSVAELoss:
    """Container for the combined LVS-VAE loss terms."""

    total: Tensor
    reconstruction: Tensor
    kl: Tensor
    value: Tensor

# This function will define a sequential neural network based on the 
# architecture passed to the function arguments.
def _make_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Build a simple feed-forward MLP body."""
    # create a list called layers where each element is a part of a neural network that is being built
    layers: list[nn.Module] = []
    # set the previous dimension to the input dimension
    prev_dim = input_dim

    # This will unpack the tuple and create a layer that starts from the prev dim
    # and ends with the ith element of the hidden_dim tuple, then append an 
    # activation function object. 
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    # return the sequential nn object, * unpacks the layers into the arguments of the object
    return nn.Sequential(*layers)


# LVSVAE inherits from nn.Module so PyTorch can track layers, parameters, and model state
class LVSVAE(nn.Module):
    """Small VAE with a supervised value head for Tigers and Goats states.

    Forward input can be raw states from the dataset:
        [cell_00 ... cell_22, goats_eaten_state, phase_state]

    The model normalizes those raw inputs internally to keep reconstruction
    loss on a stable 0..1 scale:
        board cells: 0..2 -> 0..1
        goats eaten: 0..6 -> 0..1
        phase:       0..1 -> 0..1
    """

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: tuple[int, ...] = DEFAULT_HIDDEN_DIMS,
        value_hidden_dim: int = DEFAULT_VALUE_HIDDEN_DIM,
    ) -> None:
        super().__init__()

        # Ensure dimensions are legal sizes
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size")

        # Store model dimensions on specific instances for later reference 
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # build the encoder MLP from the input size and hidden layer sizes.
        self.encoder = _make_mlp(input_dim, hidden_dims)
        # The output dim of the encoder will be the last value in hidden dims
        encoder_output_dim = hidden_dims[-1]
        # take encoder features into the latent distribution parameters:
        # mu: mean and logvar: log variance
        self.mu = nn.Linear(encoder_output_dim, latent_dim)
        self.logvar = nn.Linear(encoder_output_dim, latent_dim)

        # Build the decoder MLP by reversing the encoder architecture
        decoder_hidden_dims = tuple(reversed(hidden_dims))
        self.decoder = _make_mlp(latent_dim, decoder_hidden_dims)
        # Maps decoders last size back to the input dimension
        self.reconstruction_head = nn.Linear(decoder_hidden_dims[-1], input_dim)

        # Build the value head that predicts goat-favorability from the latent mean.
        # It outputs one scalar in [0, 1] after sigmoid is applied in predict_value_from_latent().
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

        # normalize state values roughly [0,1]
        state_scale = self._build_state_scale(input_dim)
        # stores the state scale in the model but not as a trainable parameter
        self.register_buffer("state_scale", state_scale)


    @staticmethod
    # builds a list of element wise scaling factors to normalize state info
    def _build_state_scale(input_dim: int) -> Tensor:
        if input_dim == DEFAULT_INPUT_DIM:
            return torch.tensor([2.0] * 23 + [6.0, 1.0], dtype=torch.float32)
        # if a different configuration is passed, scale by 1's
        return torch.ones(input_dim, dtype=torch.float32)

    def normalize_states(self, states: Tensor) -> Tensor:
        """Convert raw int-like state features to float values on [0,1] scale."""
        states = states.float()
        # perform element wise division on the states and clamp sclaing to min 1 for safety 
        return states / self.state_scale.clamp_min(1.0)


    def denormalize_states(self, states: Tensor) -> Tensor:
        """Convert normalized reconstructed states back to raw feature scale."""
        return states * self.state_scale.clamp_min(1.0)

    def encode(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Encode raw states into latent mean and log variance."""
        normalized_states = self.normalize_states(states)
        hidden = self.encoder(normalized_states)
        return self.mu(hidden), self.logvar(hidden)

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample z using the standard VAE reparameterization trick."""
        # convert logvar into standard deviation or sqrt(var)
        std = torch.exp(0.5 * logvar)
        # Creates random noise with same shape as std
        eps = torch.randn_like(std)
        # start at the mean, add random noise scaled by the standard deviation
        
        return mu + eps * std

    # decoder (batch size, hidden_dim[-1]) -> output (batch size, 25)
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent samples into normalized state reconstructions."""
        hidden = self.decoder(z)
        return torch.sigmoid(self.reconstruction_head(hidden))

    # apply sigmoid to the output of the value head
    def predict_value_from_latent(self, mu: Tensor) -> Tensor:
        """Predict goat-favorable value from the latent mean."""
        return torch.sigmoid(self.value_head(mu))


    def forward(self, states: Tensor) -> dict[str, Tensor]:
        """Run VAE reconstruction and value prediction for a state batch."""
        mu, logvar = self.encode(states)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        value = self.predict_value_from_latent(mu)

        # Return everything needed for the loss calculation
        return {
            "reconstruction": reconstruction,
            "value": value,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    
    @torch.no_grad() # Do not track gradients in this function
    # Frozen 
    def predict_value(self, states: Tensor) -> Tensor:
        """Predict value for frozen-model reward shaping/evaluation."""
        mu, _ = self.encode(states)
        return self.predict_value_from_latent(mu)


def lvs_vae_loss(
    model: LVSVAE,
    states: Tensor,
    value_labels: Tensor,
    outputs: dict[str, Tensor], # from forward
    beta: float = 0.001,
    value_weight: float = 1.0,
) -> LVSVAELoss:
    """Compute reconstruction, KL, value, and total LVS-VAE loss.

    `states` should be the raw 25-column state tensor from the dataset.
    `value_labels` should contain labels shaped either (batch,) or (batch, 1).
    """
    # Get the normalized states and ensure that the value labels are floats and shaped
    # (batch size, 1)
    normalized_states = model.normalize_states(states)
    value_labels = value_labels.float().view(-1, 1)

    # Calculate the mean squared error between the original state and the reconstruction
    reconstruction_loss = F.mse_loss(
        outputs["reconstruction"],
        normalized_states,
        reduction="mean",
    )

    # Calculate the mean squared error from the predicted value and the true value label
    value_loss = F.mse_loss(
        outputs["value"],
        value_labels,
        reduction="mean",
    )

    # encoder outputs q(z | s) = Normal(mu, variance). 
    # kl loss compares q(z | s) to a standard normal distribution: p(z) = Normal(0, 1)
    kl_loss = -0.5 * torch.mean(
        1 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()
    )

    # sum all the losses, and scale the kl_loss by beta, and value loss by the value weight
    total_loss = reconstruction_loss + beta * kl_loss + value_weight * value_loss

    return LVSVAELoss(
        total=total_loss,
        reconstruction=reconstruction_loss,
        kl=kl_loss,
        value=value_loss,
    )
