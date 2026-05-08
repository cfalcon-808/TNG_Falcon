# ============================================================
#  Project    : Tigers & Goats
#  Module     : Latent Value Shaping VAE Trainer
#  File       : TRAIN_VAE.py
#  Version    : lvs_vae_train1.0
#
#  Purpose / Goal:
#    Train the offline LVS-VAE on generated state/value data so the frozen
#    encoder + value head can later be used for PPO reward shaping.
#
#  Overview:
#    - Loads VAE_DATA/lvs_vae/full_20k_40_40_20/states_labels.npz by default
#    - Splits states/value labels into train and validation sets
#    - Trains LVSVAE with reconstruction, KL, and value prediction losses
#    - Saves best/final checkpoints, config, and per-epoch metrics
#
#  Quick Use:
#    Full baseline: python VAE/TRAIN_FULL_LVS_VAE.py
#    Endgame VAE : python VAE/TRAIN_END_LVS_VAE.py
#    Direct edit : edit USER CONFIGS below, then run python VAE/TRAIN_VAE.py
#
#  Integration points:
#    - VAE/DATASET_GENERATOR.ipynb
#    - VAE/LVS_VAE.py
#    - PPO reward shaping wrapper / callback
# ============================================================
from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from LVS_VAE import (
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_INPUT_DIM,
    DEFAULT_LATENT_DIM,
    DEFAULT_VALUE_HIDDEN_DIM,
    LVSVAE,
    LVSVAELoss,
    lvs_vae_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "VAE_DATA" / "full_20k_30_30_40" / "states_labels.npz"
DEFAULT_ENDGAME_DATA_PATH = PROJECT_ROOT / "VAE_DATA" / "end06_20k_40_40_20" / "states_labels.npz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lvs_vae"


# ============================================================
#  USER CONFIGS
#  Edit these values directly before running this script.
# ============================================================

DATA_PATH = DEFAULT_ENDGAME_DATA_PATH
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
RUN_NAME = "end06_20k_lvsvae_v1"  # None creates run_YYYYMMDD_HHMMSS

SEED = 42
DEVICE = "auto"  # "auto", "cpu", or "cuda"

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VALIDATION_RATIO = 0.2

BETA = 0.001
VALUE_WEIGHT = 1.0
GRAD_CLIP_NORM = 5.0

INPUT_DIM = DEFAULT_INPUT_DIM
LATENT_DIM = DEFAULT_LATENT_DIM
HIDDEN_DIMS = DEFAULT_HIDDEN_DIMS
VALUE_HIDDEN_DIM = DEFAULT_VALUE_HIDDEN_DIM

# Set to zero for stability, no need parallelization since dataset is relatively small
NUM_WORKERS = 0

# Saves loss curve PNGs at the end of training when matplotlib is installed.
PLOT_LOSS_CURVES = True

# Print each epoch as a vertical metric block instead of one long line.
PRINT_VERTICAL_METRICS = True


@dataclass(frozen=True)
class TrainConfig:
    run_name: str
    data_path: str
    output_dir: str
    seed: int
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_ratio: float
    beta: float
    value_weight: float
    grad_clip_norm: float
    input_dim: int
    latent_dim: int
    hidden_dims: tuple[int, ...]
    value_hidden_dim: int
    num_workers: int
    plot_loss_curves: bool
    print_vertical_metrics: bool


def artifact_path(output_dir: Path, run_name: str, artifact_name: str) -> Path:
    """Build a run-name-prefixed path for saved training artifacts."""
    return output_dir / f"{run_name}_{artifact_name}"


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS."""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_device(device_arg: str) -> torch.device:
    """Map device config value to a concrete torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable dataset splits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_config(config: TrainConfig) -> None:
    """Fail early when top-of-file training configs are invalid."""
    if config.epochs <= 0:
        raise ValueError(f"EPOCHS must be positive, got {config.epochs}")
    if config.batch_size <= 0:
        raise ValueError(f"BATCH_SIZE must be positive, got {config.batch_size}")
    if config.learning_rate <= 0:
        raise ValueError(f"LEARNING_RATE must be positive, got {config.learning_rate}")
    if config.input_dim <= 0:
        raise ValueError(f"INPUT_DIM must be positive, got {config.input_dim}")
    if config.latent_dim <= 0:
        raise ValueError(f"LATENT_DIM must be positive, got {config.latent_dim}")
    if not config.hidden_dims:
        raise ValueError("HIDDEN_DIMS must contain at least one layer size")
    if any(dim <= 0 for dim in config.hidden_dims):
        raise ValueError(f"HIDDEN_DIMS must all be positive, got {config.hidden_dims}")
    if config.value_hidden_dim <= 0:
        raise ValueError(
            f"VALUE_HIDDEN_DIM must be positive, got {config.value_hidden_dim}"
        )
    if config.num_workers < 0:
        raise ValueError(f"NUM_WORKERS cannot be negative, got {config.num_workers}")

# load the dataset and save them as tensors
def load_dataset(data_path: Path, input_dim: int) -> tuple[Tensor, Tensor, list[str]]:
    """Load compact state/value arrays saved by DATASET_GENERATOR.ipynb."""
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            "Run VAE/DATASET_GENERATOR.ipynb through the Save Dataset section first."
        )

    with np.load(data_path, allow_pickle=False) as data:
        states = data["states"] # used for training VAE
        labels = data["labels"] # used for training value head
        state_columns = (
            [str(col) for col in data["state_columns"]]
            if "state_columns" in data.files
            else []
        )

    # verify data is compatible with current VAE architecture
    if states.ndim != 2:
        raise ValueError(f"states must be 2D, got shape {states.shape}")
    if states.shape[1] != input_dim:
        raise ValueError(f"expected {input_dim} state columns, got {states.shape[1]}")
    if len(states) != len(labels):
        raise ValueError(f"states/labels row mismatch: {len(states)} vs {len(labels)}")

    # Convert the datasets to tensors
    states_tensor = torch.as_tensor(states, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.float32)
    return states_tensor, labels_tensor, state_columns


def split_dataset(
    states: Tensor,
    labels: Tensor,
    validation_ratio: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    """Create deterministic train/validation TensorDatasets."""
    # ensure validation ratio in correct range
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")

    # ensure the row counts data contains at least 2 rows
    row_count = len(states)
    if row_count < 2:
        raise ValueError("dataset must contain at least two rows")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(row_count, generator=generator)

    # Get the number of rows for the validation and training data sets
    validation_count = max(1, int(row_count * validation_ratio))
    train_count = row_count - validation_count
    if train_count <= 0:
        raise ValueError("validation_ratio leaves no rows for training")

    # Create a random permutation list and slice the first {validation_count} elements for the validation set
    # the rest go to the training dataset
    validation_idx = permutation[:validation_count]
    train_idx = permutation[validation_count:]

    train_dataset = TensorDataset(states[train_idx], labels[train_idx])
    validation_dataset = TensorDataset(states[validation_idx], labels[validation_idx])
    return train_dataset, validation_dataset


# accumulates loss terms average per batch later to be averaged by total rows
def update_loss_totals(
    loss_totals: dict[str, float],
    losses: LVSVAELoss,
    batch_size: int,
) -> None:
    """Accumulate loss terms weighted by batch size."""
    loss_totals["total"] += losses.total.item() * batch_size
    loss_totals["reconstruction"] += losses.reconstruction.item() * batch_size
    loss_totals["kl"] += losses.kl.item() * batch_size
    loss_totals["value"] += losses.value.item() * batch_size


    # average out the accumulated losses by the number of rows in the epoch
def average_losses(loss_totals: dict[str, float], row_count: int) -> dict[str, float]:
    """Convert accumulated batch-size-weighted loss totals to means."""
    return {name: total / row_count for name, total in loss_totals.items()}


def run_epoch(
    model: LVSVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    value_weight: float,
    optimizer: torch.optim.Optimizer | None = None, # if provided function trains, if none function evaluates
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    """Run one train or validation epoch and return mean loss terms."""
    # Trian mode when optimizer is provided
    # evaluation mode otherwise
    is_training = optimizer is not None
    model.train(is_training)

    # Initialize the losses and row counts for the epoch
    loss_totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "value": 0.0,
    }
    row_count = 0

    # Iterate over every batch in the loader, and compute the losses.
    # Gradients and weight updates only occur in training mode
    for states, value_labels in loader:
        # move data to whatever device the machine is using
        states = states.to(device)
        value_labels = value_labels.to(device)
        batch_size = len(states)

        # reset the gradient if in training mode
        if is_training:
            optimizer.zero_grad()
        
        # Calling the model runs the forward pass and returns everything needed
        # for the loss calculation
        outputs = model(states)
        # Calculate the losses after the forward pass
        losses = lvs_vae_loss(
            model=model,
            states=states,
            value_labels=value_labels,
            outputs=outputs,
            beta=beta,
            value_weight=value_weight,
        )

        # Back propogate and update model weights only in the training mode
        if is_training:
            # Compute gradients for trainable parameters from the total loss
            losses.total.backward()
            # Clip gradients to prevent unusually large updates
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            # Update the model weights
            optimizer.step()

        # Accumulate batch-size-weighted losses and count processed rows
        update_loss_totals(loss_totals, losses, batch_size)
        row_count += batch_size

    # return the mean loss terms across the whole epoch
    return average_losses(loss_totals, row_count)


# Validate will run the epoch without gradients
@torch.no_grad()
def validate(
    model: LVSVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    value_weight: float,
) -> dict[str, float]:
    """Run validation without gradient tracking."""
    return run_epoch(
        model=model,
        loader=loader,
        device=device,
        beta=beta,
        value_weight=value_weight,
        optimizer=None,
    )


def save_checkpoint(
    path: Path,
    model: LVSVAE,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    epoch: int,
    metrics: dict[str, Any],
    state_columns: list[str],
) -> None:
    """Save a trainable checkpoint with model/optimizer state and metadata."""
    torch.save(
        {
            "epoch": epoch,
            "metrics": metrics,
            "config": asdict(config),
            "state_columns": state_columns,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def write_metrics_csv(path: Path, history: list[dict[str, Any]]) -> None:
    """Write per-epoch metrics to CSV."""
    fieldnames = [
        "run_name",
        "epoch",
        "epoch_seconds",
        "elapsed_seconds",
        "elapsed_time",
        "train_total",
        "train_reconstruction",
        "train_kl",
        "train_value",
        "val_total",
        "val_reconstruction",
        "val_kl",
        "val_value",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_loss_curves(
    output_dir: Path,
    run_name: str,
    history: list[dict[str, Any]],
) -> list[Path]:
    """Save training/validation loss curves as PNG files."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping loss curve plots.")
        return []

    epochs = [row["epoch"] for row in history]
    saved_paths: list[Path] = []

    total_path = artifact_path(output_dir, run_name, "loss_curve_total.png")
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_total"] for row in history], label="train total")
    plt.plot(epochs, [row["val_total"] for row in history], label="validation total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"LVS-VAE Total Loss: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(total_path, dpi=150)
    plt.close()
    saved_paths.append(total_path)

    components_path = artifact_path(output_dir, run_name, "loss_curve_components.png")
    loss_terms = ["reconstruction", "kl", "value"]
    fig, axes = plt.subplots(len(loss_terms), 1, figsize=(9, 10), sharex=True)
    for axis, loss_term in zip(axes, loss_terms):
        axis.plot(
            epochs,
            [row[f"train_{loss_term}"] for row in history],
            label=f"train {loss_term}",
        )
        axis.plot(
            epochs,
            [row[f"val_{loss_term}"] for row in history],
            label=f"validation {loss_term}",
        )
        axis.set_ylabel("Loss")
        axis.set_title(f"{loss_term.title()} Loss: {run_name}")
        axis.legend()
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(components_path, dpi=150)
    plt.close(fig)
    saved_paths.append(components_path)

    return saved_paths


def print_epoch_metrics(metrics: dict[str, Any], vertical: bool) -> None:
    """Print one epoch's losses in compact or vertical format."""
    if not vertical:
        print(
            f"epoch={metrics['epoch']:03d} "
            f"epoch_time={metrics['epoch_seconds']:.2f}s "
            f"elapsed={metrics['elapsed_time']} "
            f"train_total={metrics['train_total']:.6f} "
            f"train_recon={metrics['train_reconstruction']:.6f} "
            f"train_kl={metrics['train_kl']:.6f} "
            f"train_value={metrics['train_value']:.6f} "
            f"val_total={metrics['val_total']:.6f} "
            f"val_recon={metrics['val_reconstruction']:.6f} "
            f"val_kl={metrics['val_kl']:.6f} "
            f"val_value={metrics['val_value']:.6f}"
        )
        return

    print(
        f"================================================================\n"
        f"epoch={metrics['epoch']:03d}\n"
        f"  time:\n"
        f"    epoch seconds  : {metrics['epoch_seconds']:.2f}\n"
        f"    elapsed        : {metrics['elapsed_time']}\n"
        f"  train:\n"
        f"    total          : {metrics['train_total']:.6f}\n"
        f"    reconstruction : {metrics['train_reconstruction']:.6f}\n"
        f"    kl             : {metrics['train_kl']:.6f}\n"
        f"    value          : {metrics['train_value']:.6f}\n"
        f"  validation:\n"
        f"    total          : {metrics['val_total']:.6f}\n"
        f"    reconstruction : {metrics['val_reconstruction']:.6f}\n"
        f"    kl             : {metrics['val_kl']:.6f}\n"
        f"    value          : {metrics['val_value']:.6f}\n"
        f"================================================================\n"
    )


def main() -> None:
    set_seed(SEED)

    device = resolve_device(DEVICE)
    run_name = RUN_NAME or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = artifact_path(output_dir, run_name, "config.json")
    best_model_path = artifact_path(output_dir, run_name, "best_model.pt")
    final_model_path = artifact_path(output_dir, run_name, "final_model.pt")
    metrics_path = artifact_path(output_dir, run_name, "metrics.csv")

    config = TrainConfig(
        run_name=run_name,
        data_path=str(DATA_PATH),
        output_dir=str(output_dir),
        seed=SEED,
        device=str(device),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        validation_ratio=VALIDATION_RATIO,
        beta=BETA,
        value_weight=VALUE_WEIGHT,
        grad_clip_norm=GRAD_CLIP_NORM,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        value_hidden_dim=VALUE_HIDDEN_DIM,
        num_workers=NUM_WORKERS,
        plot_loss_curves=PLOT_LOSS_CURVES,
        print_vertical_metrics=PRINT_VERTICAL_METRICS,
    )
    validate_config(config)

    states, labels, state_columns = load_dataset(DATA_PATH, config.input_dim)
    train_dataset, validation_dataset = split_dataset(
        states=states,
        labels=labels,
        validation_ratio=VALIDATION_RATIO,
        seed=SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = LVSVAE(
        input_dim=config.input_dim,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        value_hidden_dim=VALUE_HIDDEN_DIM,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    print(f"Training LVS-VAE on {device}")
    print(f"Dataset rows: train={len(train_dataset)} validation={len(validation_dataset)}")
    print(f"Output dir: {output_dir}")

    history: list[dict[str, Any]] = []
    best_validation_total = float("inf")
    training_start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        train_losses = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            beta=BETA,
            value_weight=VALUE_WEIGHT,
            optimizer=optimizer,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
        validation_losses = validate(
            model=model,
            loader=validation_loader,
            device=device,
            beta=BETA,
            value_weight=VALUE_WEIGHT,
        )
        epoch_seconds = time.perf_counter() - epoch_start_time
        elapsed_seconds = time.perf_counter() - training_start_time

        metrics = {
            "run_name": run_name,
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed_seconds,
            "elapsed_time": format_elapsed_time(elapsed_seconds),
            "train_total": train_losses["total"],
            "train_reconstruction": train_losses["reconstruction"],
            "train_kl": train_losses["kl"],
            "train_value": train_losses["value"],
            "val_total": validation_losses["total"],
            "val_reconstruction": validation_losses["reconstruction"],
            "val_kl": validation_losses["kl"],
            "val_value": validation_losses["value"],
        }
        history.append(metrics)

        if validation_losses["total"] < best_validation_total:
            best_validation_total = validation_losses["total"]
            save_checkpoint(
                path=best_model_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=metrics,
                state_columns=state_columns,
            )

        print_epoch_metrics(metrics, PRINT_VERTICAL_METRICS)

    final_metrics = history[-1]
    save_checkpoint(
        path=final_model_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=EPOCHS,
        metrics=final_metrics,
        state_columns=state_columns,
    )
    write_metrics_csv(metrics_path, history)

    loss_curve_paths: list[Path] = []
    if PLOT_LOSS_CURVES:
        loss_curve_paths = write_loss_curves(output_dir, run_name, history)

    print("Training complete")
    print(f"Elapsed time: {format_elapsed_time(time.perf_counter() - training_start_time)}")
    print(f"Best validation total: {best_validation_total:.6f}")
    print(f"Config: {config_path}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Final checkpoint: {final_model_path}")
    print(f"Metrics: {metrics_path}")
    for loss_curve_path in loss_curve_paths:
        print(f"Loss curve: {loss_curve_path}")


if __name__ == "__main__":
    main()
