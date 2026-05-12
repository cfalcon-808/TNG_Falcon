# ============================================================
#  Project    : Tigers & Goats
#  Module     : Placing-Survival LVS-VAE Trainer
#  File       : TRAIN_PLACING_SURVIVAL_LVS_VAE.py
#
#  Purpose / Goal:
#    Train the separate placing-survival VAE from LVS_VAE_PS.py.
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
)
from LVS_VAE_PS import LVSVAEPS, LVSVAEPSLoss, lvs_vae_ps_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "VAE_DATA" / "full_40k_40_40_20" / "states_labels.npz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "lvs_vae_ps"

DATA_PATH = DEFAULT_DATA_PATH
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
RUN_NAME = "full_40k_ps_only_lvsvae_v1"

SEED = 42
DEVICE = "auto"
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
VALIDATION_RATIO = 0.2
BETA = 0.001
PLACING_SURVIVAL_WEIGHT = 1.0
GRAD_CLIP_NORM = 5.0

INPUT_DIM = DEFAULT_INPUT_DIM
LATENT_DIM = DEFAULT_LATENT_DIM
HIDDEN_DIMS = DEFAULT_HIDDEN_DIMS
VALUE_HIDDEN_DIM = DEFAULT_VALUE_HIDDEN_DIM
NUM_WORKERS = 0
PLOT_LOSS_CURVES = True
PRINT_VERTICAL_METRICS = True
FILTER_PLACING_PHASE = True


@dataclass(frozen=True)
class TrainPSConfig:
    run_name: str
    model_kind: str
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
    placing_survival_weight: float
    grad_clip_norm: float
    input_dim: int
    latent_dim: int
    hidden_dims: tuple[int, ...]
    value_hidden_dim: int
    num_workers: int
    filter_placing_phase: bool
    plot_loss_curves: bool
    print_vertical_metrics: bool


def artifact_path(output_dir: Path, run_name: str, artifact_name: str) -> Path:
    return output_dir / f"{run_name}_{artifact_name}"


def format_elapsed_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(data_path: Path, input_dim: int) -> tuple[Tensor, Tensor, list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        states = data["states"]
        labels = data["labels"]
        state_columns = (
            [str(col) for col in data["state_columns"]]
            if "state_columns" in data.files
            else []
        )
    if states.shape[1] != input_dim:
        raise ValueError(f"expected {input_dim} state columns, got {states.shape[1]}")
    return (
        torch.as_tensor(states, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
        state_columns,
    )


def split_dataset(
    states: Tensor,
    labels: Tensor,
    validation_ratio: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(f"validation_ratio must be in (0, 1), got {validation_ratio}")
    row_count = len(states)
    if row_count < 2:
        raise ValueError("dataset must contain at least two rows")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(row_count, generator=generator)
    validation_count = max(1, int(row_count * validation_ratio))
    validation_idx = permutation[:validation_count]
    train_idx = permutation[validation_count:]
    return TensorDataset(states[train_idx], labels[train_idx]), TensorDataset(
        states[validation_idx],
        labels[validation_idx],
    )


def update_loss_totals(
    loss_totals: dict[str, float],
    losses: LVSVAEPSLoss,
    batch_size: int,
) -> None:
    loss_totals["total"] += losses.total.item() * batch_size
    loss_totals["reconstruction"] += losses.reconstruction.item() * batch_size
    loss_totals["kl"] += losses.kl.item() * batch_size
    loss_totals["placing_survival"] += losses.placing_survival.item() * batch_size


def average_losses(loss_totals: dict[str, float], row_count: int) -> dict[str, float]:
    return {name: total / row_count for name, total in loss_totals.items()}


def run_epoch(
    model: LVSVAEPS,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    placing_survival_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float = 0.0,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)
    loss_totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "placing_survival": 0.0,
    }
    row_count = 0
    for states, value_labels in loader:
        states = states.to(device)
        value_labels = value_labels.to(device)
        batch_size = len(states)
        if is_training:
            optimizer.zero_grad()
        outputs = model(states)
        losses = lvs_vae_ps_loss(
            model=model,
            states=states,
            value_labels=value_labels,
            outputs=outputs,
            beta=beta,
            placing_survival_weight=placing_survival_weight,
        )
        if is_training:
            losses.total.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        update_loss_totals(loss_totals, losses, batch_size)
        row_count += batch_size
    return average_losses(loss_totals, row_count)


@torch.no_grad()
def validate(
    model: LVSVAEPS,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    placing_survival_weight: float,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        loader=loader,
        device=device,
        beta=beta,
        placing_survival_weight=placing_survival_weight,
        optimizer=None,
    )


def save_checkpoint(
    path: Path,
    model: LVSVAEPS,
    optimizer: torch.optim.Optimizer,
    config: TrainPSConfig,
    epoch: int,
    metrics: dict[str, Any],
    state_columns: list[str],
) -> None:
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
    fieldnames = [
        "run_name",
        "epoch",
        "epoch_seconds",
        "elapsed_seconds",
        "elapsed_time",
        "train_total",
        "train_reconstruction",
        "train_kl",
        "train_placing_survival",
        "val_total",
        "val_reconstruction",
        "val_kl",
        "val_placing_survival",
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
    plt.title(f"LVS-VAE-PS Total Loss: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(total_path, dpi=150)
    plt.close()
    saved_paths.append(total_path)

    components_path = artifact_path(output_dir, run_name, "loss_curve_components.png")
    loss_terms = ["reconstruction", "kl", "placing_survival"]
    fig, axes = plt.subplots(len(loss_terms), 1, figsize=(9, 8), sharex=True)
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
    if not vertical:
        print(
            f"epoch={metrics['epoch']:03d} "
            f"epoch_time={metrics['epoch_seconds']:.2f}s "
            f"elapsed={metrics['elapsed_time']} "
            f"train_total={metrics['train_total']:.6f} "
            f"train_recon={metrics['train_reconstruction']:.6f} "
            f"train_kl={metrics['train_kl']:.6f} "
            f"train_survival={metrics['train_placing_survival']:.6f} "
            f"val_total={metrics['val_total']:.6f} "
            f"val_recon={metrics['val_reconstruction']:.6f} "
            f"val_kl={metrics['val_kl']:.6f} "
            f"val_survival={metrics['val_placing_survival']:.6f}"
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
        f"    survival       : {metrics['train_placing_survival']:.6f}\n"
        f"  validation:\n"
        f"    total          : {metrics['val_total']:.6f}\n"
        f"    reconstruction : {metrics['val_reconstruction']:.6f}\n"
        f"    kl             : {metrics['val_kl']:.6f}\n"
        f"    survival       : {metrics['val_placing_survival']:.6f}\n"
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

    config = TrainPSConfig(
        run_name=run_name,
        model_kind="lvs_vae_ps",
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
        placing_survival_weight=PLACING_SURVIVAL_WEIGHT,
        grad_clip_norm=GRAD_CLIP_NORM,
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        hidden_dims=HIDDEN_DIMS,
        value_hidden_dim=VALUE_HIDDEN_DIM,
        num_workers=NUM_WORKERS,
        filter_placing_phase=FILTER_PLACING_PHASE,
        plot_loss_curves=PLOT_LOSS_CURVES,
        print_vertical_metrics=PRINT_VERTICAL_METRICS,
    )

    states, labels, state_columns = load_dataset(DATA_PATH, INPUT_DIM)
    if FILTER_PLACING_PHASE:
        placing_mask = states[:, 24] == 0
        states = states[placing_mask]
        labels = labels[placing_mask]
        if len(states) == 0:
            raise ValueError("No placing-phase rows found in dataset.")

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

    model = LVSVAEPS(
        input_dim=INPUT_DIM,
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

    print(f"Training LVS-VAE-PS on {device}")
    print(f"Dataset rows: train={len(train_dataset)} validation={len(validation_dataset)}")
    print(f"Output dir: {output_dir}")

    history: list[dict[str, Any]] = []
    best_validation_survival = float("inf")
    training_start_time = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        train_losses = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            beta=BETA,
            placing_survival_weight=PLACING_SURVIVAL_WEIGHT,
            optimizer=optimizer,
            grad_clip_norm=GRAD_CLIP_NORM,
        )
        validation_losses = validate(
            model=model,
            loader=validation_loader,
            device=device,
            beta=BETA,
            placing_survival_weight=PLACING_SURVIVAL_WEIGHT,
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
            "train_placing_survival": train_losses["placing_survival"],
            "val_total": validation_losses["total"],
            "val_reconstruction": validation_losses["reconstruction"],
            "val_kl": validation_losses["kl"],
            "val_placing_survival": validation_losses["placing_survival"],
        }
        history.append(metrics)
        if validation_losses["placing_survival"] < best_validation_survival:
            best_validation_survival = validation_losses["placing_survival"]
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

    save_checkpoint(
        path=final_model_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=EPOCHS,
        metrics=history[-1],
        state_columns=state_columns,
    )
    write_metrics_csv(metrics_path, history)

    loss_curve_paths: list[Path] = []
    if PLOT_LOSS_CURVES:
        loss_curve_paths = write_loss_curves(output_dir, run_name, history)

    print("Training complete")
    print(f"Elapsed time: {format_elapsed_time(time.perf_counter() - training_start_time)}")
    print(f"Best validation placing_survival: {best_validation_survival:.6f}")
    print(f"Config: {config_path}")
    print(f"Best checkpoint: {best_model_path}")
    print(f"Final checkpoint: {final_model_path}")
    print(f"Metrics: {metrics_path}")
    for loss_curve_path in loss_curve_paths:
        print(f"Loss curve: {loss_curve_path}")


if __name__ == "__main__":
    main()
