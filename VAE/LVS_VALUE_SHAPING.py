# ============================================================
#  Project    : Tigers & Goats
#  Module     : Frozen LVS-VAE Reward Shaping Helper
#  File       : LVS_VALUE_SHAPING.py
#
#  Purpose / Goal:
#    Load trained LVS-VAE checkpoints and expose a reward-function wrapper
#    that adds optional goat-favorability value-delta shaping.
# ============================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

try:
    from .LVS_VAE import LVSVAE
    from .LVS_VAE_PS import LVSVAEPS
except ImportError:  # Allows direct execution when VAE/ is on sys.path.
    from LVS_VAE import LVSVAE
    from LVS_VAE_PS import LVSVAEPS


RewardFn = Callable[[np.ndarray, int, np.ndarray, bool, bool, dict], float]


@dataclass(frozen=True)
class LVSValueShapingConfig:
    """Configuration for frozen phase-gated LVS-VAE reward shaping."""

    full_checkpoint_paths: tuple[str | Path, ...]
    end_checkpoint_paths: tuple[str | Path, ...]
    placing_survival_checkpoint_paths: tuple[str | Path, ...] = ()
    shaping_coef: float = 0.02
    placing_survival_shaping_coef: float = 0.0
    placing_survival_positive_delta_only: bool = True
    progress_threshold: float = 0.7
    endgame_weight_after_threshold: float = 0.7
    progress_mode: str = "turn_max_ratio"
    max_turns: int = 100
    device: str = "cpu"


def _torch_load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint across PyTorch versions with a clear path in errors."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_lvs_model(checkpoint_path: str | Path, device: torch.device) -> LVSVAE:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LVS-VAE checkpoint not found: {path}")

    checkpoint = _torch_load_checkpoint(path, device)
    config = checkpoint.get("config", {})

    model = LVSVAE(
        input_dim=int(config.get("input_dim", 25)),
        latent_dim=int(config.get("latent_dim", 8)),
        hidden_dims=tuple(config.get("hidden_dims", (64, 64))),
        value_hidden_dim=int(config.get("value_hidden_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def _load_lvs_ps_model(checkpoint_path: str | Path, device: torch.device) -> LVSVAEPS:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"LVS-VAE-PS checkpoint not found: {path}")

    checkpoint = _torch_load_checkpoint(path, device)
    config = checkpoint.get("config", {})

    model = LVSVAEPS(
        input_dim=int(config.get("input_dim", 25)),
        latent_dim=int(config.get("latent_dim", 8)),
        hidden_dims=tuple(config.get("hidden_dims", (64, 64))),
        value_hidden_dim=int(config.get("value_hidden_dim", 32)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def _mean_model_value(models: Sequence[LVSVAE], state: np.ndarray, device: torch.device) -> float:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    values = [model.predict_value(state_tensor).view(-1)[0] for model in models]
    return float(torch.stack(values).mean().item())


def _mean_model_placing_survival(
    models: Sequence[LVSVAEPS],
    state: np.ndarray,
    device: torch.device,
) -> float:
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).view(1, -1)
    values = [model.predict_placing_survival(state_tensor).view(-1)[0] for model in models]
    return float(torch.stack(values).mean().item())


def _clip_ratio(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def _turn_max_progress(info: dict, max_turns: int) -> tuple[float, float]:
    turn_counter = int(info.get("turn_counter", 0))
    denominator = max(int(max_turns), 1)
    progress_after = _clip_ratio(turn_counter / denominator)
    progress_before = _clip_ratio(max(turn_counter - 1, 0) / denominator)
    return progress_before, progress_after


class PhaseGatedLVSValuePredictor:
    """Lazy CPU inference object used inside each env worker process."""

    def __init__(self, config: LVSValueShapingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._full_models: list[LVSVAE] | None = None
        self._end_models: list[LVSVAE] | None = None
        self._placing_survival_models: list[LVSVAEPS] | None = None

    def _ensure_loaded(self) -> None:
        if (
            self._full_models is not None
            and self._end_models is not None
            and self._placing_survival_models is not None
        ):
            return

        self._full_models = [
            _load_lvs_model(path, self.device)
            for path in self.config.full_checkpoint_paths
        ]
        self._end_models = [
            _load_lvs_model(path, self.device)
            for path in self.config.end_checkpoint_paths
        ]
        self._placing_survival_models = [
            _load_lvs_ps_model(path, self.device)
            for path in self.config.placing_survival_checkpoint_paths
        ]

        if not self._full_models:
            raise ValueError("At least one full-game LVS-VAE checkpoint is required.")
        if not self._end_models:
            raise ValueError("At least one endgame LVS-VAE checkpoint is required.")

    @torch.no_grad()
    def predict(self, state: np.ndarray, progress_ratio: float) -> float:
        self._ensure_loaded()
        assert self._full_models is not None
        assert self._end_models is not None

        full_value = _mean_model_value(self._full_models, state, self.device)
        if progress_ratio < self.config.progress_threshold:
            return full_value

        end_value = _mean_model_value(self._end_models, state, self.device)
        end_weight = float(self.config.endgame_weight_after_threshold)
        return (1.0 - end_weight) * full_value + end_weight * end_value

    @torch.no_grad()
    def predict_placing_survival(self, state: np.ndarray) -> float:
        self._ensure_loaded()
        assert self._placing_survival_models is not None

        if not self._placing_survival_models:
            raise ValueError(
                "At least one placing-survival LVS-VAE checkpoint is required "
                "when placing survival shaping is enabled."
            )

        return _mean_model_placing_survival(
            self._placing_survival_models,
            state,
            self.device,
        )


def _coerce_config(config: LVSValueShapingConfig | dict) -> LVSValueShapingConfig:
    if isinstance(config, LVSValueShapingConfig):
        return config
    if isinstance(config, dict):
        return LVSValueShapingConfig(**config)
    raise TypeError(f"Expected LVSValueShapingConfig or dict, got {type(config).__name__}")


def make_lvs_vae_value_predictor(
    config: LVSValueShapingConfig | dict,
) -> PhaseGatedLVSValuePredictor:
    """Create a reusable phase-gated LVS-VAE value predictor."""
    return PhaseGatedLVSValuePredictor(_coerce_config(config))


def smoke_check_lvs_vae_value_predictor(
    config: LVSValueShapingConfig | dict,
    state: np.ndarray | None = None,
) -> dict[str, float]:
    """Load the configured ensemble and verify values stay in [0, 1]."""
    shaping_config = _coerce_config(config)
    predictor = PhaseGatedLVSValuePredictor(shaping_config)
    if state is None:
        state = np.array([0] * 23 + [0, 0], dtype=np.float32)

    early_value = predictor.predict(state, 0.0)
    late_value = predictor.predict(state, 1.0)
    result = {
        "early_value": early_value,
        "late_value": late_value,
    }
    for name, value in {
        "early_value": early_value,
        "late_value": late_value,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise AssertionError(f"{name} must be in [0, 1], got {value}")

    if shaping_config.placing_survival_checkpoint_paths:
        placing_survival = predictor.predict_placing_survival(state)
        if not 0.0 <= placing_survival <= 1.0:
            raise AssertionError(
                f"placing_survival must be in [0, 1], got {placing_survival}"
            )
        result["placing_survival"] = placing_survival

    return result


def make_lvs_vae_reward_fn(
    base_reward_fn: RewardFn,
    config: LVSValueShapingConfig | dict,
) -> RewardFn:
    """Return an env-compatible reward function with LVS-VAE delta shaping."""
    shaping_config = _coerce_config(config)
    value_model = PhaseGatedLVSValuePredictor(shaping_config)

    def reward_fn(prev_obs, action, obs, terminated, truncated, info):
        sparse_reward = float(
            base_reward_fn(prev_obs, action, obs, terminated, truncated, info)
        )

        if shaping_config.progress_mode != "turn_max_ratio":
            raise ValueError(
                "Unsupported LVS-VAE progress_mode "
                f"{shaping_config.progress_mode!r}; expected 'turn_max_ratio'."
            )

        progress_before, progress_after = _turn_max_progress(
            info,
            shaping_config.max_turns,
        )
        value_before = value_model.predict(np.asarray(prev_obs), progress_before)
        value_after = value_model.predict(np.asarray(obs), progress_after)
        value_delta = value_after - value_before
        vae_component = float(shaping_config.shaping_coef) * value_delta
        placing_survival_component = 0.0
        placing_survival_before = None
        placing_survival_after = None
        placing_survival_delta = None

        placing_survival_enabled = (
            bool(shaping_config.placing_survival_checkpoint_paths)
            and float(shaping_config.placing_survival_shaping_coef) != 0.0
        )
        if placing_survival_enabled and int(np.asarray(prev_obs)[24]) == 0:
            placing_survival_before = value_model.predict_placing_survival(
                np.asarray(prev_obs)
            )
            placing_survival_after = value_model.predict_placing_survival(
                np.asarray(obs)
            )
            placing_survival_delta = placing_survival_after - placing_survival_before
            if shaping_config.placing_survival_positive_delta_only:
                placing_survival_delta = max(placing_survival_delta, 0.0)
            placing_survival_component = (
                float(shaping_config.placing_survival_shaping_coef)
                * placing_survival_delta
            )

        total_reward = sparse_reward + vae_component + placing_survival_component

        info["reward_sparse_component"] = sparse_reward
        info["reward_vae_component"] = vae_component
        info["reward_placing_survival_component"] = placing_survival_component
        info["reward_total"] = total_reward
        info["lvs_value_before"] = value_before
        info["lvs_value_after"] = value_after
        info["lvs_value_delta"] = value_delta
        info["lvs_progress_ratio"] = progress_after
        info["lvs_gate_active"] = float(progress_after >= shaping_config.progress_threshold)
        if placing_survival_before is not None:
            info["lvs_placing_survival_before"] = placing_survival_before
            info["lvs_placing_survival_after"] = placing_survival_after
            info["lvs_placing_survival_delta"] = placing_survival_delta

        return total_reward

    return reward_fn
