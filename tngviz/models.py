"""Trained MaskablePPO players from stable_models/models, loaded on first use in a background thread.

A model sees TnGEnv's observation (23 cells 0/1/2, goats captured, phase) and picks a TnGEnv action id
under the legal-move mask, exactly as in training (TnGEnv.get_state / get_action_mask).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .game import GOAT, REPO, TIGER, Move, Position

MODELS_DIR = REPO / "stable_models" / "models"
SIDE_DIRS = {GOAT: "Goats", TIGER: "Tigers"}
PREFIX = "model:"
_LOAD_LOCK = threading.Lock()


def pretty_name(stem: str) -> str:
    """'mppo_RobustGoat030726' -> 'RobustGoat'; 'best_tiger_vs_smart_goat_10M_20260307_225859' ->
    'best tiger vs smart goat 10M'."""
    name = re.sub(r"^mppo_", "", stem)
    name = re.sub(r"_?\d{8}_\d{6}$", "", name)       # _YYYYMMDD_HHMMSS
    name = re.sub(r"(?<=[A-Za-z])\d{6}$", "", name)   # MMDDYY glued to the name
    return name.replace("_", " ").strip() or stem


@dataclass
class ModelInfo:
    side: str
    path: Path

    @property
    def key(self) -> str:
        return PREFIX + self.path.stem

    @property
    def label(self) -> str:
        return pretty_name(self.path.stem)


def discover(models_dir: Path = MODELS_DIR) -> Dict[str, List[ModelInfo]]:
    out: Dict[str, List[ModelInfo]] = {GOAT: [], TIGER: []}
    for side, sub in SIDE_DIRS.items():
        folder = models_dir / sub
        if folder.is_dir():
            out[side] = [ModelInfo(side, p) for p in sorted(folder.glob("*.zip"))]
    return out


def observation(pos: Position) -> np.ndarray:
    """TnGEnv.get_state() for this position."""
    obs = np.zeros(25, dtype=np.int8)
    for p in range(23):
        obs[p] = 2 if (pos.tm >> p) & 1 else 1 if (pos.gm >> p) & 1 else 0
    obs[23] = pos.captured
    obs[24] = 0 if pos.placing else 1
    return obs


def load_policy(path: Path):
    """The MaskableActorCriticPolicy saved in an SB3 zip, on the CPU.

    Not MaskablePPO.load: SB3 hands torch.load a stream inside the zip, which newer torch (2.15) cannot read
    for these checkpoints (saved with torch 2.5). Reading policy.pth into memory works with any version, and
    play needs only the policy, not the optimizer state.
    """
    import io
    import zipfile

    import torch
    import sb3_contrib  # noqa: F401  (registers the maskable policy classes for unpickling)
    from stable_baselines3.common.save_util import json_to_data

    with zipfile.ZipFile(path) as z:
        data = json_to_data(z.read("data").decode())
        state = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    policy = data["policy_class"](data["observation_space"], data["action_space"], lambda _: 0.0,
                                  **data.get("policy_kwargs", {}))
    policy.load_state_dict(state)
    policy.set_training_mode(False)
    return policy


@dataclass
class ModelPlayer:
    info: ModelInfo
    model: object = None
    error: Optional[str] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._load, daemon=True)
            self._thread.start()

    def _load(self) -> None:
        try:
            with _LOAD_LOCK:     # two threads importing torch at once deadlock
                self.model = load_policy(self.info.path)
        except ImportError as exc:
            self.error = (f"{exc.name or 'sb3-contrib'} is not installed; install tngviz/requirements-models.txt "
                          "to use the trained models")
        except Exception as exc:
            self.error = f"could not load {self.info.path.name}: {exc}"

    @property
    def ready(self) -> bool:
        return self.model is not None

    def choose(self, pos: Position, moves: List[Move]) -> Optional[Move]:
        """The model's deterministic move, or None if it proposes an illegal action."""
        mask = np.zeros(int(self.model.action_space.n), dtype=bool)
        by_action = {m.action: m for m in moves}
        mask[list(by_action)] = True
        action, _ = self.model.predict(observation(pos), deterministic=True, action_masks=mask)
        return by_action.get(int(np.asarray(action).item()))
