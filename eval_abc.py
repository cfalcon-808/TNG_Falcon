# ============================================================
#  Project    : Tigers & Goats - Falcon
#  Module     : Unified Evaluation Runner (Single + Sweep)
#  File       : eval_abc.py
#  Version    : eval3.0
#  Last Update: 2026-03-08
#
#  Purpose / Goal:
#    Evaluate trained goat or tiger policies in two modes:
#      1) Single-mode debug + evaluation for one configured matchup
#      2) Sweep-mode permutation matrix across goat-model and tiger-model sets
#
#  Overview:
#    - Uses env_tng_abc.TnGEnv with action masking (MaskablePPO-compatible)
#    - Supports goat learner and tiger learner evaluation
#    - Supports scripted and model-driven opponents:
#        tiger_ai: greedy | smart | model
#        goat_opponent_ai: random | model
#    - Sweep mode runs full goat-model x tiger-model permutations and exports:
#        artifacts/eval_sweeps/sweep_results_<timestamp>.json
#        artifacts/eval_sweeps/sweep_results_<timestamp>.csv
#
#  Quick Start:
#    1) Open USER CONFIG in this file:
#         EVAL_MODE = "single" or "sweep"
#
#    2) For single mode:
#         - Configure SINGLE_CONFIG:
#             model_path, learner_role, tiger_ai/goat_opponent_ai,
#             optional tiger_model_path/goat_model_path
#         - Run:
#             python eval_abc.py
#
#    3) For sweep mode:
#         - Set EVAL_MODE = "sweep"
#         - Fill SWEEP_CONFIG["goat_models"] and SWEEP_CONFIG["tiger_models"]
#         - Run:
#             python eval_abc.py
#
#    4) Check outputs:
#         - Per-run logs in artifacts/logging/eval_debug/
#         - Sweep summary files in artifacts/eval_sweeps/
# ============================================================


import os
import glob
import sys
import json
import csv
from datetime import datetime
from typing import Optional, Dict, List, TextIO, Tuple

import numpy as np
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO

from env_tng_abc import (
    TnGEnv as BaseEnv,
    DIR_CODES,
    TIGER_AI_GREEDY,
    TIGER_AI_SMART,
    TIGER_AI_MODEL,
    GOAT_LEARNER,
    TIGER_LEARNER,
    GOAT_AI_RANDOM,
    GOAT_AI_MODEL,
)


def parse_cli_args():
    """
    Parse command-line args for tiger AI mode and model path.

    Supported patterns:
        python eval_falcon.py
        python eval_falcon.py model.zip
        python eval_falcon.py normal
        python eval_falcon.py normal model.zip
        python eval_falcon.py battle
        python eval_falcon.py battle model.zip

    Returns:
        tiger_ai_mode (str)
        explicit_model_path (str or None)
    """
    tiger_ai_mode = TIGER_AI_GREEDY  # default = greedy tiger
    model_path = None

    def resolve_mode(token: str) -> Optional[str]:
        t = token.lower()
        if t.startswith("norm") or t == "greedy":
            return TIGER_AI_GREEDY
        if t.startswith("batt") or t == "smart":
            return TIGER_AI_SMART
        return None

    args = sys.argv[1:]

    # No args -> greedy + latest model
    if not args:
        return tiger_ai_mode, model_path
    
    # One arg -> either mode or model file
    if len(args) == 1:
        mode = resolve_mode(args[0])
        if mode is not None:
            tiger_ai_mode = mode
        elif args[0].lower().endswith(".zip"):
            # treat as model file, default tiger = greedy
            model_path = args[0]
        else:
            print(f"[WARN] Unknown argument '{args[0]}', defaulting to greedy tiger")
        return tiger_ai_mode, model_path
        
    # Two or more args -> first is mode, second is model path
    mode = resolve_mode(args[0])
    if mode is not None:
        tiger_ai_mode = mode
    else:
        print(f"[WARN] Unknown mode '{args[0]}' defaulting to greedy tiger")
    
    model_path = args[1]
    return tiger_ai_mode, model_path


# ============================================================
#  CLI FALLBACK (single-mode convenience)
# ============================================================
TIGER_AI_MODE, EXPLICIT_MODEL_PATH = parse_cli_args()


# ============================================================
#  GLOBAL CONSTANTS
# ============================================================

# Model / evaluation
MODELS_DIR: str = "artifacts/models/experiment"        # Folder with .zip models
MODEL_EXT: str = ".zip"

# Debug episodes
N_DEBUG_EPISODES: int = 3         # Number of episodes to run, step-by-step
MAX_STEPS_PER_EP: int = 100       # safety cap so we don't hang forever
DETERMINISTIC_DEBUG: bool = True  # deterministic policy for clarity
SEED: int = 42

# Batch evaluation
N_EVAL_GAMES: int = 100           # number of evaluation games
DETERMINISTIC_EVAL: bool = True   # deterministic policy for stable stats

# Reward thresholds (fallback if env doesn't provide winner)
WIN_REWARD_THRESHOLD: float = 9.0
LOSS_REWARD_THRESHOLD: float = -9.5

# Logging
DEBUG_LOG_DIR: str = "artifacts/logging/eval_debug"

# If False: all debug/eval output goes ONLY to log file
# If True : mirror output to both log file AND console
PRINT_DEBUG_TO_CONSOLE: bool = False

# ============================================================
#  USER CONFIG - Mode Selection
# ============================================================

# True  -> run step-by-step debug episodes before batch evaluation (single mode)
# False -> skip debug and run only batch evaluation
DEBUG_MODE: bool = True

# "single": evaluate one learner model against one opponent setup
# "sweep" : evaluate all tiger-model x goat-model permutations
EVAL_MODE: str = "sweep"  # "single" | "sweep"

# SINGLE MODE CONFIG
SINGLE_CONFIG = {
    # learner model to evaluate; None -> EXPLICIT_MODEL_PATH (CLI) -> latest in MODELS_DIR
    "model_path": None,
    # "goat" or "tiger"
    "learner_role": GOAT_LEARNER,
    # used when learner_role == goat
    "tiger_ai": TIGER_AI_MODE,      # CLI mode fallback (greedy/smart/model)
    "tiger_model_path": None,       # required if tiger_ai == "model"
    # used when learner_role == tiger
    "goat_opponent_ai": GOAT_AI_RANDOM,   # "random" | "model"
    "goat_model_path": None,        # required if goat_opponent_ai == "model"
    # run controls
    "run_debug": DEBUG_MODE,
    "n_debug_episodes": N_DEBUG_EPISODES,
    "n_eval_games": N_EVAL_GAMES,
}

# SWEEP MODE CONFIG
SWEEP_CONFIG = {
    "n_eval_games": 100,
    "deterministic": True,
    "save_results": True,
    "results_dir": "artifacts/eval_sweeps",
    "tiger_models": {
        "robust_tiger_030726": r"stable_models\models\Tigers\mppo_RobustTiger030726.zip",
        "best_tiger_vs_normal_goat_10M": r"stable_models\models\Tigers\best_tiger_vs_normal_goat_10M_20260307_220408.zip",
        "best_tiger_vs_smart_goat_10M": r"stable_models\models\Tigers\best_tiger_vs_smart_goat_10M_20260307_225859.zip",
    },
    "goat_models": {
        "normal_goat_030726": r"stable_models\models\Goats\mppo_NormalGoat030726.zip",
        "smart_goat_030726": r"stable_models\models\Goats\mppo_SmartGoat030726.zip",
        "robust_goat_030726": r"stable_models\models\Goats\mppo_RobustGoat030726.zip",
    },
}


# ============================================================
#  LOGGING / TEE
# ============================================================

class Tee:
    """
    Simple tee: mirror all stdout to multiple streams.
    Used to send prints/env.render() to both terminal and a log file.
    """
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def log(*args, **kwargs):
    """Write ONLY to the log file (via sys.stdout)."""
    print(*args, **kwargs)


def log_and_console(*args, **kwargs):
    """Write to both the log AND real console."""
    sys.__stdout__.write(" ".join(map(str, args)) + "\n")
    sys.__stdout__.flush()
    print(*args, **kwargs)


def build_log_path(model_path: Optional[str], mode_tag: str = "single") -> str:
    """
    Build a log file path inside DEBUG_LOG_DIR based on the model filename.

    Examples:
      models/battle_falcon_final_env1.0_mppo_battle1.0_run_3.zip
        -> debug_logs/battle_falcon_final_env1.0_mppo_battle1.0_run_3_debug_eval.txt

      models/battle_falcon_env1.0_mppo_battle1.0_250000_steps.zip
        -> debug_logs/battle_falcon_env1.0_mppo_battle1.0_250000_steps_debug_eval.txt
    """
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

    if model_path is None:
        stub = "latest_model"
    else:
        base = os.path.basename(model_path)
        stub = os.path.splitext(base)[0]

    filename = f"{stub}_{mode_tag}_eval.txt"
    return os.path.join(DEBUG_LOG_DIR, filename)


def setup_logging(log_path: str, log_tag: str = "single") -> Tuple[TextIO, TextIO]:
    """
    Set up logging so that all stdout goes to a log file, and optionally
    to the console as well (controlled by PRINT_DEBUG_TO_CONSOLE).

    Returns:
        original_stdout: original sys.stdout (so we can restore later)
        log_file_handle: opened file handle (caller should close at the end)
    """
    original_stdout: TextIO = sys.stdout
    log_f: TextIO = open(log_path, "w", encoding="utf-8")  # overwrite each run

    # Always write to log file; optionally mirror to console
    streams: List[TextIO] = [log_f]
    if PRINT_DEBUG_TO_CONSOLE:
        streams.append(sys.__stdout__)

    sys.stdout = Tee(*streams)
    print(f"[LOG] Eval log started ({log_tag}): {log_path}")
    return original_stdout, log_f


# ============================================================
#  MODEL LOADING
# ============================================================

def load_latest_model_path(model_dir: str = MODELS_DIR) -> Optional[str]:
    """
    Find the most recent .zip model in model_dir (by creation time).
    """
    os.makedirs(model_dir, exist_ok=True)
    pattern = os.path.join(model_dir, f"*{MODEL_EXT}")
    zips = glob.glob(pattern)
    if not zips:
        return None
    return max(zips, key=os.path.getctime)


# ============================================================
#  ENVIRONMENT FACTORIES
# ============================================================

def _mask_fn(env) -> np.ndarray:
    """
    Mask function used by ActionMasker.

    Delegates to env.get_action_mask(), which is implemented by
    the underlying wrapped environment.
    """
    return env.unwrapped.get_action_mask()


def make_goat_model_predict_fn(model_path: Optional[str]):
    """
    Build a lazy goat-model policy callable for tiger-learner evaluation.
    Signature expected by env: fn(obs, mask) -> flat_action
    """
    if not model_path:
        return None

    model = None

    def _predict(obs, mask):
        nonlocal model
        if model is None:
            model = MaskablePPO.load(model_path)

        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        mask_arr = None
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(1, -1)

        action, _ = model.predict(obs_arr, deterministic=True, action_masks=mask_arr)
        if isinstance(action, np.ndarray):
            return int(action[0])
        return int(action)

    return _predict


def make_tiger_model_predict_fn(model_path: Optional[str]):
    """
    Build a lazy tiger-model policy callable for goat-learner evaluation.
    Signature expected by env: fn(obs, mask) -> flat_action
    """
    if not model_path:
        return None

    model = None

    def _predict(obs, mask):
        nonlocal model
        if model is None:
            model = MaskablePPO.load(model_path)

        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)

        mask_arr = None
        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.ndim == 1:
                mask_arr = mask_arr.reshape(1, -1)

        action, _ = model.predict(obs_arr, deterministic=True, action_masks=mask_arr)
        if isinstance(action, np.ndarray):
            return int(action[0])
        return int(action)

    return _predict


def describe_matchup(
    learner_role: str,
    tiger_ai: str,
    goat_opponent_ai: str,
    goat_model_path: Optional[str],
    tiger_model_path: Optional[str] = None,
) -> str:
    if learner_role == GOAT_LEARNER:
        if tiger_ai == TIGER_AI_MODEL:
            tiger_label = os.path.splitext(os.path.basename(tiger_model_path or "unknown_tiger"))[0]
            return f"GOAT MODEL vs TIGER MODEL ({tiger_label})"
        tiger_label = "SMART" if tiger_ai == TIGER_AI_SMART else "GREEDY"
        return f"GOAT MODEL vs {tiger_label} TIGER"

    if goat_opponent_ai == GOAT_AI_MODEL:
        goat_name = os.path.splitext(os.path.basename(goat_model_path or "unknown_goat"))[0]
        return f"TIGER MODEL vs GOAT MODEL ({goat_name})"

    return "TIGER MODEL vs RANDOM GOAT"


def make_debug_env(
    seed: Optional[int] = SEED,
    learner_role: str = GOAT_LEARNER,
    tiger_ai: str = TIGER_AI_GREEDY,
    goat_opponent_ai: str = GOAT_AI_RANDOM,
    goat_model_path: Optional[str] = None,
    tiger_model_path: Optional[str] = None,
):
    """
    Build the debug environment:

      TnGEnv(configured matchup) -> Monitor -> ActionMasker
    """
    goat_predict_fn = make_goat_model_predict_fn(goat_model_path)
    tiger_predict_fn = make_tiger_model_predict_fn(tiger_model_path)

    env = BaseEnv(
        tiger_ai=tiger_ai,
        learner_role=learner_role,
        goat_opponent_ai=goat_opponent_ai,
        goat_model_predict_fn=goat_predict_fn,
        tiger_model_predict_fn=tiger_predict_fn,
    )
    env = Monitor(env)
    env = ActionMasker(env, _mask_fn)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def make_eval_env(
    learner_role: str = GOAT_LEARNER,
    tiger_ai: str = TIGER_AI_GREEDY,
    goat_opponent_ai: str = GOAT_AI_RANDOM,
    goat_model_path: Optional[str] = None,
    tiger_model_path: Optional[str] = None,
):
    """
    Build the evaluation environment:

      TnGEnv(configured matchup) -> ActionMasker
    """
    goat_predict_fn = make_goat_model_predict_fn(goat_model_path)
    tiger_predict_fn = make_tiger_model_predict_fn(tiger_model_path)
    env = BaseEnv(
        tiger_ai=tiger_ai,
        learner_role=learner_role,
        goat_opponent_ai=goat_opponent_ai,
        goat_model_predict_fn=goat_predict_fn,
        tiger_model_predict_fn=tiger_predict_fn,
    )
    env = ActionMasker(env, _mask_fn)
    return env


# ============================================================
#  ACTION DECODING
# ============================================================

def decode_flat_action(flat_action: int, dir_codes: int = DIR_CODES) -> Tuple[int, int]:
    """
    Decode a flat Discrete action into (position_index, direction_code).

    Args:
        flat_action: The integer action chosen by the policy.
        dir_codes:   Number of direction codes (e.g., 5).

    Returns:
        (pos, dir_code)
    """
    pos = flat_action // dir_codes
    dir_code = flat_action % dir_codes
    return pos, dir_code


# ============================================================
#  DEBUG EPISODE LOOP (STEP-BY-STEP)
# ============================================================

def run_debug_episodes(
    model_path: str,
    n_episodes: int = N_DEBUG_EPISODES,
    deterministic: bool = DETERMINISTIC_DEBUG,
    max_steps_per_ep: int = MAX_STEPS_PER_EP,
    learner_role: str = GOAT_LEARNER,
    tiger_ai: str = TIGER_AI_GREEDY,
    goat_opponent_ai: str = GOAT_AI_RANDOM,
    goat_model_path: Optional[str] = None,
    tiger_model_path: Optional[str] = None,
) -> None:
    """
    Run a small number of evaluation episodes with a loaded model,
    printing/logging each step, including decoded action and env.render().
    """
    matchup_label = describe_matchup(
        learner_role,
        tiger_ai,
        goat_opponent_ai,
        goat_model_path,
        tiger_model_path=tiger_model_path,
    )
    print("\n" + "=" * 60)
    print(f"[DBG] STEP-BY-STEP {matchup_label} DEBUG EPISODES")
    print("=" * 60)

    print(f"[DBG] Loading model: {model_path}")
    model = MaskablePPO.load(model_path)

    env = make_debug_env(
        seed=SEED,
        learner_role=learner_role,
        tiger_ai=tiger_ai,
        goat_opponent_ai=goat_opponent_ai,
        goat_model_path=goat_model_path,
        tiger_model_path=tiger_model_path,
    )

    wins: Dict[str, int] = {
        "Tiger": 0,
        "Goat": 0,
        "MaxTimeout": 0,
        "RepeatTimeout": 0,
        "Unknown": 0,
    }
    rewards_all: List[float] = []

    print(f"[DBG] Starting debug run for {n_episodes} episode(s).")
    print(f"[DBG] Deterministic policy: {deterministic}")
    print(f"[DBG] Max steps per episode: {max_steps_per_ep}")

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        step_idx = 0

        print("\n---------------- EPISODE", ep, "----------------")
        if hasattr(env, "render"):
            env.render()

        while not (done or truncated) and step_idx < max_steps_per_ep:
            # Action mask from wrapped env
            mask = get_action_masks(env)
            if mask is None:
                raise RuntimeError("No action mask available for evaluation.")

            mask = np.asarray(mask, dtype=bool)
            expected_n = env.action_space.n

            assert mask.ndim == 1, f"Mask must be 1D, got {mask.shape}"
            assert mask.size == expected_n, (
                f"Mask size {mask.size} != action_space.n {expected_n}"
            )
            if not mask.any():
                print("[WARN] Mask had no valid actions; forcing action 0 valid.")
                mask[0] = True

            # Policy action
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=mask,
            )

            flat = int(action)
            pos, dir_code = decode_flat_action(flat)
            print(f"Step {step_idx:02d}: flat_action={flat} -> (pos={pos}, dir={dir_code})")

            # Env step
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            step_idx += 1

            if hasattr(env, "render"):
                env.render()

            if done or truncated:
                print(
                    f"Episode ended at step {step_idx}, "
                    f"reward={ep_reward:.3f}, info={info}"
                )

        winner = classify_outcome(ep_reward, info)
        wins[winner] = wins.get(winner, 0) + 1
        rewards_all.append(ep_reward)

        print(
            f"Episode {ep:02d} SUMMARY | "
            f"Steps: {step_idx} | Reward: {ep_reward:.3f} | Winner: {winner}"
        )

    # Overall debug summary
    mean_r = float(np.mean(rewards_all)) if rewards_all else 0.0
    std_r = float(np.std(rewards_all)) if rewards_all else 0.0

    log_and_console(f"\n[DBG] ========= OVERALL {matchup_label} DEBUG SUMMARY =========")
    log_and_console(f"[DBG] Episodes run : {n_episodes}")
    log_and_console(f"[DBG] Avg Reward   : {mean_r:.3f} (+/-{std_r:.3f})")
    log_and_console(
        "[DBG] Wins         : "
        f"Tiger={wins.get('Tiger', 0)}, "
        f"Goat={wins.get('Goat', 0)}, "
        f"MaxTimeout={wins.get('MaxTimeout', 0)}, "
        f"RepeatTimeout={wins.get('RepeatTimeout', 0)}, "
        f"Unknown={wins.get('Unknown', 0)}"
    )
    log_and_console("[DBG] ===============================================\n")


# ============================================================
#  EVALUATION HELPERS
# ============================================================

def classify_outcome(episode_reward: float, info: dict) -> str:
    """
    Given the final episode info and total reward, classify the outcome.

    Priority:
    1. Use info["winner"] if provided.
    2. Fall back to reward-based thresholds.

    Returns:
        "Goat", "Tiger", "MaxTimeout", "RepeatTimeout", or "Unknown"
    """
    winner = info.get("winner", None)
    mapping = {
        "GoatTimeout": "MaxTimeout",
        "StallTimeout": "RepeatTimeout",
    }
    if winner in mapping:
        winner = mapping[winner]

    if winner in ("Goat", "Tiger", "MaxTimeout", "RepeatTimeout"):
        return winner

    # Fallback based on reward if env didn't provide a winner
    role = info.get("learner_role", GOAT_LEARNER)
    if episode_reward >= WIN_REWARD_THRESHOLD:
        return "Goat" if role == GOAT_LEARNER else "Tiger"
    if episode_reward <= LOSS_REWARD_THRESHOLD:
        return "Tiger" if role == GOAT_LEARNER else "Goat"
    return "Unknown"


# ============================================================
#  BATCH EVALUATION LOOP
# ============================================================

def run_batch_evaluation(
    model_path: str,
    n_games: int = N_EVAL_GAMES,
    deterministic: bool = DETERMINISTIC_EVAL,
    learner_role: str = GOAT_LEARNER,
    tiger_ai: str = TIGER_AI_GREEDY,
    goat_opponent_ai: str = GOAT_AI_RANDOM,
    goat_model_path: Optional[str] = None,
    tiger_model_path: Optional[str] = None,
    summary_label: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a MaskablePPO model over n_games episodes.

    Prints:
        - Goat/Tiger/Timeout counts and percentages
        - Average reward
        - Average episode length
        - Condensed single-line summary
    Returns:
        Dict of evaluation metrics for this matchup.
    """
    matchup_label = summary_label or describe_matchup(
        learner_role,
        tiger_ai,
        goat_opponent_ai,
        goat_model_path,
        tiger_model_path=tiger_model_path,
    )

    print("\n" + "=" * 60)
    print(f"[EVAL] {matchup_label} EVALUATION")
    print("=" * 60)

    if not os.path.isfile(model_path):
        print(f"[EVAL] Model file not found: {model_path}")
        return {}

    if learner_role == TIGER_LEARNER and goat_opponent_ai == GOAT_AI_MODEL and not goat_model_path:
        print("[EVAL] goat_model_path is required for tiger-learner vs goat-model evaluation.")
        return {}
    if learner_role == GOAT_LEARNER and tiger_ai == TIGER_AI_MODEL and not tiger_model_path:
        print("[EVAL] tiger_model_path is required for goat-learner vs tiger-model evaluation.")
        return {}

    print(f"[EVAL] Loading model from: {model_path}")
    model = MaskablePPO.load(model_path)

    env = make_eval_env(
        learner_role=learner_role,
        tiger_ai=tiger_ai,
        goat_opponent_ai=goat_opponent_ai,
        goat_model_path=goat_model_path,
        tiger_model_path=tiger_model_path,
    )

    counts = {
        "Goat": 0,
        "Tiger": 0,
        "MaxTimeout": 0,
        "RepeatTimeout": 0,
        "Unknown": 0,
    }

    total_rewards: List[float] = []
    game_lengths: List[int] = []

    print(f"[EVAL] Starting evaluation of {n_games} games...")
    print(f"[EVAL] Deterministic policy: {deterministic}")

    for i in range(n_games):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward: float = 0.0
        steps = 0

        while not (terminated or truncated):
            # Get current action mask from wrapped env
            action_masks = get_action_masks(env)
            if action_masks is None:
                raise RuntimeError("No action mask available for batch evaluation.")

            action, _ = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

        total_rewards.append(episode_reward)
        game_lengths.append(steps)

        outcome = classify_outcome(episode_reward, info)
        counts[outcome] = counts.get(outcome, 0) + 1

    # Summary stats
    def pct(count: int) -> float:
        return (count / n_games * 100.0) if n_games > 0 else 0.0

    log_and_console("\n" + "=" * 40)
    log_and_console(f"  {matchup_label} EVALUATION RESULTS (N={n_games})")
    log_and_console("=" * 40)

    log_and_console(f"Goat Wins     : {counts['Goat']:4d} ({pct(counts['Goat']):5.1f}%)")
    log_and_console(f"Tiger Wins    : {counts['Tiger']:4d} ({pct(counts['Tiger']):5.1f}%)")
    log_and_console(f"Max Timeout   : {counts['MaxTimeout']:4d} ({pct(counts['MaxTimeout']):5.1f}%)")
    log_and_console(f"Repeat Timeout: {counts['RepeatTimeout']:4d} ({pct(counts['RepeatTimeout']):5.1f}%)")
    log_and_console(f"Unknown       : {counts['Unknown']:4d} ({pct(counts['Unknown']):5.1f}%)")

    log_and_console("-" * 40)
    log_and_console(f"Avg Reward:     {np.mean(total_rewards):7.3f}")
    log_and_console(f"Avg Length:     {np.mean(game_lengths):7.2f} steps")
    log_and_console("=" * 40 + "\n")

    # ========================================================
    #  Condensed Summary (Single-Line)
    # ========================================================
    condensed = (
        f"G {counts['Goat']} | "
        f"T {counts['Tiger']} | "
        f"MTO {counts['MaxTimeout']} | "
        f"RTO {counts['RepeatTimeout']} | "
        f"AR {np.mean(total_rewards):.2f} | "
        f"AL {np.mean(game_lengths):.3f}"
    )
    log_and_console("[CONDENSED] " + condensed + "\n")

    env.close()

    return {
        "matchup": matchup_label,
        "model_path": model_path,
        "goat_model_path": goat_model_path or "",
        "tiger_model_path": tiger_model_path or "",
        "episodes": int(n_games),
        "goat_wins": int(counts["Goat"]),
        "tiger_wins": int(counts["Tiger"]),
        "max_timeouts": int(counts["MaxTimeout"]),
        "repeat_timeouts": int(counts["RepeatTimeout"]),
        "unknown": int(counts["Unknown"]),
        "goat_win_rate": float(pct(counts["Goat"])),
        "tiger_win_rate": float(pct(counts["Tiger"])),
        "max_timeout_rate": float(pct(counts["MaxTimeout"])),
        "repeat_timeout_rate": float(pct(counts["RepeatTimeout"])),
        "unknown_rate": float(pct(counts["Unknown"])),
        "avg_reward": float(np.mean(total_rewards)),
        "avg_length": float(np.mean(game_lengths)),
    }


def save_sweep_results(results: List[dict]) -> Tuple[Optional[str], Optional[str]]:
    if not results or not SWEEP_CONFIG.get("save_results", True):
        return None, None

    results_dir = SWEEP_CONFIG.get("results_dir", "artifacts/eval_sweeps")
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(results_dir, f"sweep_results_{stamp}.json")
    csv_path = os.path.join(results_dir, f"sweep_results_{stamp}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = [
        "tiger_name",
        "goat_name",
        "matchup",
        "episodes",
        "goat_wins",
        "tiger_wins",
        "max_timeouts",
        "repeat_timeouts",
        "unknown",
        "goat_win_rate",
        "tiger_win_rate",
        "max_timeout_rate",
        "repeat_timeout_rate",
        "unknown_rate",
        "avg_reward",
        "avg_length",
        "model_path",
        "goat_model_path",
        "tiger_model_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    return json_path, csv_path


def run_sweep_mode() -> List[dict]:
    tiger_models: Dict[str, str] = SWEEP_CONFIG.get("tiger_models", {})
    goat_models: Dict[str, str] = SWEEP_CONFIG.get("goat_models", {})
    n_games = int(SWEEP_CONFIG.get("n_eval_games", N_EVAL_GAMES))
    deterministic = bool(SWEEP_CONFIG.get("deterministic", True))

    if not tiger_models:
        print("[SWEEP] No tiger models configured in SWEEP_CONFIG['tiger_models'].")
        return []
    if not goat_models:
        print("[SWEEP] No goat models configured in SWEEP_CONFIG['goat_models'].")
        return []

    print("\n" + "=" * 70)
    print(f"[SWEEP] Starting permutation sweep: {len(goat_models)} goat x {len(tiger_models)} tiger")
    print(f"[SWEEP] Games per matchup: {n_games} | Deterministic: {deterministic}")
    print("=" * 70)

    results: List[dict] = []

    for goat_name, goat_path in goat_models.items():
        for tiger_name, tiger_path in tiger_models.items():
            label = f"{goat_name} vs {tiger_name}"
            print(f"\n[SWEEP] Matchup: {label}")
            res = run_batch_evaluation(
                model_path=goat_path,
                n_games=n_games,
                deterministic=deterministic,
                learner_role=GOAT_LEARNER,
                tiger_ai=TIGER_AI_MODEL,
                goat_opponent_ai=GOAT_AI_RANDOM,
                tiger_model_path=tiger_path,
                summary_label=label,
            )
            if not res:
                continue
            res["tiger_name"] = tiger_name
            res["goat_name"] = goat_name
            results.append(res)

    print("\n" + "=" * 70)
    print("[SWEEP] FINAL SUMMARY")
    print("=" * 70)
    for row in results:
        print(
            f"{row['goat_name']} vs {row['tiger_name']} | "
            f"G {row['goat_wins']} | T {row['tiger_wins']} | "
            f"MTO {row['max_timeouts']} | RTO {row['repeat_timeouts']} | "
            f"AR {row['avg_reward']:.2f} | AL {row['avg_length']:.2f}"
        )

    # Compact recap block for quick scanning across all matchups.
    print("\n" + "=" * 70)
    print("[SWEEP] SHORT SUMMARY")
    print("=" * 70)
    for row in results:
        print(
            f"{row['goat_name']} vs {row['tiger_name']} | "
            f"G% {row['goat_win_rate']:.1f} | "
            f"T% {row['tiger_win_rate']:.1f} | "
            f"MTO {row['max_timeouts']} | "
            f"RTO {row['repeat_timeouts']}"
        )

    json_path, csv_path = save_sweep_results(results)
    if json_path and csv_path:
        print(f"[SWEEP] Saved JSON results: {json_path}")
        print(f"[SWEEP] Saved CSV results : {csv_path}")

    return results


def run_single_mode() -> None:
    learner_role = (SINGLE_CONFIG.get("learner_role", GOAT_LEARNER) or GOAT_LEARNER).lower()
    tiger_ai = SINGLE_CONFIG.get("tiger_ai", TIGER_AI_GREEDY)
    tiger_model_path = SINGLE_CONFIG.get("tiger_model_path", None)
    goat_opponent_ai = SINGLE_CONFIG.get("goat_opponent_ai", GOAT_AI_RANDOM)
    goat_model_path = SINGLE_CONFIG.get("goat_model_path", None)
    run_debug = bool(SINGLE_CONFIG.get("run_debug", DEBUG_MODE))
    n_debug_episodes = int(SINGLE_CONFIG.get("n_debug_episodes", N_DEBUG_EPISODES))
    n_eval_games = int(SINGLE_CONFIG.get("n_eval_games", N_EVAL_GAMES))

    explicit_cfg_model_path = SINGLE_CONFIG.get("model_path", None)
    explicit_path = explicit_cfg_model_path or EXPLICIT_MODEL_PATH
    if explicit_path is None:
        model_path = load_latest_model_path(MODELS_DIR)
    else:
        model_path = explicit_path

    if model_path is None:
        print(f"[!] No models found in '{MODELS_DIR}' (pattern *{MODEL_EXT}).")
        return

    if learner_role not in {GOAT_LEARNER, TIGER_LEARNER}:
        print(f"[!] Invalid SINGLE_CONFIG['learner_role']: {learner_role}")
        return

    if learner_role == TIGER_LEARNER and goat_opponent_ai == GOAT_AI_MODEL and not goat_model_path:
        print("[!] SINGLE_CONFIG requires goat_model_path when tiger learner uses goat_model opponent.")
        return
    if learner_role == GOAT_LEARNER and tiger_ai == TIGER_AI_MODEL and not tiger_model_path:
        print("[!] SINGLE_CONFIG requires tiger_model_path when goat learner uses tiger model opponent.")
        return

    mode_tag = f"single_{learner_role}"
    log_path = build_log_path(model_path, mode_tag=mode_tag)
    original_stdout, log_f = setup_logging(log_path, log_tag=mode_tag)

    try:
        if run_debug:
            run_debug_episodes(
                model_path=model_path,
                n_episodes=n_debug_episodes,
                learner_role=learner_role,
                tiger_ai=tiger_ai,
                goat_opponent_ai=goat_opponent_ai,
                goat_model_path=goat_model_path,
                tiger_model_path=tiger_model_path,
            )

        run_batch_evaluation(
            model_path=model_path,
            n_games=n_eval_games,
            learner_role=learner_role,
            tiger_ai=tiger_ai,
            goat_opponent_ai=goat_opponent_ai,
            goat_model_path=goat_model_path,
            tiger_model_path=tiger_model_path,
        )
        print("[DONE] Single-mode evaluation completed.")
    finally:
        sys.stdout = original_stdout
        log_f.close()


if __name__ == "__main__":
    mode = (EVAL_MODE or "single").strip().lower()
    if mode == "single":
        run_single_mode()
    elif mode == "sweep":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = build_log_path(None, mode_tag=f"sweep_{stamp}")
        original_stdout, log_f = setup_logging(log_path, log_tag="sweep")
        try:
            run_sweep_mode()
        finally:
            sys.stdout = original_stdout
            log_f.close()
    else:
        print(f"[!] Unknown EVAL_MODE='{EVAL_MODE}'. Use 'single' or 'sweep'.")


