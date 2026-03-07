# ============================================================
#  Project    : Tigers & Goats - Falcon Branch
#  Module     : Unified Debug + Evaluation Runner (Greedy vs Smart Tiger)
#  File       : eval_falcon.py
#  Version    : eval2.0
#  Last Update: 2025-12-27
#
#  Purpose / Goal:
#    Provide a single script to (1) step-debug a trained goat policy and
#    (2) batch-evaluate win/timeout stats against either tiger opponent,
#    without swapping environment files.
#
#  Overview:
#    - Always uses env_tng_falcon.TnGEnv (unified env backend)
#    - Selects tiger opponent at runtime via CLI:
#        • normal / greedy  -> TIGER_AI_GREEDY  (ENV_NAME="NORMAL")
#        • battle / smart   -> TIGER_AI_SMART   (ENV_NAME="BATTLE")
#    - Two capabilities:
#        1) Step-by-step Debug:
#           • runs N_DEBUG_EPISODES episodes
#           • prints decoded actions (flat -> pos,dir)
#           • shows masks, rewards, termination/truncation, and winner info
#           • optionally renders the board each turn if env.render() exists
#        2) Batch Evaluation:
#           • runs N_EVAL_GAMES episodes deterministically
#           • tallies Goat / Tiger / GoatTimeout / StallTimeout outcomes
#           • reports average reward + average episode length
#           • prints a condensed single-line summary for quick comparisons
#
#  Workflow:
#    1) Train a model (saved as .zip in artifacts/models/experiment/)
#    2) Run debug + eval:
#         python eval_falcon.py
#           -> NORMAL (greedy tiger), latest model in MODELS_DIR
#
#         python eval_falcon.py my_model.zip
#           -> NORMAL (greedy tiger), evaluate explicit model path
#
#         python eval_falcon.py normal my_model.zip
#           -> NORMAL (greedy tiger), evaluate explicit model path
#
#         python eval_falcon.py battle
#           -> BATTLE (smart tiger), latest model in MODELS_DIR
#
#         python eval_falcon.py battle my_model.zip
#           -> BATTLE (smart tiger), evaluate explicit model path
#
#    3) Review the log file under artifacts/logging/eval_debug/
#
#  Quick Use:
#    - Change evaluation size:
#        N_DEBUG_EPISODES = 3
#        N_EVAL_GAMES     = 100
#    - Console mirroring (in addition to file logging):
#        PRINT_DEBUG_TO_CONSOLE = True
#    - Safety cap for debug loops:
#        MAX_STEPS_PER_EP = 100
#
#  Compatibility:
#    - stable-baselines3 + sb3-contrib:
#        • MaskablePPO
#        • ActionMasker + get_action_masks
#    - Uses native Discrete action env + masking (no flatten wrapper)
#    - Designed to pair with:
#        • env_tng_falcon.py     (unified greedy/smart tiger env)
#        • experiment_sweep.py   (Reward variation sweeps that produce models to eval)
# ============================================================



import os
import glob
import sys
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
#  ENVIRONMENT SWITCH
#    TIGER_AI_MODE picks greedy vs smart tiger in the unified env
# ============================================================
TIGER_AI_MODE, EXPLICIT_MODEL_PATH = parse_cli_args()
ENV_NAME = "BATTLE" if TIGER_AI_MODE == TIGER_AI_SMART else "NORMAL"
OPPONENT_LABEL = "GOAT VS SMART TIGER" if TIGER_AI_MODE == TIGER_AI_SMART else "GOAT VS GREEDY TIGER"


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


def build_log_path(model_path: Optional[str]) -> str:
    """
    Build a log file path inside DEBUG_LOG_DIR based on the model filename.

    Examples:
      models/battle_falcon_final_env1.0_mppo_battle1.0_run_3.zip
        -> debug_logs/battle_falcon_final_env1.0_mppo_battle1.0_run_3_debug_eval.txt

      models/battle_falcon_env1.0_mppo_battle1.0_250000_steps.zip
        -> debug_logs/battle_falcon_env1.0_mppo_battle1.0_250000_steps_debug_eval.txt
    """
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

    # Tag to indicate which env was used for this eval
    env_tag = ENV_NAME.lower()  # "normal" or "battle"

    if model_path is None:
        stub = "latest_model"
    else:
        base = os.path.basename(model_path)
        stub = os.path.splitext(base)[0]

    filename = f"{stub}_{env_tag}_eval.txt"
    return os.path.join(DEBUG_LOG_DIR, filename)


def setup_logging(log_path: str) -> Tuple[TextIO, TextIO]:
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
    print(f"[LOG] Debug + Eval log started ({ENV_NAME}): {log_path}")
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


def make_debug_env(seed: Optional[int] = SEED):
    """
    Build the debug environment:

      TnGEnv(tiger_ai=...) -> Monitor -> ActionMasker
    """
    env = BaseEnv(tiger_ai=TIGER_AI_MODE)
    env = Monitor(env)
    env = ActionMasker(env, _mask_fn)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env


def make_eval_env():
    """
    Build the evaluation environment:

      TnGEnv(tiger_ai=...) -> ActionMasker
    """
    env = BaseEnv(tiger_ai=TIGER_AI_MODE)
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
) -> None:
    """
    Run a small number of evaluation episodes with a loaded model,
    printing/logging each step, including decoded action and env.render().
    """
    print("\n" + "=" * 60)
    print(f"[DBG] STEP-BY-STEP {OPPONENT_LABEL} DEBUG EPISODES")
    print("=" * 60)

    print(f"[DBG] Loading model: {model_path}")
    model = MaskablePPO.load(model_path)

    env = make_debug_env(seed=SEED)

    wins: Dict[str, int] = {
        "Tiger": 0,
        "Goat": 0,
        "GoatTimeout": 0,
        "StallTimeout": 0,
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

        winner = info.get("winner", "Unknown")
        wins[winner] = wins.get(winner, 0) + 1
        rewards_all.append(ep_reward)

        print(
            f"Episode {ep:02d} SUMMARY | "
            f"Steps: {step_idx} | Reward: {ep_reward:.3f} | Winner: {winner}"
        )

    # Overall debug summary
    mean_r = float(np.mean(rewards_all)) if rewards_all else 0.0
    std_r = float(np.std(rewards_all)) if rewards_all else 0.0

    log_and_console(f"\n[DBG] ========= OVERALL {OPPONENT_LABEL} DEBUG SUMMARY =========")
    log_and_console(f"[DBG] Episodes run : {n_episodes}")
    log_and_console(f"[DBG] Avg Reward   : {mean_r:.3f}  (±{std_r:.3f})")
    log_and_console(
        "[DBG] Wins         : "
        f"Tiger={wins.get('Tiger', 0)}, "
        f"Goat={wins.get('Goat', 0)}, "
        f"GoatTimeout={wins.get('GoatTimeout', 0)}, "
        f"StallTimeout={wins.get('StallTimeout', 0)}, "
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
        "Goat", "Tiger", "GoatTimeout", "StallTimeout", or "Unknown"
    """
    winner = info.get("winner", None)

    if winner in ("Goat", "Tiger", "GoatTimeout", "StallTimeout"):
        return winner

    # Fallback based on reward if env didn't provide a winner
    if episode_reward >= WIN_REWARD_THRESHOLD:
        return "Goat"
    if episode_reward <= LOSS_REWARD_THRESHOLD:
        return "Tiger"
    return "Unknown"


# ============================================================
#  BATCH EVALUATION LOOP
# ============================================================

def run_batch_evaluation(
    model_path: str,
    n_games: int = N_EVAL_GAMES,
    model_dir: str = MODELS_DIR,
) -> None:
    """
    Evaluate a MaskablePPO model over n_games episodes.

    Prints:
        - Goat/Tiger/Timeout counts and percentages
        - Average reward
        - Average episode length
        - Condensed single-line summary
    """
    print("\n" + "=" * 60)
    print(f"[EVAL] {OPPONENT_LABEL} EVALUATION")
    print("=" * 60)

    if not os.path.isfile(model_path):
        print(f"[EVAL] Model file not found: {model_path}")
        return

    print(f"[EVAL] Loading model from: {model_path}")
    model = MaskablePPO.load(model_path)

    env = make_eval_env()

    wins = 0
    losses = 0
    timeouts = 0
    stall_timeouts = 0

    total_rewards: List[float] = []
    game_lengths: List[int] = []

    print(f"[EVAL] Starting evaluation of {n_games} games...")
    print(f"[EVAL] Deterministic policy: {DETERMINISTIC_EVAL}")

    for i in range(n_games):
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward: float = 0.0
        steps = 0

        while not (terminated or truncated):
            # Get current action mask from wrapped env
            action_masks = get_action_masks(env)

            action, _ = model.predict(
                obs,
                action_masks=action_masks,
                deterministic=DETERMINISTIC_EVAL,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

        total_rewards.append(episode_reward)
        game_lengths.append(steps)

        outcome = classify_outcome(episode_reward, info)
        if outcome == "Goat":
            wins += 1
        elif outcome == "Tiger":
            losses += 1
        elif outcome == "GoatTimeout":
            timeouts += 1
        elif outcome == "StallTimeout":
            stall_timeouts += 1

    # Summary stats
    def pct(count: int) -> float:
        return (count / n_games * 100.0) if n_games > 0 else 0.0

    log_and_console("\n" + "=" * 40)
    log_and_console(f"  {OPPONENT_LABEL} EVALUATION RESULTS (N={n_games})")
    log_and_console("=" * 40)

    log_and_console(f"Goat Wins :     {wins:4d} ({pct(wins):5.1f}%)")
    log_and_console(f"Tiger Wins:     {losses:4d} ({pct(losses):5.1f}%)")
    log_and_console(f"Goat Timeout  : {timeouts:4d} ({pct(timeouts):5.1f}%)")
    log_and_console(f"Stall Timeout  : {stall_timeouts:4d} ({pct(stall_timeouts):5.1f}%)")

    log_and_console("-" * 40)
    log_and_console(f"Avg Reward:     {np.mean(total_rewards):7.3f}")
    log_and_console(f"Avg Length:     {np.mean(game_lengths):7.2f} steps")
    log_and_console("=" * 40 + "\n")

    # ========================================================
    #  Condensed Summary (Single-Line)
    # ========================================================
    condensed = (
        f"G {wins} | "
        f"T {losses} | "
        f"TO {timeouts} | "
        f"ST {stall_timeouts} | "
        f"AR {np.mean(total_rewards):.2f} | "
        f"AL {np.mean(game_lengths):.3f}"
    )
    log_and_console("[CONDENSED] " + condensed + "\n")


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    # 1) Optional command-line arg: path to a specific checkpoint
    explicit_path = EXPLICIT_MODEL_PATH

    # 2) Resolve model path (explicit > latest in directory)
    if explicit_path is None:
        model_path = load_latest_model_path(MODELS_DIR)
    else:
        model_path = explicit_path

    if model_path is None:
        # No logging redirection yet, so this prints to console
        print(f"[!] No models found in '{MODELS_DIR}' (pattern *{MODEL_EXT}).")
        sys.exit(1)

    # 3) Build log path based on model filename (preserves run id)
    log_path = build_log_path(model_path)

    # 4) Redirect stdout -> log (and optionally console)
    original_stdout, log_f = setup_logging(log_path)

    try:
        # 5) Run step-by-step debug episodes
        run_debug_episodes(model_path=model_path)

        # 6) Append a batch evaluation to the same log file
        run_batch_evaluation(model_path=model_path, n_games=N_EVAL_GAMES)

        print(f"[DONE] {OPPONENT_LABEL} debug + evaluation completed.")
    finally:
        # 7) Restore stdout and close log file
        sys.stdout = original_stdout
        log_f.close()
