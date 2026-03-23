# ============================================================
#  Project    : Tigers & Goats – Falcon Branch
#  Module     : Experiment Sweep Runner (Training + Variations)
#  File       : train_falcon.py
#  Version    : mppo_exp2.0
#  Last Update: 2025-12-27
#
#  What this file does:
#    This script runs an *experiment suite* made of multiple *variations*.
#    Each variation is a full training run (or multi-phase run) of a MaskablePPO
#    goat policy against a selected tiger opponent mode, using your unified
#    TnG environment + action masking wrappers.
#
#    The core idea is:
#      • You define a baseline + variations (ablations / weight changes / knobs).
#      • The script trains one model per variation.
#      • It writes models + checkpoints + TensorBoard metrics into clean folders
#        so you can compare learning curves and win-rates across variations.
#
#  Key concepts:
#    • EXPERIMENT_NAME:
#         A suite name. Everything (logs/models) is grouped under this.
#
#    • CORE (matchup tag):
#         Determined by TIGER_AI_MODE:
#           - GvNT: Goat vs Normal/Greedy Tiger
#           - GvST: Goat vs Smart Tiger
#
#    • VARIATIONS:
#         A dict mapping variation_name -> config.
#         A config can be:
#           (A) None
#               → “baseline / default settings” (env defaults)
#           (B) dict
#               → reward_weights overrides passed into the environment
#           (C) list of dict phases (curriculum / staged training)
#               → continues training the *same* model across phases, but can
#                 change timesteps/opponent/weights per phase.
#
#    • Action masking:
#         Each worker env is wrapped with ActionMasker so MaskablePPO only
#         samples legal actions. This is required for stable learning.
#
#  Outputs (where stuff goes):
#    • TensorBoard logs:
#         artifacts/logging/train/{ALGO_TAG}/{CORE}/{EXPERIMENT_NAME}/
#
#    • Models + checkpoints:
#         artifacts/models/train/{ALGO_TAG}/{CORE}/{EXPERIMENT_NAME}/
#
#    • Naming conventions:
#         - Checkpoints include cp_{ALGO_TAG}_{CORE}_{variation}_...
#         - Final model is {ALGO_TAG}_{CORE}_{variation}.zip
#           (auto-suffixed to avoid overwriting)
#
#  Metrics you should expect in TensorBoard:
#    • Overall episode counts + win/loss outcome rates (goat win, tiger win,
#      timeouts, stalls), plus windowed (recent-N) versions so you can judge
#      stability rather than noisy single-episode results.
#
#  Quick Use:
#    1) Choose the matchup (tiger opponent):
#         TIGER_AI_MODE = TIGER_AI_GREEDY   # “normal/greedy”
#         # or
#         TIGER_AI_MODE = TIGER_AI_SMART    # “smart tiger”
#
#    2) Set the experiment identity + compute:
#         EXPERIMENT_NAME = "my_ablation_suite"
#         TIMESTEPS       = 10_000_000
#         NUM_CPU         = 8
#         DEVICE_MODE     = "cuda"  # or "cpu"
#
#    3) Define variations:
#         VARIATIONS = {
#             "defaultSettings": None,
#             "no_bubble": {"bubble": 0.0},
#             "no_block": {"block_tiger": 0.0},
#             "weaker_capture_bias": {"BASE_TIGER_CAPTURE_BIAS": 0.8},
#         }
#
#       Curriculum-style (multi-phase) example:
#         VARIATIONS = {
#           "curriculumA": [
#              {"timesteps": 5_000_000,  "tiger_ai": TIGER_AI_GREEDY,
#               "reward_weights": {"bubble": 1.0, "block_tiger": 1.0}},
#              {"timesteps": 15_000_000, "tiger_ai": TIGER_AI_SMART,
#               "reward_weights": {"bubble": 0.5, "block_tiger": 0.5}},
#           ]
#         }
#
#    4) Run:
#         python experiment_sweep.py
#
#    5) Watch training:
#         tensorboard --logdir artifacts/logging/train --port 6006
#
#  Workflow (recommended for ablation studies):
#    1) Establish a baseline:
#         - Run VARIATIONS={"defaultSettings": None}
#         - Confirm it learns and produces stable win-rate trends.
#
#    2) One-at-a-time ablations:
#         - For each shaping term, set it to 0.0 while keeping others the same.
#         - Compare:
#             • time-to-first-wins
#             • asymptotic win-rate
#             • stability (windowed curves)
#
#    3) Group ablations (optional):
#         - Remove entire families: “movement shaping”, “anti-capture shaping”, etc.
#         - Helps detect interactions between terms.
#
#    4) Magnitude sweeps (optional):
#         - If a term matters, sweep it: {0.0, 0.25, 0.5, 1.0, 2.0}
#         - Look for diminishing returns or destabilization.
#
#    5) Curriculum runs (optional but powerful):
#         - Train vs greedy tiger first to learn basics quickly
#         - Switch to smart tiger to harden strategy
#         - Keep logging separated by phase to see where improvements happen.
#
#    6) Evaluate the winners:
#         - Use eval_falcon.py (or your eval script) to run N episodes per model
#           and compare win-rate + qualitative behavior (sacrifices, traps, etc.).
#
# ============================================================


import os
import numpy as np
import torch
import glob
from collections import deque
import random

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env_goat_falcon import (
    TnGEnv as BaseEnv,
    FlattenTnGActionWrapper as FlattenWrapper,
    TIGER_AI_GREEDY,
    TIGER_AI_SMART,
)

GOAT_LEARNER  = "goat"
TIGER_LEARNER = "tiger"

def core_tag(learner_role: str, tiger_ai: str, mix: bool = False) -> str:
    learner = "G" if learner_role == GOAT_LEARNER else "T"
    if mix:
        opp = "MixT"
    else:
        opp = "NT" if tiger_ai == TIGER_AI_GREEDY else "ST"
    return f"{learner}v{opp}"


# ============================================================
#  USER CONFIG — Opponent & Naming
# ============================================================

EXPERIMENT_NAME = "RobustGoatTraining"
TIGER_AI_MODE   = TIGER_AI_SMART
LEARNER_ROLE    = GOAT_LEARNER
ALGO_TAG        = "mppo"
ENV_VER         = "env_4.0"
MODEL_VER       = "mppo_exp2.0"

USE_MIX_TAG     = True
CORE            = core_tag(LEARNER_ROLE, TIGER_AI_MODE, mix=USE_MIX_TAG)
SUITE_TAG       = EXPERIMENT_NAME

# ============================================================
#  USER CONFIG — Scale / Hardware
# ============================================================

DEVICE_MODE = "gpu"
DEBUG_MODE  = False
TIMESTEPS   = 1_000_000 if DEBUG_MODE else 60_000_000
NUM_CPU     = 16
SEED        = 42

# ============================================================
#  USER CONFIG — PPO Hyperparameters
# ============================================================

GAMMA         = 0.99
LEARNING_RATE = 3e-4
ENT_COEF      = 0.02

if DEVICE_MODE == "gpu":
    DEVICE     = "cuda"
    N_STEPS    = 4096
    BATCH_SIZE = 256
    N_EPOCHS   = 10
    torch.set_num_threads(1)
elif DEVICE_MODE == "cpu":
    DEVICE     = "cpu"
    N_STEPS    = 1024
    BATCH_SIZE = 128
    N_EPOCHS   = 10
else:
    raise ValueError("DEVICE_MODE must be 'gpu' or 'cpu'")

# ============================================================
#  USER CONFIG — Artifacts / Checkpointing
# ============================================================

LOG_DIR   = f"artifacts/logging/train/{ALGO_TAG}/{CORE}/{SUITE_TAG}"
MODEL_DIR = f"artifacts/models/train/{ALGO_TAG}/{CORE}/{SUITE_TAG}"

CHECKPOINTS_PER_RUN = 10
CHECKPOINT_INTERNAL_STEPS = TIMESTEPS // CHECKPOINTS_PER_RUN
SAVE_FREQ = max(CHECKPOINT_INTERNAL_STEPS // NUM_CPU, 1)

RESUME_MODEL_PATH = None

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
#  VARIATIONS
# ============================================================

VARIATIONS = {
  
  "greedy_to_smart_mix": [
    {"timesteps": 10_000_000, "tiger_ai": TIGER_AI_GREEDY, "reward_weights": None},
    {"timesteps": 10_000_000, "tiger_ai": TIGER_AI_SMART,  "reward_weights": None},
    {"timesteps": 30_000_000, "tiger_ai": TIGER_AI_SMART,  "mix_prob": 0.3, "reward_weights": None},
    {"timesteps": 30_000_000, "tiger_ai": TIGER_AI_SMART,  "mix_prob": 0.5, "reward_weights": None}
  ],
}

# ============================================================
# Helpers
# ============================================================

def mask_fn(env):
    return env.unwrapped.get_action_mask()

def tiger_code(tiger_ai: str) -> int:
    return 1 if tiger_ai == TIGER_AI_SMART else 0

def tb_add_text(model_obj, tag: str, text: str, step: int):
    """
    Optional: add Text to TensorBoard if a TB writer exists.
    Safe no-op if not.
    """
    try:
        for fmt in model_obj.logger.output_formats:
            w = getattr(fmt, "writer", None)
            if w is not None:
                w.add_text(tag, text, step)
                break
    except Exception:
        pass

# ============================================================
# Callbacks
# ============================================================

class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_count = 0

    def _on_step(self):
        dones = self.locals.get("dones", None)
        if dones is not None:
            self.episode_count += int(np.sum(dones))
        self.logger.record("1.episode_stats/2.episodes_total_done", float(self.episode_count))
        return True

class WinStatsCallback(BaseCallback):
    """
    Windowed win/timeout stats + optional realized tiger-mode tracking.
    If env includes info["tiger_ai"] at episode end, we will track it.
    """
    def __init__(self, log_every_episodes: int = 200, window_episodes: int = 1_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_episodes = log_every_episodes
        self.last_logged_episodes = 0

        # cumulative
        self.episodes = 0
        self.goat_wins = 0
        self.tiger_wins = 0
        self.goat_timeouts = 0
        self.stall_timeouts = 0

        # rolling window of winners
        self.window = deque(maxlen=window_episodes)

        # rolling window of realized tiger mode (optional)
        self.window_tiger_ai = deque(maxlen=window_episodes)  # values: "greedy"/"smart"/None

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if not done:
                    continue

                winner = info.get("winner", None)

                self.episodes += 1
                if winner == "Goat":
                    self.goat_wins += 1
                elif winner == "Tiger":
                    self.tiger_wins += 1
                elif winner == "GoatTimeout":
                    self.goat_timeouts += 1
                elif winner == "StallTimeout":
                    self.stall_timeouts += 1

                self.window.append(winner)

                # optional: realized tiger mode
                # env_tng_falcon can set info["tiger_ai"] = TIGER_AI_GREEDY/SMART (or strings)
                self.window_tiger_ai.append(info.get("tiger_ai", None))

        # always keep current episode index in logs
        self.logger.record("1.episode_stats/1.episode_index", float(self.episodes))

        if (self.episodes - self.last_logged_episodes) >= self.log_every_episodes and self.episodes > 0:
            total = float(self.episodes)

            self.last_logged_episodes = self.episodes
            # Order of logs (to control TB display grouping)
            self.logger.record("1.episode_stats/3.episodes_total_winstats", total)
            self.logger.record("1.episode_stats/4.goat_win_rate", self.goat_wins / total)
            self.logger.record("1.episode_stats/5.tiger_win_rate", self.tiger_wins / total)
            self.logger.record("1.episode_stats/6.goat_timeout_rate", self.goat_timeouts / total)
            self.logger.record("1.episode_stats/7.stall_timeout_rate", self.stall_timeouts / total)

            w = len(self.window)
            if w > 0:
                goat = sum(1 for x in self.window if x == "Goat") / w
                tiger = sum(1 for x in self.window if x == "Tiger") / w
                to = sum(1 for x in self.window if x == "GoatTimeout") / w
                st = sum(1 for x in self.window if x == "StallTimeout") / w

                self.logger.record("2.window_stats/1.episodes_tracked", float(w))
                self.logger.record("2.window_stats/2.goat_win_rate", goat)
                self.logger.record("2.window_stats/3.tiger_win_rate", tiger)
                self.logger.record("2.window_stats/4.goat_timeout_rate", to)
                self.logger.record("2.window_stats/5.stall_timeout_rate", st)

                # optional realized tiger distribution in the last-N window
                smart = 0
                greedy = 0
                for v in self.window_tiger_ai:
                    if v == TIGER_AI_SMART or v == "smart" or v == "SMART":
                        smart += 1
                    elif v == TIGER_AI_GREEDY or v == "greedy" or v == "GREEDY":
                        greedy += 1
                denom = smart + greedy
                if denom > 0:
                    self.logger.record("2.window_stats/6.realized_greedy_frac", greedy / denom)
                    self.logger.record("2.window_stats/7.realized_smart_frac", smart / denom)

        return True
    

class RealizedTigerStatsCallback(BaseCallback):
    def __init__(self, log_every_episodes: int = 200, window_episodes: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every_episodes = log_every_episodes
        self.last_logged = 0
        self.episodes = 0
        self.smart = 0
        self.greedy = 0
        self.window = deque(maxlen=window_episodes)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue

            tiger_ai = info.get("episode_tiger_ai", info.get("tiger_ai", None))

            if tiger_ai in (TIGER_AI_SMART, "smart", "SMART"):
                self.episodes += 1
                self.smart += 1
                self.window.append(1)
            elif tiger_ai in (TIGER_AI_GREEDY, "greedy", "GREEDY"):
                self.episodes += 1
                self.greedy += 1
                self.window.append(0)
            else:
                # unknown -> don't count
                continue

        if (self.episodes - self.last_logged) >= self.log_every_episodes and self.episodes > 0:
            self.last_logged = self.episodes
            total = float(self.episodes)

            self.logger.record("3.meta/3.realized_greedy_rate", self.greedy / total)
            self.logger.record("3.meta/4.realized_smart_rate", self.smart / total)

        return True


class TigerMixWrapper(Monitor):
    """
    Monitor wrapper that re-samples tiger_ai on every reset using a callable.
    Keeps it top-level (pickle-friendly) and avoids nested class definitions.
    """
    def __init__(self, env, sample_ai_fn):
        super().__init__(env)
        self._sample_ai_fn = sample_ai_fn

    def reset(self, *, seed=None, options=None):
        # Make sure we are modifying the underlying env that actually owns tiger_ai
        self.tiger_ai = self._sample_ai_fn()
        return super().reset(seed=seed, options=options)


# ============================================================
# Env factory (supports per-reset mixing)
# ============================================================

def make_env(rank: int, seed: int, reward_weights, tiger_ai=None, mix_prob=None):
    def _init():
        def _sample_ai():
            if mix_prob is None:
                return tiger_ai if tiger_ai is not None else TIGER_AI_MODE
            p = min(max(float(mix_prob), 0.0), 1.0)
            return TIGER_AI_SMART if random.random() < p else TIGER_AI_GREEDY

        base_env = BaseEnv(
            reward_weights=reward_weights,
            tiger_ai=_sample_ai(),
        )

        # If mixing, resample tiger_ai on every reset
        if mix_prob is not None:
            env = TigerMixWrapper(base_env, _sample_ai)
        else:
            env = Monitor(base_env)

        env = FlattenWrapper(env)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed + rank)
        return env
    return _init

# ============================================================
# Resume helpers
# ============================================================

def resolve_resume_path(path: str | None) -> str | None:
    if not path:
        return None
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        zips = glob.glob(os.path.join(path, "*.zip"))
        if not zips:
            return None
        return max(zips, key=os.path.getctime)
    return None

def unique_filename(directory: str, base_name: str, ext: str = ".zip") -> str:
    full_path = os.path.join(directory, base_name + ext)
    if not os.path.exists(full_path):
        return full_path
    counter = 1
    while True:
        candidate = os.path.join(directory, f"{base_name}_{counter}{ext}")
        if not os.path.exists(candidate):
            return candidate
        counter += 1

# ============================================================
# Tiger meta logger (scalar + optional text)
# ============================================================

def log_phase_meta(model_obj, tiger_ai, mix_prob):
    tiger_id = 1.0 if tiger_ai == TIGER_AI_SMART else 0.0
    model_obj.logger.record("3.meta/1.phase_is_mixed", 0.0 if mix_prob is None else 1.0)
    model_obj.logger.record("3.meta/2.phase_tiger_ai_id", tiger_id)
    if mix_prob is not None:
        model_obj.logger.record("meta/5.phase_mix_prob", float(mix_prob))
    model_obj.logger.dump(step=model_obj.num_timesteps)

# ============================================================
# Core: run one variation
# ============================================================

def run_single_variation(variation_name: str, reward_weights):
    set_random_seed(SEED)

    print(f"Starting {NUM_CPU} env workers ({CORE})...")
    print(f"ENV_VER={ENV_VER}, MODEL_VER={MODEL_VER}, TIMESTEPS={TIMESTEPS}")
    print(f"DEVICE_MODE={DEVICE_MODE}, DEVICE={DEVICE}")
    print("=" * 70)
    print(f"Starting variation: {variation_name}")
    print("=" * 70)

    tb_subdir = f"{ALGO_TAG}_{CORE}_{variation_name}"

    if isinstance(reward_weights, list):
        phases = reward_weights
    elif isinstance(reward_weights, dict):
        # Single-phase dict: allow inline timesteps/tiger_ai/mix_prob/checkpoints_per_run
        phase_timesteps = reward_weights.get("timesteps", TIMESTEPS)
        phase_checkpoints = reward_weights.get("checkpoints_per_run", CHECKPOINTS_PER_RUN)
        phase_tiger = reward_weights.get("tiger_ai", TIGER_AI_MODE)
        phase_mix = reward_weights.get("mix_prob", None)
        phase_rewards = {
            k: v for k, v in reward_weights.items()
            if k not in {"timesteps", "tiger_ai", "mix_prob", "checkpoints_per_run"}
        } or None
        phases = [{
            "timesteps": phase_timesteps,
            "tiger_ai": phase_tiger,
            "mix_prob": phase_mix,
            "checkpoints_per_run": phase_checkpoints,
            "reward_weights": phase_rewards,
        }]
    else:
        phases = [{
            "timesteps": TIMESTEPS,
            "tiger_ai": TIGER_AI_MODE,
            "reward_weights": reward_weights,
            "checkpoints_per_run": CHECKPOINTS_PER_RUN,
        }]

    policy_kwargs = dict(net_arch=[256, 256, 256])

    resume_path = resolve_resume_path(RESUME_MODEL_PATH)
    if RESUME_MODEL_PATH and not resume_path:
        print(f"[{variation_name}] [WARN] RESUME_MODEL_PATH set but no .zip found at: {RESUME_MODEL_PATH}")
    resumed = bool(resume_path)

    model = None

    total_goat = total_tiger = total_to = total_st = 0
    total_eps = 0

    for phase_idx, phase in enumerate(phases):
        phase_steps = phase.get("timesteps", TIMESTEPS)
        phase_rewards = phase.get("reward_weights", None)
        phase_tiger = phase.get("tiger_ai", TIGER_AI_MODE)
        phase_mix = phase.get("mix_prob", None)
        phase_cpr = phase.get("checkpoints_per_run", CHECKPOINTS_PER_RUN)
        phase_save_freq = max((phase_steps // max(phase_cpr, 1)) // NUM_CPU, 1)

        print(f"Phase {phase_idx}: timesteps={phase_steps}, tiger_ai={phase_tiger}, mix_prob={phase_mix}")

        env_fns = [
            make_env(
                i,
                seed=SEED,
                reward_weights=phase_rewards,
                tiger_ai=phase_tiger,
                mix_prob=phase_mix,
            )
            for i in range(NUM_CPU)
        ]
        vec_env = SubprocVecEnv(env_fns)

        win_stats_callback = WinStatsCallback()
        ep_callback = EpisodeCounterCallback()

        realized_tiger_callback = RealizedTigerStatsCallback(
            log_every_episodes=200, 
            window_episodes=1000
        )

        chk_callback = CheckpointCallback(
            save_freq=phase_save_freq,
            save_path=MODEL_DIR,
            name_prefix=f"cp_{ALGO_TAG}_{CORE}_{variation_name}",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        callback = CallbackList([
            chk_callback, 
            ep_callback, 
            win_stats_callback, 
            realized_tiger_callback
            ]
        )

        phase_tb = f"{tb_subdir}_p{phase_idx}"
        log_path = os.path.join(LOG_DIR, phase_tb)

        if model is None:
            if resume_path:
                print(f"[{variation_name}] Loading existing model for continued training:\n  {resume_path}")
                model = MaskablePPO.load(resume_path, env=vec_env, device=DEVICE)
                model.set_env(vec_env)
                model.verbose = 1
                model.set_logger(configure(log_path, ["tensorboard", "stdout"]))
            else:
                model = MaskablePPO(
                    "MlpPolicy",
                    vec_env,
                    verbose=1,
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE,
                    n_epochs=N_EPOCHS,
                    gamma=GAMMA,
                    learning_rate=LEARNING_RATE,
                    ent_coef=ENT_COEF,
                    device=DEVICE,
                    seed=SEED,
                    policy_kwargs=policy_kwargs,
                )
                # KEY FIX: configure logger immediately
                model.set_logger(configure(log_path, ["tensorboard", "stdout"]))
        else:
            model.set_env(vec_env)
            model.set_logger(configure(log_path, ["tensorboard", "stdout"]))

        cont = resumed or phase_idx > 0

        # log tiger metadata AFTER logger exists
        log_phase_meta(model, phase_tiger, phase_mix)


        model.learn(
            total_timesteps=phase_steps,
            callback=callback,
            progress_bar=True,
            reset_num_timesteps=not cont,
        )

        total_eps += win_stats_callback.episodes
        total_goat += win_stats_callback.goat_wins
        total_tiger += win_stats_callback.tiger_wins
        total_to += win_stats_callback.goat_timeouts
        total_st += win_stats_callback.stall_timeouts

        vec_env.close()

    if total_eps > 0:
        goat = total_goat / float(total_eps)
        tiger = total_tiger / float(total_eps)
        to = total_to / float(total_eps)
        st = total_st / float(total_eps)
    else:
        goat = tiger = to = st = 0.0

    print(f"[{variation_name}] Summary:")
    print(f"  Goat {goat:.3f} | Tiger {tiger:.3f} | Timeout {to:.3f} | Stall {st:.3f}")

    # Save final model with phase suffix similar to TB naming
    final_name = f"{ALGO_TAG}_{CORE}_{variation_name}_p{len(phases)-1}"
    final_path = unique_filename(MODEL_DIR, final_name)

    model.save(final_path)
    print(f"Saved final model: {final_path}")

    return {
        "variation": variation_name,
        "episodes": total_eps,
        "goat": goat,
        "tiger": tiger,
        "timeout": to,
        "stall": st,
        "final_path": final_path,
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    all_results = []

    print("\n" + "=" * 70)
    print(f"EXPERIMENT (suite): {EXPERIMENT_NAME}")
    print(f"ALGO={ALGO_TAG} | MATCHUP={CORE} | VARIATIONS={len(VARIATIONS)}")
    print("=" * 70)

    for variation_name, weights in VARIATIONS.items():
        res = run_single_variation(variation_name, weights)
        all_results.append(res)

    print("\n=================== VARIATION RESULTS ===================")
    for r in all_results:
        print(
            f"{r['variation']:16s} | "
            f"G {r['goat']:.3f} | "
            f"T {r['tiger']:.3f} | "
            f"TO {r['timeout']:.3f} | "
            f"ST {r['stall']:.3f} | "
            f"Eps {r['episodes']}"
        )
