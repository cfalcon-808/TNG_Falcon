# ============================================================
#  Project    : Tigers & Goats – Falcon Branch
#  Module     : Maskable PPO Environment (Full Game)
#  File       : env_tng_falcon.py
#  Version    : env3.2
#  Last Update: 2025-12-06
#
#  Overview:
#    Full Tigers & Goats environment with shaped rewards,
#    action masking, repetition detection, and a unified
#    configuration system built around two dictionaries:
#
#        • DEFAULT_KNOBS   – Base magnitudes and constants
#        • DEFAULT_WEIGHTS – Shaping-term multipliers
#
#    All numeric values used in reward calculation are taken
#    from these dictionaries via the helper:
#
#         self._w(key)
#
#    This allows the caller to change any reward component,
#    constant, penalty, or shaping magnitude at runtime without
#    modifying the environment source.
#
#  Knob/weight Dictionaries:
#
#    • To modify a shaping term (weight):
#         env = TnGEnv(reward_weights={ "bubble": 0.0 })
#         → disables bubble shaping
#
#    • To modify a constant (knob):
#         env = TnGEnv(reward_weights={ "REWARD_GOAT_WIN": 5.0 })
#         → increases goat-win reward magnitude
#
#    • To adjust tiger behavior:
#         env = TnGEnv(reward_weights={ "BASE_TIGER_CAPTURE_BIAS": 0.7 })
#         → makes the tiger less greedy for captures
#
#    • self._w(key) resolves keys in both dicts:
#         - If key exists in DEFAULT_WEIGHTS → return weight
#         - If key exists in DEFAULT_KNOBS  → return constant
#
#    This makes all reward logic flexible, tunable, and safe
#    for integration with Maskable PPO training loops.
#
# ============================================================
#  Implementation Notes:
#    • Ablation weights accessed via self._w(term)
#    • All shaping signals, penalties, and terminal rewards
#      multiplied by the appropriate weight
#    • Fully backward compatible: reward_weights=None preserves
#      original env behavior exactly
# ============================================================


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from env_core_falcon import (
    BOARD_SIZE,
    DIR_CODES,
    TOTAL_GOATS_TO_PLACE,
    GOATS_EATEN_FOR_TIGER_WIN,
    TIGER_START_POSITIONS,
    KEY_CENTERS,
    DEBUG_INVALID,
    FalconEnvCore,  # shared board mechanics / render
)


# ============================================================
#  GLOBAL TUNING KNOBS
#  All numeric values (constants + shaping magnitudes + scales)
#  ***Note: these can be changed in the training file***
# ============================================================

DEFAULT_KNOBS = {

    # Core terminal rewards
    "REWARD_GOAT_WIN":          3.2,    # reward when goats immobilize all tigers
    "REWARD_TIGER_WIN":        -3.2,    # penalty when tigers win by eating goats

    # Step & goat-eaten shaping
    "REWARD_STEP":             -0.001,  # per-step penalty to discourage stalling
    "REWARD_GOAT_EATEN":       -0.35,   # penalty for each goat captured by tigers

    # Timeout scales
    "GOAT_TIMEOUT_SCALE":       1.0,    # scale for turn-limit (max turns) timeout penalty
    "GOAT_STALL_SCALE":         0.8,    # scale for stall-timeout (repeat-state) penalty
    "MAX_TURNS":                100,    # max turns before enforcing goat-timeout

    # Move penalty shaping
    "MOVE_STEP_BASE":          -0.01,   # base negative reward for moving-phase steps
    "MOVE_STEP_SLOPE":         -0.01,   # step penalty increases as move_steps grows
    "MOVE_STEP_MIN":           -0.6,    # minimum move penalty (lower bound)

    # Decay for goat win  
    "GOAT_WIN_TURN_DECAY":      0.01,   # goat win reward reduced slightly per turn

    # Shaping magnitudes  
    "REWARD_BLOCK_TIGER":       0.08,   # reward for reducing tiger mobility
    "REWARD_BUBBLE_SPACE":      0.02,   # reward for increasing goat “bubble” territory
    "REWARD_CLUSTER_TIGERS":    0.02,   # reward for spreading tigers apart
    "REWARD_CENTER_GOAT":       0.05,   # reward for goats occupying strong center nodes
    "REWARD_CENTER_TIGER":     -0.03,   # penalty for tigers holding center nodes

    # Special shaping
    "NEAR_LOCK_BONUS":          0.3,    # bonus when goats push tigers into near-lock states
    "LATE_GAME_START_TURN":     40,     # turn at which late-game shaping begins to scale
    "MOBILITY_BACKSLIDE_SCALE": 0.5,    # penalty multiplier when goats worsen tiger mobility

    # Anti-repeat penalty
    "REWARD_INVALID_SOFT":     -0.05,   # small penalty for soft invalid actions
    "REWARD_INVALID_HARD":     -1.0,    # large penalty for hard invalid actions (terminate)
    "REWARD_REPEAT_STATE":     -0.5,    # (Quadratic along count) penalty for repeating a previous board state
    "MAX_REPEATS":              6,      # how many repeats before a stall timeout triggers

    # Tiger AI difficulty
    "BASE_TIGER_CAPTURE_BIAS": 1.0,     # how greedy tigers are toward captures (1.0 = full greedy)
}


# ============================================================
#  REWARD Scaling – Term Weights
#     1.0 = keep as-is
#     0.0 = ablate (remove)
#     >1  = up-weight, <1 = down-weight
# ============================================================

DEFAULT_WEIGHTS = {
    # Per-step costs
    "step":          1.0,   # REWARD_STEP + move-phase step cost

    # Terminal outcomes
    "goat_win":      1.0,   # REWARD_GOAT_WIN (+ decay)
    "tiger_win":     1.0,   # REWARD_TIGER_WIN
    "goat_timeout":  1.0,   # GOAT_TIMEOUT_SCALE * REWARD_TIGER_WIN
    "goat_stall":    1.0,   # GOAT_STALL_SCALE   * REWARD_TIGER_WIN

    # Shaping / intermediate signals
    "goat_eaten":    1.0,   # REWARD_GOAT_EATEN
    "block_tiger":   1.0,   # REWARD_BLOCK_TIGER (+ backslide)
    "near_lock":     1.0,   # NEAR_LOCK_BONUS
    "bubble":        1.0,   # REWARD_BUBBLE_SPACE
    "cluster":       1.0,   # REWARD_CLUSTER_TIGERS
    "center":        1.0,   # REWARD_CENTER_GOAT / REWARD_CENTER_TIGER

    # Penalties
    "repeat_state":  1.0,   # REWARD_REPEAT_STATE
    "invalid_soft":  1.0,   # REWARD_INVALID_SOFT
    "invalid_hard":  1.0,   # REWARD_INVALID_HARD

}

# ============================================================
#  BASE ENVIRONMENT
# ============================================================

class TnGEnv(FalconEnvCore, gym.Env):
    """
    Tigers and Goats Environment.

    Observation (Box(25,)):
      - 23 board cells: 0=empty, 1=goat, 2=tiger
      - 1 value: goats eaten (0..6)
      - 1 value: phase (0=placing, 1=moving)

    Action (MultiDiscrete([23, 5])):
      - position index (0..22)
      - direction code (0..4), interpreted via move_map
    """

    def __init__(self, reward_weights=None):
        super().__init__()

        # Knob weights
        self.knobs = dict(DEFAULT_KNOBS)
        # Reward weights
        self.reward_weights = dict(DEFAULT_WEIGHTS)

        # Load any tuning values passed to this function into
        # their respective dictionaries.
        if reward_weights is not None:
            for key, val in reward_weights.items():
                if key in self.knobs:
                    self.knobs[key] = val
                elif key in self.reward_weights:
                    self.reward_weights[key] = val
                else:
                    print(f"[warning] Unknown tuning key: {key}")
            

        # -------------------------
        # Observation space
        # -------------------------
        self.observation_space = spaces.Box(
            low=np.array([0] * BOARD_SIZE + [0] + [0], dtype=np.int8),
            high=np.array([2] * BOARD_SIZE + [6] + [1], dtype=np.int8),
            shape=(BOARD_SIZE + 2,),
            dtype=np.int8,
        )

        # -------------------------
        # Action space
        # -------------------------
        # Internally we use (pos, dir_code)
        self.action_space = spaces.MultiDiscrete([BOARD_SIZE, DIR_CODES])

        # Core board state + trackers (from FalconEnvCore)
        self._init_board_state()

        # Cached geometry for shaping (FalconEnvCore helpers)
        tiger_opts = self._tiger_moves()
        self.prev_tiger_moves = len(tiger_opts)
        self.prev_unreachable = self._compute_unreachable_safe_cells()
        self.prev_tiger_spread = self._tiger_spread()
    # end def __init__()

    # --------------------------------------------------------
    #  Weight/knob dictionary helper
    # --------------------------------------------------------
    def _w(self, key) -> float:
        """
        Unified accessor:
            -If key is a reward-term weight -> return weight
            -If key is a constant knob -> return constant
            -If unknown -> throw an error
        """
        if key in self.reward_weights:
            return self.reward_weights[key]
        if key in self.knobs:
            return self.knobs[key]
        
        raise KeyError(
            f"[TnGEnv] Unknown tuning key '{key}'. "
            f"Valid weight keys: {list(self.reward_weights.keys())}. "
            f"Valid knob keys: {list(self.knobs.keys())}."
        )

    # --------------------------------------------------------
    #  Gym API: reset
    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Reset board, game state, and valid moves."""
        super().reset(seed=seed)

        self._init_board_state()  # core: board pieces + valid_moves

        # initial tiger mobility + geometry metrics
        tiger_opts = self._tiger_moves()  # core: tiger move generator
        self.prev_tiger_moves = len(tiger_opts)
        self.prev_unreachable = self._compute_unreachable_safe_cells()  # core: bubble metric
        self.prev_tiger_spread = self._tiger_spread()  # core: tiger spread metric

        obs = self.get_state()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    # --------------------------------------------------------
    #  State helpers
    # --------------------------------------------------------
    def get_state(self):
        """
        Flattened state: [board(23), eaten(1), phase(1)] as int8.
        """
        return np.concatenate(
            (self.board, [self.eaten, self.phase])
        ).astype(np.int8)

    def render(self, show_coords=True):
        """Render using the shared board renderer (from FalconEnvcore)."""
        self.render_board(
            show_coords=show_coords,
            last_move_desc=getattr(self, "last_move_desc", ""),
        )

    # --------------------------------------------------------
    #  Gym API: step
    # --------------------------------------------------------
    def step(self, action):
        """
        Step the environment by one combined Goat+Tiger turn:
          1) Goat action (place or move, depending on phase)
          2) Check goat win (no tiger moves)
          3) Tiger greedy move (captures if possible)
          4) Check tiger win (enough goats eaten)
          5) Update goat legal moves
        """
        # If game is already over, just return terminal state again.
        if self.terminate:
            return self.get_state(), 0.0, True, False, {
                "action_mask": self.get_action_mask()
            }
        
        self.turns += 1

        pos, dir_code = self._decode_action(action)

        # soft invalid: out of bounds
        if not (0 <= pos < BOARD_SIZE) or not (0 <= dir_code < DIR_CODES):
            return (
                self.get_state(), 
                self._w("invalid_soft") * self._w("REWARD_INVALID_SOFT"), 
                False, 
                False,
                {
                "invalid_action": True,
                "action_mask": self.get_action_mask(),
                }
            )
        
        # metrics before the goat move
        tiger_moves_before   = self.prev_tiger_moves
        bubble_before        = self.prev_unreachable
        spread_before        = self.prev_tiger_spread
        goats_eaten_before   = self.eaten

        # -------------------------
        # Base per-step penalty
        # -------------------------
        if self.phase == 0:
            # placing phase: keep it cheap so goats can build a structure
            shaped_reward = self._w("step") * self._w("REWARD_STEP")
        else:
            # moving phase: ramp up per-step penalty as move_steps increases
            step_pen = (
                self._w("MOVE_STEP_BASE") 
                + self._w("MOVE_STEP_SLOPE") 
                * self.move_steps)
            # clamp so it doesn't go beyond MOVE_STEP_MIN (more negative)
            step_pen = max(step_pen, self._w("MOVE_STEP_MIN"))
            shaped_reward = self._w("step") * step_pen

        # -------------------------
        # Goat turn
        # -------------------------
        if self.phase == 0:
            # Placing phase: must place on empty cell with dir_code = 0
            if dir_code != 0 or self.board[pos] != 0:
                if DEBUG_INVALID:
                    print("how did you manage this error?", action)
                # treat as soft invalid, do not advance game
                return (
                    self.get_state(), 
                    self._w("invalid_soft") * self._w("REWARD_INVALID_SOFT"),
                    False, 
                    False, {
                    "invalid_action": True,
                    "action_mask": self.get_action_mask(),
                    }
                )

            # Place goat
            self.board[pos] = 1
            self.goats_placed += 1

            # Switch to moving phase after all goats are placed
            if self.goats_placed >= TOTAL_GOATS_TO_PLACE:
                self.phase = 1

        else:
            # Moving phase: move an existing goat according to move_map
            dest = self.move_map[pos].get(dir_code, None)

            # invalid move: wrong piece, invalid dir, or dest not empty
            if dest is None or self.board[pos] != 1 or self.board[dest] != 0:
                if DEBUG_INVALID:
                    print("this error also shouldnt happen", action)
                self.terminate = True
                return (
                    self.get_state(),
                    self._w("invalid_hard") * self._w("REWARD_INVALID_HARD"),
                    True, 
                    False, {
                    "error": "Invalid move",
                    "action_mask": self.get_action_mask(),
                    }
                )

            # Perform goat move
            self.board[pos] = 0
            self.board[dest] = 1

        if self.phase == 1:
            self.move_steps += 1

        # -------------------------
        # After goat move: recompute geometry and tiger moves
        # -------------------------
        tiger_options = self._tiger_moves()
        tiger_moves_after = len(tiger_options)
    
        # update cached mobility for next step
        self.prev_tiger_moves = tiger_moves_after

        # Bonus for "near-lock" states (tigers almost immobilized)
        if 0 < tiger_moves_after <= 2:
            shaped_reward += self._w("near_lock") * self._w("NEAR_LOCK_BONUS")

        late = min(
            max(self.turns - self._w("LATE_GAME_START_TURN"), 0)
            / (self._w("MAX_TURNS") - self._w("LATE_GAME_START_TURN")),
            1.0,
        )

        # 1) Mobility shaping: reward reducing tiger moves
        # delta moves is positive if tiger is less mobile
        delta_moves = tiger_moves_before - tiger_moves_after
        # if delta moves is positive, then positive reward
        if delta_moves > 0:
            shaped_reward += (
                self._w("block_tiger")
                * self._w("REWARD_BLOCK_TIGER")
                * delta_moves
                * (1.0 + late)
            )
        # if delta moves is negative, then negative reward
        elif delta_moves < 0:
            # Penalize making tigers more mobile
            shaped_reward += (
                self._w("block_tiger")
                * self._w("REWARD_BLOCK_TIGER")
                * self._w("MOBILITY_BACKSLIDE_SCALE")
                * delta_moves
            )
    
        # 2) Bubble shaping: reward more unreachable safe cells
        bubble_after = self._compute_unreachable_safe_cells()
        self.prev_unreachable = bubble_after
        delta_bubble = bubble_after - bubble_before
        if delta_bubble > 0:
            shaped_reward += (
                self._w("bubble")
                * self._w("REWARD_BUBBLE_SPACE")
                * delta_bubble
                * (1.0 + late)
            )
    
        # 3) Tiger clustering: reward decreased spread
        spread_after = self._tiger_spread()
        self.prev_tiger_spread = spread_after
        delta_spread = spread_before - spread_after  # positive if more clustered
        if delta_spread > 0:
            shaped_reward += (
                self._w("cluster")
                * self._w("REWARD_CLUSTER_TIGERS")
                * delta_spread
                * (1.0 + late)
            )
    
        # 4) Center control: reward goats on centers, penalize tigers
        center_score = 0.0
        for idx in KEY_CENTERS:
            if self.board[idx] == 1:
                center_score += self._w("REWARD_CENTER_GOAT")
            elif self.board[idx] == 2:
                center_score += self._w("REWARD_CENTER_TIGER")
        shaped_reward += self._w("center") * center_score
    
        # 5) Goat win: no tiger moves
        if tiger_moves_after == 0:
            self.terminate = True
            shaped_reward = (
                self._w("goat_win") * self._w("REWARD_GOAT_WIN")
                - self._w("GOAT_WIN_TURN_DECAY") * self.turns
            )
            return self.get_state(), shaped_reward, True, False, {
                "winner": "Goat",
                "action_mask": self.get_action_mask(),
            }

        # -------------------------
        # Tiger turn (biased capturing logic)
        # -------------------------
        captures = [m for m in tiger_options if m[2]]  # (from, to, is_capture)

        t_from = t_to = None
        took_capture = False

        capture_bias = self._w("BASE_TIGER_CAPTURE_BIAS")

        if captures and random.random() < capture_bias:
            # With probability TIGER_CAPTURE_BIAS, prefer a capture
            t_from, t_to, _ = random.choice(captures)
            took_capture = True

        elif tiger_options:
            # Otherwise, choose a random legal move (may or may not be a capture)
            t_from, t_to, took_capture = random.choice(tiger_options)

        else:
            # No tiger moves means goat wins (redundant safety)
            self.terminate = True
            shaped_reward += self._w("goat_win") * self._w("REWARD_GOAT_WIN")
            return self.get_state(), shaped_reward, True, False, {
                "winner": "Goat",
                "action_mask": self.get_action_mask(),
            }

        # If this move is a capture, remove the jumped goat
        if took_capture:
            for d_code, neigh in self.move_map[t_from].items():
                jump = self.move_map.get(neigh, {}).get(d_code, None)
                if t_from == 0:
                    # special-case b0 jump direction
                    jump = self.move_map.get(neigh, {}).get(3, None)

                if jump == t_to and self.board[neigh] == 1:
                    self.board[neigh] = 0
                    self.eaten += 1
                    break

        # Apply tiger move
        self.board[t_to] = 2
        self.board[t_from] = 0

        # goats eaten this turn
        goats_eaten_this_turn = self.eaten - goats_eaten_before
        if goats_eaten_this_turn > 0:
            shaped_reward += (
                self._w("goat_eaten")
                * self._w("REWARD_GOAT_EATEN")
                * goats_eaten_this_turn
            )

        # -------------------------
        # Check tiger win condition
        # -------------------------
        if self.eaten >= GOATS_EATEN_FOR_TIGER_WIN:
            self.terminate = True
            shaped_reward += self._w("tiger_win") * self._w("REWARD_TIGER_WIN")
            return self.get_state(), shaped_reward, True, False, {
                "winner": "Tiger",
                "action_mask": self.get_action_mask(),
            }
        
        # -------------------------
        # Max turn limit
        # -------------------------
        if self.turns >= self._w("MAX_TURNS"):
            self.terminate = True
            shaped_reward += (
                self._w("goat_timeout")
                * self._w("REWARD_TIGER_WIN")
                * self._w("GOAT_TIMEOUT_SCALE")
            )
            return self.get_state(), shaped_reward, True, False, {
                "winner": "GoatTimeout",
                "action_mask": self.get_action_mask(),
            }
        
        # -------------------------
        # Repetition penalty (simple board hash, escalating)
        # -------------------------
        # Use bytes for the board to avoid list/tuple allocations
        board_hash = (self.board.tobytes(), self.eaten, self.phase)

        prev_count = self.state_history.get(board_hash, 0)
        new_count = prev_count + 1
        self.state_history[board_hash] = new_count

        # If a state has been repeated too many times, treat it as a stall timeout
        if new_count >= self._w("MAX_REPEATS"):
            self.terminate = True
            # Treat as a "bad" timeout: goats failed to make progress
            shaped_reward += (
                self._w("goat_stall")
                * self._w("REWARD_TIGER_WIN")
                * self._w("GOAT_STALL_SCALE")
            )
            return self.get_state(), shaped_reward, True, False, {
                "winner": "StallTimeout",
                "action_mask": self.get_action_mask(),
            }

        if prev_count > 0:
            # Quadratic growth in repeat count
            repeat_pen = self._w("REWARD_REPEAT_STATE") * (prev_count ** 2)
            shaped_reward += self._w("repeat_state") * repeat_pen

        # -------------------------
        # Update goat legal moves
        # -------------------------
        self._update_valid_moves()

        # Non-terminal step with shaped reward
        return self.get_state(), shaped_reward, False, False, {
            "action_mask": self.get_action_mask()
        }

# ============================================================
#  WRAPPER: Flatten MultiDiscrete action → Discrete(115)
# ============================================================

class FlattenTnGActionWrapper(gym.Wrapper):
    """
    Wraps TnGEnv so that the external action space is Discrete(115),
    but internally the base env still uses MultiDiscrete([23, 5]).
    """

    def __init__(self, env):
        super().__init__(env)
        # 23 positions × 5 dir_codes
        self.action_space = spaces.Discrete(BOARD_SIZE * DIR_CODES)
        # Observation space is unchanged
        self.observation_space = env.observation_space

    def get_action_mask(self):
        # delegate to the base env (TnGEnv)
        return self.env.get_action_mask()  # type: ignore[attr-defined]

    def step(self, action):
        """
        Convert flat Discrete action back to (pos, dir_code)
        and pass to underlying MultiDiscrete env.
        """
        flat = int(action)
        pos = flat // DIR_CODES
        dir_code = flat % DIR_CODES

        md_action = np.array([pos, dir_code], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(md_action)

        # info["action_mask"] is provided by base env
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
