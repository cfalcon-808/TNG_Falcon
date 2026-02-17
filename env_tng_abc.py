# ============================================================
#  Project    : Tigers & Goats - Falcon Branch
#  Module     : Maskable PPO Environment (Full Game, Unified Tigers)
#  File       : env_tng_abc.py
#  Version    : env6.0
#  Last Update: 2026-02-15
#
#  Purpose / Goal:
#    Provide a stable Gymnasium environment for training robust goat/tiger
#    policies with a unified Discrete(115) action interface and strict
#    step/info semantics.
#
#  Overview:
#    - Full game rules: placing phase + moving phase
#    - Native Discrete(115) action space (no flatten wrapper)
#    - Action masking for valid actions (MaskablePPO-compatible)
#    - Unified tiger opponent selection (greedy / smart)
#    - Reward hook via self.reward_fn (default: sparse_reward)
#    - Consistent terminated vs truncated handling + strict info schema
#    - Repetition tracking and timeout handling
#
#  Workflow:
#    1) Create env (choose tiger_ai / learner_role / reward overrides)
#    2) Train/evaluate directly with Discrete actions + env masks
#    3) Use probabilistic opponent sampling in training wrappers if desired
#    4) Compare robustness across tiger styles during evaluation
#
#  Quick Use:
#    env = TnGEnv()                                      # goat learner vs greedy tiger
#    env = TnGEnv(tiger_ai=TIGER_AI_SMART)              # goat learner vs smart tiger
#    env = TnGEnv(learner_role=TIGER_LEARNER)           # tiger learner mode
#    env = TnGEnv(max_turns=150)                        # explicit turn truncation limit
#    env = TnGEnv(reward_weights={"bubble": 0.0})       # ablate shaping term
#
#  Compatibility:
#    - Designed for MaskablePPO training loops
#    - Exposes native Discrete(115) action space
#    - Intended integration points:
#        - train_falcon.py
#        - eval_falcon.py
#        - tng_GUI_falcon.py
# ============================================================
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from collections import deque


# ============================================================
#  GLOBAL CONSTANTS – Game Rules
# ============================================================

BOARD_SIZE                = 23          # board positions (0–22)
DIR_CODES                 = 5           # direction codes (0–4)
TOTAL_GOATS_TO_PLACE      = 15          # goats placed before moving phase begins
GOATS_EATEN_FOR_TIGER_WIN = 6           # tiger wins after this many goats eaten
TIGER_START_POSITIONS     = [0, 3, 4]   # starting indices for tigers
KEY_CENTERS               = [0, 9, 10, 15, 16] # top hub + center spots to control

# Debug flag to avoid slow prints in training
DEBUG_INVALID = False

TIGER_AI_GREEDY = "greedy"
TIGER_AI_SMART = "smart"
VALID_TIGER_AI = {TIGER_AI_GREEDY, TIGER_AI_SMART}
GOAT_LEARNER = "goat"
TIGER_LEARNER = "tiger"
GOAT_AI_RANDOM = "random"
GOAT_AI_MODEL = "model"


# ============================================================
#  GLOBAL TUNING KNOBS
#  All numeric values (constants + shaping magnitudes + scales)
#  ***Note: these can be changed in the training file***
# ============================================================

DEFAULT_KNOBS = {

    # Goat-learner terminal rewards
    "REWARD_GOAT_WIN":          3.2,    # reward when goats immobilize all tigers
    "REWARD_TIGER_WIN":        -3.2,    # penalty when tigers win by eating goats

    # Tiger-learner rewards (positive magnitudes)
    "REWARD_TIGER_CAPTURE":     0.35,   # reward per goat captured (tiger learner)
    "REWARD_TIGER_WIN_BONUS":   3.2,    # reward for tiger win (tiger learner)
    "REWARD_TIGER_LOSS_PENALTY": 3.2,   # penalty when tiger loses or times out (tiger learner)

    # Goat-learner step & capture shaping
    "REWARD_STEP":             -0.001,  # per-step penalty to discourage stalling
    "REWARD_GOAT_EATEN":       -0.35,   # penalty for each goat captured by tigers

    # Timeout scales
    "MAX_TIMEOUT_SCALE":        1.0,    # scale for turn-limit (max turns) timeout penalty
    "REPEAT_STALL_SCALE":       0.8,    # scale for stall-timeout (repeat-state) penalty
    "MAX_TURNS":                100,    # max turns before enforcing max-timeout

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
    "max_timeout":   1.0,   # MAX_TIMEOUT_SCALE * REWARD_TIGER_WIN
    "repeat_stall":  1.0,   # REPEAT_STALL_SCALE * REWARD_TIGER_WIN

    # Tiger-learner rewards
    "tiger_capture": 1.0,   # REWARD_TIGER_CAPTURE
    "tiger_win_bonus": 1.0, # REWARD_TIGER_WIN_BONUS
    "tiger_loss":    1.0,   # REWARD_TIGER_LOSS_PENALTY

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
#  BOARD COORDINATES – Human-Readable Labels
# ============================================================

COORD_LABELS = [
    "b0",        # 0
    "a1", "b1", "c1", "d1", "e1", "f1",      # 1–6
    "a2", "b2", "c2", "d2", "e2", "f2",      # 7–12
    "a3", "b3", "c3", "d3", "e3", "f3",      # 13–18
    "b4", "c4", "d4", "e4",                  # 19–22
]

# Special hub capture jumps from node 0 (b0)
HUB0_CAPTURE_JUMP = {
    1: 8,   # 0 -> 2 (b1) -> 8 (b2)
    2: 9,   # 0 -> 3 (c1) -> 9 (c2)
    3: 10,  # 0 -> 4 (d1) -> 10 (d2)
    4: 11,  # 0 -> 5 (e1) -> 11 (e2)
}


# ============================================================
#  Board index to coordinate helper
# ============================================================

def idx_to_coord(idx: int) -> str:
    """Safely convert a board index to its coordinate label."""
    if 0 <= idx < len(COORD_LABELS):
        return COORD_LABELS[idx]
    return str(idx)


# ============================================================
#  TIGER AI ƒ?" smart_tiger (shared across modes)
# ============================================================

def smart_tiger(board, tiger_moves, move_map, debug: bool = False):
    """
    Selects a tiger move based on:
      1) Capture logic
      2) Anchors: 0 and {9 or 10} as 2 conceptual anchors
      3) Roaming (high-value squares)
      4) Fallback random

    tiger_moves: list of (from_pos, to_pos, is_capture)
    """

    POS_VALS = {
        0: 8, 1: 4, 2: 6, 3: 7, 4: 7, 5: 6, 6: 4, 7: 4, 8: 7, 9: 8, 10: 8,
        11: 7, 12: 4, 13: 4, 14: 6, 15: 7, 16: 7, 17: 6, 18: 4, 19: 4,
        20: 5, 21: 5, 22: 4
    }

    def debug_print(*args):
        if debug:
            print("[TIGER DEBUG]", *args)

    def bfs_find_next_step(start_pos, target_positions):
        """
        Find the first move on a shortest path from start_pos to any target.
        Returns (first_step, distance). Returns (None, None) if unreachable.
        """

        # Start already on a target
        if start_pos in target_positions:
            return (None, 0)

        queue = []
        visited = {start_pos}

        # Explore immediate neighbors first
        for _, neighbor in move_map[start_pos].items():
            if neighbor is None:
                continue

            # Walkable if empty or is a target
            is_walkable = (board[neighbor] == 0) or (neighbor in target_positions)

            if is_walkable:
                # Direct hit: one-step path
                if neighbor in target_positions:
                    return (neighbor, 1)

                # Store (current, first_step, distance)
                queue.append((neighbor, neighbor, 1))
                visited.add(neighbor)

        # BFS for multi-step paths
        while queue:
            curr, first_step, dist = queue.pop(0)

            # Found a target
            if curr in target_positions:
                return (first_step, dist)

            # Explore neighbors of current position
            for _, neighbor in move_map[curr].items():
                if neighbor is None or neighbor in visited:
                    continue

                is_walkable = (board[neighbor] == 0) or (neighbor in target_positions)

                if is_walkable:
                    visited.add(neighbor)
                    queue.append((neighbor, first_step, dist + 1))

        # No path to any target
        return (None, None)

    # ====== MAIN LOGIC ======
    if not tiger_moves:
        debug_print("No legal tiger moves.")
        return None

    debug_print("Board:", board)
    debug_print("Legal tiger moves:", tiger_moves)

    # --- 1. CAPTURE LOGIC ---
    captures = [m for m in tiger_moves if m[2]]
    if captures:
        random.shuffle(captures)
        best_capture = max(captures, key=lambda x: POS_VALS[x[1]])
        debug_print(
            "Capture phase:",
            "candidate captures =", captures,
            "chosen =", best_capture,
            "dest_pos_val =", POS_VALS[best_capture[1]]
        )
        return best_capture

    # --- 2. ANCHOR LOGIC (0 and {9,10}) ---
    current_tiger_positions = [i for i in range(23) if board[i] == 2]

    anchor_0_occupied = (board[0] == 2)
    anchor_primary_occupied = (board[9] == 2 or board[10] == 2)

    need_anchor0 = not anchor_0_occupied
    need_primary = not anchor_primary_occupied

    debug_print(
        "Anchor status:",
        f"anchor_0_occupied={anchor_0_occupied}, "
        f"anchor_primary_occupied={anchor_primary_occupied}, "
        f"need_anchor0={need_anchor0}, need_primary={need_primary}"
    )

    if need_anchor0 or need_primary:
        anchor_groups = []

        if need_anchor0:
            anchor_groups.append(("secondary", [0]))

        if need_primary:
            primary_targets = [p for p in (9, 10) if board[p] == 0]
            if not primary_targets:
                primary_targets = [9, 10]
            anchor_groups.append(("primary", primary_targets))

        debug_print("Anchor groups:", anchor_groups)
        best = None  # (dist, from_pos, first_step, group_name, targets)

        for t_pos in current_tiger_positions:
            if anchor_0_occupied and t_pos == 0:
                debug_print("Skipping tiger at 0 (already satisfying anchor 0).")
                continue
            if anchor_primary_occupied and t_pos in (9, 10):
                debug_print(f"Skipping tiger at {t_pos} (already satisfying primary anchor).")
                continue

            for group_name, targets in anchor_groups:
                step, dist = bfs_find_next_step(t_pos, targets)
                if step is None:
                    continue

                debug_print(
                    f"Anchor candidate: tiger at {t_pos} -> step {step} "
                    f"(group {group_name}, targets {targets}, dist {dist})"
                )

                if (best is None) or (dist < best[0]):
                    best = (dist, t_pos, step, group_name, targets)

        if best is not None:
            dist, from_pos, step, group_name, targets = best
            chosen_move = None
            for move in tiger_moves:
                if move[0] == from_pos and move[1] == step:
                    chosen_move = move
                    break

            if chosen_move is not None:
                debug_print(
                    "Anchor choice:",
                    f"group={group_name}, targets={targets}, dist={dist},",
                    f"from={from_pos}, to={step}, move={chosen_move}"
                )
                return chosen_move

            debug_print(
                "Anchor choice failed: BFS suggested step not in tiger_moves.",
                f"group={group_name}, targets={targets}, dist={dist},",
                f"from={from_pos}, to={step}"
            )
        else:
            debug_print("Anchor logic: no reachable anchor path found, falling through.")

    # --- 3. ROAMING LOGIC ---
    high_value_moves = []
    for move in tiger_moves:
        t_from, t_to, _ = move

        if t_from == 0:
            continue

        if (t_from == 9 or t_from == 10) and anchor_primary_occupied:
            if board[9] == 2 and board[10] != 2 and t_from == 9:
                continue
            if board[10] == 2 and board[9] != 2 and t_from == 10:
                continue

        if POS_VALS[t_to] >= 5:
            high_value_moves.append(move)

    if high_value_moves:
        chosen = random.choice(high_value_moves)
        debug_print(
            "Roaming phase:",
            "high_value_moves =", high_value_moves,
            "chosen =", chosen,
            "dest_pos_val =", POS_VALS[chosen[1]]
        )
        return chosen

    # --- 4. FALLBACK ---
    fallback = random.choice(tiger_moves)
    debug_print(
        "Fallback phase: no captures, no anchor moves, no high-value roam.",
        "Fallback chosen move =", fallback
    )
    return fallback

# ============================================================
#  BASE ENVIRONMENT
# ============================================================

class TnGEnv(gym.Env):
    """
    Tigers and Goats Environment.

    Observation (Box(25,)):
      - 23 board cells: 0=empty, 1=goat, 2=tiger
      - 1 value: goats eaten (0..6)
      - 1 value: phase (0=placing, 1=moving)

    Action (Discrete(115)):
      - flat action index in [0, 114]
      - decode rule: pos = action // DIR_CODES, dir_code = action % DIR_CODES
    """

    def get_action_mask(self):
        """
        Returns a Boolean mask of shape (BOARD_SIZE * DIR_CODES,),
        where each index corresponds to (pos, dir_code) = (i // DIR_CODES, i % DIR_CODES).

        - Goat learner: uses self.valid_moves in transition-safe form:
            - pair: [pos, dir_code] or (pos, dir_code)
            - flat: int action id
        - Tiger learner: uses _tiger_moves(include_dir=True) to mark legal tiger actions.
        """
        mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)

        if getattr(self, "learner_role", GOAT_LEARNER) == TIGER_LEARNER:
            for from_pos, _to_pos, _cap, dir_code in self._tiger_moves(include_dir=True):
                flat = from_pos * DIR_CODES + dir_code
                if 0 <= flat < mask.size:
                    mask[flat] = True
            return mask

        for mv in self.valid_moves:
            if np.isscalar(mv):
                flat = int(mv)
            else:
                pos, dir_code = mv
                flat = self.encode_action(pos, dir_code)
            if 0 <= flat < mask.size:
                mask[flat] = True

        return mask


    def __init__(
            self, 
            reward_weights=None, 
            tiger_ai: str = TIGER_AI_GREEDY, 
            learner_role: str = GOAT_LEARNER, 
            goat_opponent_ai: str = GOAT_AI_RANDOM, 
            goat_model_predict_fn=None, 
            reward_fn=None,
            max_turns: int | None = None,
            ):
        super(TnGEnv, self).__init__()

        # Knob weights
        self.knobs = dict(DEFAULT_KNOBS)
        # Reward weights
        self.reward_weights = dict(DEFAULT_WEIGHTS)

        self.learner_role = (learner_role or GOAT_LEARNER).lower()
        self.goat_opponent_ai = (goat_opponent_ai or GOAT_AI_RANDOM).lower()
        self._goat_model_predict_fn = goat_model_predict_fn  # callable(obs, mask)->flat action
        self.reward_fn = reward_fn or self.sparse_reward

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

        configured_max_turns = self.knobs["MAX_TURNS"] if max_turns is None else max_turns
        self.max_turns = int(configured_max_turns)
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got: {configured_max_turns}")
        # Keep _w("MAX_TURNS") and explicit max_turns aligned.
        self.knobs["MAX_TURNS"] = self.max_turns

        self.tiger_ai = (tiger_ai or TIGER_AI_GREEDY).lower()
        if self.tiger_ai not in VALID_TIGER_AI:
            raise ValueError(f"tiger_ai must be one of {VALID_TIGER_AI}")
        if self.learner_role not in {GOAT_LEARNER, TIGER_LEARNER}:
            raise ValueError(f"learner_role must be one of: {GOAT_LEARNER}, {TIGER_LEARNER}")
        if self.goat_opponent_ai not in {GOAT_AI_RANDOM, GOAT_AI_MODEL}:
            raise ValueError(f"goat_opponent_ai must be one of: {GOAT_AI_RANDOM}, {GOAT_AI_MODEL}")

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
        # Action space (flat)
        # -------------------------
        self.action_space = spaces.Discrete(BOARD_SIZE * DIR_CODES)

        # Game state variables
        self.terminate = False
        self.board = np.zeros(BOARD_SIZE, dtype=np.int8)
        for idx in TIGER_START_POSITIONS:
            self.board[idx] = 2

        self.eaten = 0          # goats eaten by tigers
        self.phase = 0          # 0 = placing phase, 1 = moving phase
        self.goats_placed = 0

        # For reward shaping
        self.prev_tiger_moves = 0
        self.prev_unreachable = 0
        self.prev_tiger_spread = 0.0

        # Repetition tracker
        self.turns = 0
        self.move_steps = 0     # Counts moves taken in move phase
        self.state_history = {}

        # For human readable move tracking in render()\
        self.last_move_desc = ""

        # -------------------------
        # Move map:
        # pos -> {dir_code: destination_pos}
        # -------------------------
        self.move_map = {
            0: {1: 2, 2: 3, 3: 4, 4: 5},  # b0 special

            1: {2: 2, 3: 7},                 # a1
            2: {1: 0, 2: 3, 3: 8, 4: 1},     # b1
            3: {1: 0, 2: 4, 3: 9, 4: 2},     # c1
            4: {1: 0, 2: 5, 3: 10, 4: 3},    # d1
            5: {1: 0, 2: 6, 3: 11, 4: 4},    # e1
            6: {3: 12, 4: 5},                # f1

            7: {1: 1, 2: 8, 3: 13},          # a2
            8: {1: 2, 2: 9, 3: 14, 4: 7},    # b2
            9: {1: 3, 2: 10, 3: 15, 4: 8},   # c2
            10: {1: 4, 2: 11, 3: 16, 4: 9},  # d2
            11: {1: 5, 2: 12, 3: 17, 4: 10}, # e2
            12: {1: 6, 3: 18, 4: 11},        # f2

            13: {1: 7, 2: 14},               # a3
            14: {1: 8, 2: 15, 3: 19, 4: 13}, # b3
            15: {1: 9, 2: 16, 3: 20, 4: 14}, # c3
            16: {1: 10, 2: 17, 3: 21, 4: 15},# d3
            17: {1: 11, 2: 18, 3: 22, 4: 16},# e3
            18: {1: 12, 4: 17},              # f3

            19: {1: 14, 2: 20},              # b4
            20: {1: 15, 2: 21, 4: 19},       # c4
            21: {1: 16, 2: 22, 4: 20},       # d4
            22: {1: 17, 4: 21},              # e4
        }

        # Precompute all-pairs shortest path lengths on the static board graph
        self._dist_matrix = self._precompute_all_pairs_dist()

        # Pre-allocated visited buffer for bubble BFS
        self._visited_buffer = np.zeros(BOARD_SIZE, dtype=bool)

        # Initial valid moves
        self.valid_moves = []
        self._update_valid_moves()
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

        # Normalize + validate tiger_ai (important if wrappers modify it between episodes)
        self.tiger_ai = (self.tiger_ai or TIGER_AI_GREEDY).lower()
        if self.tiger_ai not in VALID_TIGER_AI:
            raise ValueError(f"tiger_ai must be one of {VALID_TIGER_AI}")

        # Snapshot tiger used for THIS episode (critical for mixed training)
        self._episode_tiger_ai = self.tiger_ai

        self.board[:] = 0
        for idx in TIGER_START_POSITIONS:
            self.board[idx] = 2

        self.eaten = 0
        self.phase = 0
        self.goats_placed = 0
        self.terminate = False

        self.turns = 0
        self.move_steps = 0
        self.state_history = {}

        # clear last move description
        self.last_move_desc = ""

        # initial tiger mobility + geometry metrics
        tiger_opts = self._tiger_moves()
        self.prev_tiger_moves = len(tiger_opts)
        self.prev_unreachable = self._compute_unreachable_safe_cells()
        self.prev_tiger_spread = self._tiger_spread()

        self._update_valid_moves()

        obs = self.get_state()
        info = {
            "action_mask": self.get_action_mask(),
            "tiger_ai": self.tiger_ai,
            "tiger_ai_id": float(1 if self.tiger_ai == TIGER_AI_SMART else 0),
            "episode_tiger_ai": self._episode_tiger_ai,
            "episode_tiger_ai_id": float(1 if self._episode_tiger_ai == TIGER_AI_SMART else 0)
            }
        return obs, info

    # --------------------------------------------------------
    #  State + Rendering helpers
    # --------------------------------------------------------
    def get_state(self):
        """
        Flattened state: [board(23), eaten(1), phase(1)] as int8.
        """
        return np.concatenate(
            (self.board, [self.eaten, self.phase])
        ).astype(np.int8)

    def render(self, show_coords=True):
        """
        Polished ASCII render for easier human analysis.
            - Empty cells shown as '0' instead of '.'
            - Cleaner alignment and spacing
        - Optional coordinate labels
        - Status bar showing game metrics
        """

        # Updated symbol map: ie. empty = 0,...
        symbols = {0: "0", 1: "G", 2: "T"}    
        cells = [symbols[v] for v in self.board]
        goats_eaten = self.eaten
        phase_str = "Place" if self.phase == 0 else "Move"
    
        # pull board coordinates from global COORD_LABELS
        coord = (
            [f" {COORD_LABELS[i]} " for i in range(BOARD_SIZE)] 
            if show_coords else [""] * BOARD_SIZE
        )
    
        # Build each visual line
        line0 = f"                  {cells[0]}{coord[0]}"
        line1 = (
            f"   {cells[1]}{coord[1]} {cells[2]}{coord[2]} "
            f"{cells[3]}{coord[3]} {cells[4]}{coord[4]} "
            f"{cells[5]}{coord[5]} {cells[6]}{coord[6]}"
        )
        line2 = (
            f"   {cells[7]}{coord[7]} {cells[8]}{coord[8]} "
            f"{cells[9]}{coord[9]} {cells[10]}{coord[10]} "
            f"{cells[11]}{coord[11]} {cells[12]}{coord[12]}"
        )
        line3 = (
            f"   {cells[13]}{coord[13]} {cells[14]}{coord[14]} "
            f"{cells[15]}{coord[15]} {cells[16]}{coord[16]} "
            f"{cells[17]}{coord[17]} {cells[18]}{coord[18]}"
        )
        line4 = (
            f"         {cells[19]}{coord[19]} {cells[20]}{coord[20]} "
            f"{cells[21]}{coord[21]} {cells[22]}{coord[22]}"
        )
    
        # Status panel
        status = (
            f"-----------------------------------------\n"
            f" Phase: {phase_str:<5} | Goats Eaten: {goats_eaten} | Turn: {self.turns}\n"
            f"-----------------------------------------"
        )
    
        # Last-move description
        if getattr(self, "last_move_desc",""):
            status += f"\n Last move: {self.last_move_desc}"

        print(
            "\n"
            "============== TIGERS & GOATS BOARD ==============\n"
            f"{line0}\n{line1}\n{line2}\n{line3}\n{line4}\n{status}\n"
        )
    # end def render()

    def encode_action(self, pos: int, dir_code: int) -> int:
        return int(pos) * DIR_CODES + int(dir_code)

    def decode_action(self, action):
        """
        Primary decode helper for Discrete(115) action ids.
        Also accepts [pos, dir] vectors for compatibility.
        """
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()

        if np.isscalar(action):
            a = int(action)
            return a // DIR_CODES, a % DIR_CODES

        if isinstance(action, (list, tuple, np.ndarray)):
            arr = np.asarray(action).reshape(-1)
            if arr.size == 2:
                return int(arr[0]), int(arr[1])

        raise ValueError(f"Unrecognized action format: {action} (type {type(action)})")

    def _apply_reward_fn(self, prev_obs, action, result):
        obs, reward, terminated, truncated, info = result
        try:
            final_reward = self.reward_fn(
                prev_obs,
                action,
                obs,
                bool(terminated),
                bool(truncated),
                info,
            )
        except TypeError as first_err:
            # Backward compatibility: older callbacks may still expect base_reward.
            try:
                final_reward = self.reward_fn(
                    prev_obs,
                    action,
                    obs,
                    float(reward),
                    bool(terminated),
                    bool(truncated),
                    info,
                )
            except TypeError:
                raise first_err
        return obs, float(final_reward), bool(terminated), bool(truncated), info

    # --------------------------------------------------------
    #  Action decoding helper
    # --------------------------------------------------------
    def _decode_action(self, action):
        return self.decode_action(action)

    def _action_to_flat(self, action):
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()
        if np.isscalar(action):
            return int(action)
        if isinstance(action, (list, tuple, np.ndarray)):
            arr = np.asarray(action).reshape(-1)
            if arr.size == 2:
                return self.encode_action(int(arr[0]), int(arr[1]))
        return None

    def _safe_decode_action(self, action):
        try:
            pos, dir_code = self._decode_action(action)
            return int(pos), int(dir_code)
        except Exception:
            return None

    def _build_step_info(self, **extra):
        info = {
            "reason": None,
            "winner": None,
            "goat_action": None,
            "goat_decoded": None,
            "tiger_move": None,
            "phase": int(self.phase),
            "goats_eaten": int(self.eaten),
            "goats_placed": int(self.goats_placed),
            "turn_counter": int(self.turns),
            "learner_role": self.learner_role,
            "action_mask": self.get_action_mask(),
            "tiger_ai": self.tiger_ai,
            "tiger_ai_id": float(1 if self.tiger_ai == TIGER_AI_SMART else 0),
            "episode_tiger_ai": self._episode_tiger_ai,
            "episode_tiger_ai_id": float(1 if self._episode_tiger_ai == TIGER_AI_SMART else 0),
            "goat_opponent_ai": self.goat_opponent_ai,
        }
        info.update(extra)
        return info

    def sparse_reward(self, prev_obs, action, obs, terminated, truncated, info):
        role = info.get("learner_role", self.learner_role)
        if role == TIGER_LEARNER:
            return self._sparse_reward_tiger(prev_obs, action, obs, terminated, truncated, info)
        return self._sparse_reward_goat(prev_obs, action, obs, terminated, truncated, info)

    def _sparse_reward_goat(self, prev_obs, action, obs, terminated, truncated, info):
        reason = info.get("reason")
        if reason == "already_terminated":
            return 0.0
        if reason == "invalid_soft":
            return self._w("invalid_soft") * self._w("REWARD_INVALID_SOFT")
        if reason == "invalid_hard":
            return self._w("invalid_hard") * self._w("REWARD_INVALID_HARD")
        if reason == "goat_win_no_tiger_moves":
            return (
                self._w("goat_win") * self._w("REWARD_GOAT_WIN")
                - self._w("GOAT_WIN_TURN_DECAY") * self.turns
            )

        shaped_reward = self._w("step") * float(info.get("step_penalty", self._w("REWARD_STEP")))
        if bool(info.get("near_lock", False)):
            shaped_reward += self._w("near_lock") * self._w("NEAR_LOCK_BONUS")

        late = float(info.get("late", 0.0))
        delta_moves = int(info.get("delta_moves", 0))
        if delta_moves > 0:
            shaped_reward += (
                abs(self._w("block_tiger") * self._w("REWARD_BLOCK_TIGER"))
                * delta_moves
                * (1.0 + late)
            )
        elif delta_moves < 0:
            shaped_reward += (
                abs(self._w("block_tiger") * self._w("REWARD_BLOCK_TIGER"))
                * self._w("MOBILITY_BACKSLIDE_SCALE")
                * delta_moves
            )

        delta_bubble = int(info.get("delta_bubble", 0))
        if delta_bubble > 0:
            shaped_reward += (
                self._w("bubble")
                * self._w("REWARD_BUBBLE_SPACE")
                * delta_bubble
                * (1.0 + late)
            )

        delta_spread = float(info.get("delta_spread", 0.0))
        if delta_spread > 0:
            shaped_reward += (
                self._w("cluster")
                * self._w("REWARD_CLUSTER_TIGERS")
                * delta_spread
                * (1.0 + late)
            )

        shaped_reward += self._w("center") * float(info.get("center_score", 0.0))

        goats_eaten_this_turn = int(info.get("goats_eaten_this_turn", 0))
        if goats_eaten_this_turn > 0:
            shaped_reward += (
                self._w("goat_eaten")
                * self._w("REWARD_GOAT_EATEN")
                * goats_eaten_this_turn
            )

        if reason == "tiger_win_capture_threshold":
            shaped_reward += self._w("tiger_win") * self._w("REWARD_TIGER_WIN")
        elif reason == "max_timeout":
            shaped_reward += (
                self._w("max_timeout")
                * self._w("REWARD_TIGER_WIN")
                * self._w("MAX_TIMEOUT_SCALE")
            )
        elif reason == "repeat_timeout":
            shaped_reward += (
                self._w("repeat_stall")
                * self._w("REWARD_TIGER_WIN")
                * self._w("REPEAT_STALL_SCALE")
            )
        else:
            prev_count = int(info.get("repeat_prev_count", 0))
            if prev_count > 0:
                repeat_pen = self._w("REWARD_REPEAT_STATE") * (prev_count ** 2)
                shaped_reward += self._w("repeat_state") * repeat_pen

        return shaped_reward

    def _sparse_reward_tiger(self, prev_obs, action, obs, terminated, truncated, info):
        reason = info.get("reason")
        if reason == "already_terminated":
            return 0.0
        if reason == "invalid_soft":
            return self._w("invalid_soft") * self._w("REWARD_INVALID_SOFT")
        if reason == "invalid_hard":
            return self._w("invalid_hard") * abs(self._w("REWARD_INVALID_HARD"))

        shaped_reward = self._w("step") * float(info.get("step_penalty", self._w("REWARD_STEP")))

        goats_eaten_this_turn = int(info.get("goats_eaten_this_turn", 0))
        if goats_eaten_this_turn > 0:
            shaped_reward += (
                self._w("tiger_capture")
                * self._w("REWARD_TIGER_CAPTURE")
                * goats_eaten_this_turn
            )

        delta_moves = int(info.get("delta_moves", 0))
        if delta_moves != 0:
            shaped_reward += abs(self._w("block_tiger") * self._w("REWARD_BLOCK_TIGER")) * delta_moves

        shaped_reward += self._w("center") * float(info.get("center_score", 0.0))

        if reason in {"tiger_win_capture_threshold", "tiger_win_no_goat_moves"}:
            shaped_reward += self._w("tiger_win_bonus") * self._w("REWARD_TIGER_WIN_BONUS")
        elif reason == "goat_win_no_tiger_moves":
            shaped_reward -= self._w("tiger_loss") * self._w("REWARD_TIGER_LOSS_PENALTY")
        elif reason == "max_timeout":
            shaped_reward -= (
                self._w("tiger_loss")
                * self._w("REWARD_TIGER_LOSS_PENALTY")
                * self._w("MAX_TIMEOUT_SCALE")
            )

        return shaped_reward
    

    
    # --------------------------------------------------------
    #  Gym API: step
    # --------------------------------------------------------
    def step(self, action):
        prev_obs = self.get_state().copy()
        if self.learner_role == TIGER_LEARNER:
            result = self._step_tiger_transition(action)
        else:
            result = self._step_goat_transition(action)
        return self._apply_reward_fn(prev_obs, action, result)


    def _step_goat_transition(self, action):
        if self.terminate:
            return self.get_state(), 0.0, True, False, self._build_step_info(reason="already_terminated")

        self.turns += 1
        flat_action = self._action_to_flat(action)
        decoded = self._safe_decode_action(action)
        base_info = {
            "goat_action": flat_action,
            "goat_decoded": decoded,
        }
        if decoded is None:
            info = self._build_step_info(reason="invalid_soft", invalid_action=True, **base_info)
            self.terminate = True
            return self.get_state(), 0.0, False, True, info

        pos, dir_code = decoded
        goats_eaten_before = int(self.eaten)

        goat_desc = ""
        tiger_desc = ""

        if not (0 <= pos < BOARD_SIZE) or not (0 <= dir_code < DIR_CODES):
            info = self._build_step_info(reason="invalid_soft", invalid_action=True, **base_info)
            self.terminate = True
            return self.get_state(), 0.0, False, True, info

        tiger_moves_before = int(self.prev_tiger_moves)
        bubble_before = int(self.prev_unreachable)
        spread_before = float(self.prev_tiger_spread)

        if self.phase == 0:
            step_penalty = self._w("REWARD_STEP")
        else:
            step_penalty = max(
                self._w("MOVE_STEP_BASE") + self._w("MOVE_STEP_SLOPE") * self.move_steps,
                self._w("MOVE_STEP_MIN"),
            )

        did_goat_move = False
        if self.phase == 0:
            if dir_code != 0 or self.board[pos] != 0:
                if DEBUG_INVALID:
                    print("how did you manage this error?", action)
                info = self._build_step_info(reason="invalid_soft", invalid_action=True, **base_info)
                self.terminate = True
                return self.get_state(), 0.0, False, True, info

            self.board[pos] = 1
            self.goats_placed += 1
            goat_desc = f"Goat placed at {idx_to_coord(pos)}"
            if self.goats_placed >= TOTAL_GOATS_TO_PLACE:
                self.phase = 1
        else:
            dest = self.move_map.get(pos, {}).get(dir_code, None)
            if dest is None or self.board[pos] != 1 or self.board[dest] != 0:
                if DEBUG_INVALID:
                    print("Invalid goat move:", action)
                self.terminate = True
                info = self._build_step_info(
                    reason="invalid_hard",
                    invalid_action=True,
                    error="Invalid move",
                    **base_info,
                )
                return self.get_state(), 0.0, False, True, info

            self.board[pos] = 0
            self.board[dest] = 1
            did_goat_move = True
            goat_desc = f"Goat {idx_to_coord(pos)} -> {idx_to_coord(dest)}"

        if did_goat_move:
            self.move_steps += 1

        tiger_options = self._tiger_moves(include_dir=False)
        tiger_moves_after = len(tiger_options)
        self.prev_tiger_moves = tiger_moves_after

        near_lock = (0 < tiger_moves_after <= 2)
        late = min(
            max(self.turns - self._w("LATE_GAME_START_TURN"), 0)
            / (self._w("MAX_TURNS") - self._w("LATE_GAME_START_TURN")),
            1.0,
        )
        delta_moves = tiger_moves_before - tiger_moves_after

        bubble_after = self._compute_unreachable_safe_cells()
        self.prev_unreachable = bubble_after
        delta_bubble = bubble_after - bubble_before

        spread_after = self._tiger_spread()
        self.prev_tiger_spread = spread_after
        delta_spread = spread_before - spread_after

        center_score = 0.0
        for idx in KEY_CENTERS:
            if self.board[idx] == 1:
                center_score += self._w("REWARD_CENTER_GOAT")
            elif self.board[idx] == 2:
                center_score += self._w("REWARD_CENTER_TIGER")

        common = dict(
            step_penalty=float(step_penalty),
            near_lock=bool(near_lock),
            late=float(late),
            delta_moves=int(delta_moves),
            delta_bubble=int(delta_bubble),
            delta_spread=float(delta_spread),
            center_score=float(center_score),
            goats_eaten_this_turn=0,
            repeat_prev_count=0,
        )

        if tiger_moves_after == 0:
            self.terminate = True
            self.last_move_desc = goat_desc or "Goat moved, then tigers had no moves"
            info = self._build_step_info(reason="goat_win_no_tiger_moves", winner="Goat", **base_info, **common)
            return self.get_state(), 0.0, True, False, info

        chosen = self._select_tiger_move(tiger_options)
        if chosen is None:
            raise RuntimeError(
                f"Tiger policy '{self.tiger_ai}' returned None despite "
                f"{len(tiger_options)} legal moves. State={self.get_state().tolist()}"
            )

        t_from, t_to, took_capture = chosen
        if took_capture:
            jumped_goat_pos = self._find_jumped_goat(t_from, t_to)
            if jumped_goat_pos is not None and self.board[jumped_goat_pos] == 1:
                self.board[jumped_goat_pos] = 0
                self.eaten += 1

        self.board[t_to] = 2
        self.board[t_from] = 0

        if t_from is not None and t_to is not None:
            if took_capture:
                tiger_desc = f"Tiger {idx_to_coord(t_from)} -> {idx_to_coord(t_to)} capture"
            else:
                tiger_desc = f"Tiger {idx_to_coord(t_from)} -> {idx_to_coord(t_to)}"

        goats_eaten_this_turn = int(self.eaten - goats_eaten_before)
        common["goats_eaten_this_turn"] = goats_eaten_this_turn
        common["tiger_move"] = (int(t_from), int(t_to), bool(took_capture))

        if goat_desc and tiger_desc:
            self.last_move_desc = f"{goat_desc} | {tiger_desc}"
        elif goat_desc:
            self.last_move_desc = goat_desc
        elif tiger_desc:
            self.last_move_desc = tiger_desc
        else:
            self.last_move_desc = ""

        if self.eaten >= GOATS_EATEN_FOR_TIGER_WIN:
            self.terminate = True
            info = self._build_step_info(reason="tiger_win_capture_threshold", winner="Tiger", **base_info, **common)
            return self.get_state(), 0.0, True, False, info

        if self.turns >= self.max_turns:
            self.terminate = True
            info = self._build_step_info(reason="max_timeout", winner="MaxTimeout", **base_info, **common)
            return self.get_state(), 0.0, False, True, info

        board_hash = (self.board.tobytes(), self.eaten, self.phase)
        prev_count = int(self.state_history.get(board_hash, 0))
        new_count = prev_count + 1
        self.state_history[board_hash] = new_count

        if new_count >= self._w("MAX_REPEATS"):
            self.terminate = True
            common["repeat_prev_count"] = prev_count
            info = self._build_step_info(reason="repeat_timeout", winner="RepeatTimeout", **base_info, **common)
            return self.get_state(), 0.0, False, True, info

        common["repeat_prev_count"] = prev_count
        self._update_valid_moves()
        info = self._build_step_info(**base_info, **common)
        return self.get_state(), 0.0, False, False, info

    def _step_tiger_transition(self, action):
        if self.terminate:
            return self.get_state(), 0.0, True, False, self._build_step_info(reason="already_terminated")

        self.turns += 1
        flat_action = self._action_to_flat(action)
        decoded = self._safe_decode_action(action)
        base_info = {
            "goat_action": flat_action,
            "goat_decoded": decoded,
        }
        if decoded is None:
            info = self._build_step_info(reason="invalid_soft", invalid_action=True, **base_info)
            self.terminate = True
            return self.get_state(), 0.0, False, True, info

        pos, dir_code = decoded
        goats_eaten_before = int(self.eaten)

        if not (0 <= pos < BOARD_SIZE) or not (0 <= dir_code < DIR_CODES):
            info = self._build_step_info(reason="invalid_soft", invalid_action=True, **base_info)
            self.terminate = True
            return self.get_state(), 0.0, False, True, info

        legal = {}
        for f, t, cap, d in self._tiger_moves(include_dir=True):
            key = (f, d)
            if key in legal:
                raise ValueError(f"duplicate tiger action key (from_pos,dir_code)={key}")
            legal[key] = (t, cap)

        if (pos, dir_code) not in legal:
            self.terminate = True
            info = self._build_step_info(
                reason="invalid_hard",
                invalid_action=True,
                error="Invalid tiger action",
                **base_info,
            )
            return self.get_state(), 0.0, False, True, info

        dest, took_capture = legal[(pos, dir_code)]
        step_penalty = self._w("REWARD_STEP")

        self.board[dest] = 2
        self.board[pos] = 0

        if took_capture:
            jumped_goat_pos = self._find_jumped_goat(pos, dest)
            if jumped_goat_pos is not None and self.board[jumped_goat_pos] == 1:
                self.board[jumped_goat_pos] = 0
                self.eaten += 1

        goats_eaten_this_turn = int(self.eaten - goats_eaten_before)

        tiger_before = int(self.prev_tiger_moves)
        tiger_after = len(self._tiger_moves())
        self.prev_tiger_moves = tiger_after
        delta_moves = tiger_after - tiger_before

        center_score = 0.0
        for idx in KEY_CENTERS:
            if self.board[idx] == 2:
                center_score += abs(self._w("REWARD_CENTER_GOAT"))
            elif self.board[idx] == 1:
                center_score -= abs(self._w("REWARD_CENTER_GOAT"))

        common = dict(
            step_penalty=float(step_penalty),
            goats_eaten_this_turn=goats_eaten_this_turn,
            delta_moves=int(delta_moves),
            center_score=float(center_score),
            tiger_move=(int(pos), int(dest), bool(took_capture)),
        )

        if self.eaten >= GOATS_EATEN_FOR_TIGER_WIN:
            self.terminate = True
            info = self._build_step_info(reason="tiger_win_capture_threshold", winner="Tiger", **base_info, **common)
            return self.get_state(), 0.0, True, False, info

        self._update_valid_moves()

        def goat_mask():
            mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)
            for mv in self.valid_moves:
                if np.isscalar(mv):
                    flat = int(mv)
                else:
                    flat = self.encode_action(mv[0], mv[1])
                if 0 <= flat < mask.size:
                    mask[flat] = True
            return mask

        def goat_random_action():
            mask = goat_mask()
            legal_idxs = np.flatnonzero(mask)
            if legal_idxs.size == 0:
                return None
            flat = int(np.random.choice(legal_idxs))
            return flat // DIR_CODES, flat % DIR_CODES

        def goat_model_action():
            if self._goat_model_predict_fn is None:
                return None
            mask = goat_mask()
            obs = self.get_state()
            try:
                flat = int(self._goat_model_predict_fn(obs, mask))
                if 0 <= flat < mask.size and mask[flat]:
                    return flat // DIR_CODES, flat % DIR_CODES
            except Exception:
                return None
            return None

        goat_choice = None
        if self.goat_opponent_ai == GOAT_AI_MODEL:
            goat_choice = goat_model_action()
        if goat_choice is None:
            goat_choice = goat_random_action()

        if goat_choice is None:
            self.terminate = True
            info = self._build_step_info(reason="tiger_win_no_goat_moves", winner="Tiger", **base_info, **common)
            return self.get_state(), 0.0, True, False, info

        g_pos, g_dir = goat_choice
        if self.phase == 0:
            if self.board[g_pos] == 0 and g_dir == 0:
                self.board[g_pos] = 1
                self.goats_placed += 1
                if self.goats_placed >= TOTAL_GOATS_TO_PLACE:
                    self.phase = 1
        else:
            g_dest = self.move_map[g_pos].get(g_dir, None)
            if g_dest is not None and self.board[g_pos] == 1 and self.board[g_dest] == 0:
                self.board[g_pos] = 0
                self.board[g_dest] = 1
                self.move_steps += 1

        if len(self._tiger_moves()) == 0:
            self.terminate = True
            info = self._build_step_info(reason="goat_win_no_tiger_moves", winner="Goat", **base_info, **common)
            return self.get_state(), 0.0, True, False, info

        if self.turns >= self.max_turns:
            self.terminate = True
            info = self._build_step_info(reason="max_timeout", winner="MaxTimeout", **base_info, **common)
            return self.get_state(), 0.0, False, True, info

        self._update_valid_moves()
        info = self._build_step_info(**base_info, **common)
        return self.get_state(), 0.0, False, False, info


    def _find_jumped_goat(self, t_from: int, t_to: int):
        """
        Given a tiger capture move t_from -> t_to (which must be a jump),
        find the intermediate neighbor that was jumped over.
        Returns jumped_pos or None if not found.
        """
        # Special hub rule for node 0
        if t_from == 0:
            for dir_code, neigh in self.move_map[0].items():
                if neigh is None:
                    continue
                jump = HUB0_CAPTURE_JUMP.get(dir_code, None)
                if jump == t_to:
                    return neigh
            return None

        # Default rule (same dir_code twice)
        for dir_code, neigh in self.move_map.get(t_from, {}).items():
            if neigh is None:
                continue
            jump = self.move_map.get(neigh, {}).get(dir_code, None)
            if jump == t_to:
                return neigh
        return None



    # --------------------------------------------------------
    #  Tiger AI selection helpers
    # --------------------------------------------------------
    def _select_tiger_move(self, tiger_options):
        if self.learner_role == GOAT_LEARNER:
            # tiger_options here should be 3-tuples
            if tiger_options and len(tiger_options[0]) != 3:
                raise ValueError("Goat-learner tiger_options must be (from,to,cap) 3-tuples")
        
        if self.tiger_ai == TIGER_AI_SMART:
            return smart_tiger(self.board, tiger_options, self.move_map, debug=False)
        return self._greedy_tiger_move(tiger_options)

    def _greedy_tiger_move(self, tiger_options):
        captures = [m for m in tiger_options if m[2]]
        capture_bias = self._w("BASE_TIGER_CAPTURE_BIAS")

        if captures and random.random() < capture_bias:
            return random.choice(captures)

        if tiger_options:
            return random.choice(tiger_options)

        return None


    # --------------------------------------------------------
    #  Tiger move generation
    # --------------------------------------------------------
    def _tiger_moves(self, include_dir: bool = False):
        """
        Compute all legal tiger moves.

        Returns:
          list of (from_pos, to_pos, is_capture[, dir_code])

        Deterministic rule:
          - dir_code is always the actual move_map direction used for both:
              (from -> neighbor) and (neighbor -> jump)
          - No special-case direction remapping for node 0.
        """
        moves = []

        for i in range(BOARD_SIZE):
            if self.board[i] != 2:
                continue

            for dir_code, neigh in self.move_map.get(i, {}).items():
                if neigh is None:
                    continue

                # normal move
                if self.board[neigh] == 0:
                    moves.append((i, neigh, False, dir_code) if include_dir else (i, neigh, False))
                    continue

                # capture candidate: goat in neighbor
                if self.board[neigh] != 1:
                    continue

                # jump destination
                if i == 0:
                    jump = HUB0_CAPTURE_JUMP.get(dir_code, None)
                else:
                    jump = self.move_map.get(neigh, {}).get(dir_code, None)

                if jump is None or self.board[jump] != 0:
                    continue

                moves.append((i, jump, True, dir_code) if include_dir else (i, jump, True))

        return moves





    # --------------------------------------------------------
    #  Goat move generation (valid_moves for masking)
    # --------------------------------------------------------
    def _update_valid_moves(self):
        """
        Recompute all legal goat actions and store in self.valid_moves.

        - During placing phase: any empty cell with dir_code=0
        - During moving phase: goats can move to empty neighbors
        """
        new_moves = []

        if self.phase == 0:
            # placing phase: any empty cell is valid
            for i in range(BOARD_SIZE):
                if self.board[i] == 0:
                    new_moves.append(self.encode_action(i, 0))
        else:
            # moving phase: move goats along legal directions
            for i in range(BOARD_SIZE):
                if self.board[i] == 1:  # goat
                    for dir_code, dest in self.move_map[i].items():
                        if dest is not None and self.board[dest] == 0:
                            new_moves.append(self.encode_action(i, dir_code))

        self.valid_moves = new_moves


    # ========================================================
    # Reward Shaping Helper Definitions
    # ========================================================

    def _compute_unreachable_safe_cells(self) -> int:
        """
        Count how many cells are unreachable by any tiger if tigers
        can only walk through empty cells and onto empty cells.
        (Goats act as walls for this reachability notion.)

        This approximates "bubbles": safe pockets tigers can't reach.
        """
        # 1) Find starting tiger positions
        tiger_starts = [i for i in range(BOARD_SIZE) if self.board[i] == 2]

        if not tiger_starts:
            # no tigers: treat all empty/goat cells as unreachable (safe)
            return int(np.sum((self.board == 0) | (self.board == 1)))
        
        # 2) BFS from all tigers through empty cells only
        visited = self._visited_buffer
        visited[:] = False

        q = deque()
        for t in tiger_starts:
            visited[t] = True
            q.append(t)

        while q:
            cur = q.popleft()
            for _, dest in self.move_map.get(cur, {}).items():
                # tiger can walk into empty neighbor (ignore goats in reachability)
                if self.board[dest] == 0 and not visited[dest]:
                    visited[dest] = True
                    q.append(dest)

        # 3) Count unreachable cells that are useful to goats (empty or goat)
        unreachable = 0
        for i in range(BOARD_SIZE):
            if not visited[i] and self.board[i] in (0, 1):
                unreachable += 1
        
        return unreachable
    

    def _precompute_all_pairs_dist(self) -> np.ndarray:
        """
        Precompute BFS shortest path lengths between all board positions
        in the static graph defined by move_map, ignoring pieces.
        """
        dist = np.full((BOARD_SIZE, BOARD_SIZE), BOARD_SIZE, dtype=np.int8)

        for start in range(BOARD_SIZE):
            visited = np.zeros(BOARD_SIZE, dtype=bool)
            q = deque()
            q.append((start, 0))
            visited[start] = True
            dist[start, start] = 0

            while q:
                cur, d = q.popleft()
                for _, dest in self.move_map.get(cur, {}).items():
                    if not visited[dest]:
                        visited[dest] = True
                        dist[start, dest] = d + 1
                        q.append((dest, d + 1))

        return dist
    

    def _shortest_path_len(self, start: int, goal: int) -> int:
        """
        Shortest path length in the static board graph using precomputed
        all-pairs distances.
        """
        return int(self._dist_matrix[start, goal])
    
    
    # we get a positive reward if tiger spread decreases
    def _tiger_spread(self) -> float:
        """
        Sum of pairwise distances between tigers.
        Smaller => tigers more clustered / forced together.
        """
        tiger_positions = [i for i in range(BOARD_SIZE) if self.board[i] == 2]
        if len(tiger_positions) < 2:
            return 0.0

        total = 0.0
        for idx_a in range(len(tiger_positions)):
            for idx_b in range(idx_a + 1, len(tiger_positions)):
                a = tiger_positions[idx_a]
                b = tiger_positions[idx_b]
                total += self._shortest_path_len(a, b)

        return float(total)
# end class TnGEnv()


