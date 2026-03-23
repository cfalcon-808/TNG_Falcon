# ============================================================
#  Project    : Tigers & Goats – Falcon Branch
#  Module     : Battle Environment (Goat vs smart_tiger)
#  File       : env_battle_falcon.py
#  Version    : envb1.1
#  Last update: 12/4/25
#
#  Reward Structure Overview (battle_env1.0):
#    • Core outcomes:
#         - Goat win:          +REWARD_GOAT_WIN (slightly reduced by turn count)
#         - Tiger win:         +REWARD_TIGER_WIN (strong negative for goats)
#         - GoatTimeout:       scaled tiger-win penalty via GOAT_TIMEOUT_SCALE
#         - StallTimeout:      scaled tiger-win penalty via GOAT_STALL_SCALE
#
#    • Per-step incentives:
#         - Placing phase:     small negative REWARD_STEP to nudge progress
#         - Moving phase:      ramping penalty:
#                               MOVE_STEP_BASE + MOVE_STEP_SLOPE * move_steps
#                               clamped at MOVE_STEP_MIN
#
#    • Strategic shaping (goat-focused):
#         - Mobility shaping:   REWARD_BLOCK_TIGER for reducing tiger moves
#         - Near-lock bonus:    extra reward when 0 < tiger_moves_after ≤ 2
#         - Bubble creation:    REWARD_BUBBLE_SPACE for unreachable safe cells
#         - Tiger clustering:   REWARD_CLUSTER_TIGERS for decreased tiger spread
#         - Center control:     goats gain +REWARD_CENTER_GOAT on KEY_CENTERS;
#                               tigers incur REWARD_CENTER_TIGER on KEY_CENTERS
#
#    • Penalties:
#         - Goat captured:      REWARD_GOAT_EATEN (moderate negative per capture)
#         - Invalid actions:    REWARD_INVALID_SOFT for masked / OOB actions,
#                               REWARD_INVALID_HARD for illegal goat moves
#                               (terminal training stabilizer)
#         - State repetition:   quadratic REWARD_REPEAT_STATE * (k²)
#                               discourages loops and stall cycling
#
#  Tiger Policy (Opponent):
#    • Uses smart_tiger() heuristic policy with:
#         - Capture priority (positional scoring via POS_VALS)
#         - Anchor logic at positions 0 and {9,10} to maintain structure
#         - Roaming preference for high-value target squares
#         - Randomized fallback among remaining legal moves
#
#  Purpose:
#    Provide a battle-focused environment where a learning Goat agent:
#      • faces a strong heuristic smart_tiger opponent,
#      • is rewarded for limiting tiger mobility and forming safe bubbles,
#      • is guided toward center and structural control of the board,
#      • is penalized for being captured, stalling, or repeating states,
#      • and learns to convert positional advantage into immobilization wins.
# ============================================================
#
#
#  Reward Structure Overview (battle_env1.0)
#  (Line numbers reference where each constant is applied below.)
#
#    • Core outcomes:
#         - Goat win:
#              +REWARD_GOAT_WIN
#                Used at goat win lines:
#                   • L~476 — goat immobilizes tiger
#                   • L~498 — fallback win if tiger AI fails
#
#         - Tiger win:
#              +REWARD_TIGER_WIN
#                Used at:
#                   • L~515 — tiger wins by eating 6 goats
#                   • L~528 — max-turn timeout scaling
#                   • L~553 — stall timeout scaling
#
#         - GoatTimeout:
#              GOAT_TIMEOUT_SCALE (× REWARD_TIGER_WIN)
#                Used at:
#                   • L~528 — max turns exceeded
#
#         - StallTimeout:
#              GOAT_STALL_SCALE (× REWARD_TIGER_WIN)
#                Used at:
#                   • L~553 — repeated-state threshold exceeded
#
#    • Per-step incentives:
#         - REWARD_STEP
#                Applied at L~392 — placing phase step penalty
#
#         - MOVE_STEP_BASE, MOVE_STEP_SLOPE, MOVE_STEP_MIN
#                Applied at L~395–398 — moving-phase ramp penalty
#
#    • Strategic shaping (goat-focused):
#         - REWARD_BLOCK_TIGER
#                Applied at L~431–437 — mobility delta shaping
#
#         - REWARD_BUBBLE_SPACE
#                Applied at L~445 — unreachable safe-cell shaping
#
#         - REWARD_CLUSTER_TIGERS
#                Applied at L~455 — decreasing tiger spread
#
#         - REWARD_CENTER_GOAT / REWARD_CENTER_TIGER
#                Applied at L~463–468 — center-control shaping
#
#         - Near-lock Bonus:
#                Hardcoded +0.3 bonus at L~414
#
#    • Penalties:
#         - REWARD_GOAT_EATEN
#                Applied at L~509 — after tiger turn if goat is captured
#
#         - REWARD_INVALID_SOFT
#                Applied at:
#                   • L~371 (OOB or masked)
#                   • L~384 (invalid placement)
#
#         - REWARD_INVALID_HARD
#                Applied at:
#                   • L~402 — illegal goat movement (terminal)
#
#         - REWARD_REPEAT_STATE
#                Applied at:
#                   • L~545 — quadratic penalty for repeated states
#
#         - MAX_REPEATS
#                Checked at:
#                   • L~539–551 — stall timeout condition
#
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
    idx_to_coord,
    FalconEnvCore,  # shared board mechanics / render
)

# ============================================================
#  GLOBAL CONSTANTS – Game Rules / Debug
# ============================================================

MAX_TURNS                 = 100         # hard episode cap (timeouts)


# ============================================================
#  REWARD CONSTANTS – Terminal Outcomes
# ============================================================

REWARD_GOAT_WIN           =  3.2        # base reward when goats win
REWARD_TIGER_WIN          = -3.2        # base reward when tigers win

GOAT_WIN_TURN_DECAY       =  0.01       # goat-win reward decay per turn
GOAT_TIMEOUT_SCALE        =  1.0        # scale goat-win reward on timeouts
GOAT_STALL_SCALE          =  0.8        # scale goat-win reward on stall timeouts


# ============================================================
#  REWARD CONSTANTS – Per-Step Costs
# ============================================================

REWARD_STEP               = -0.001      # baseline per-step cost (all phases)

MOVE_STEP_BASE            = -0.01       # base move-phase step penalty
MOVE_STEP_SLOPE           = -0.01       # extra move penalty per move step
MOVE_STEP_MIN             = -0.6        # clamp for move-phase penalty floor


# ============================================================
#  REWARD SHAPING – Strategic Incentives
# ============================================================

REWARD_GOAT_EATEN         = -0.35       # penalty per goat captured

REWARD_BLOCK_TIGER        =  0.08       # reward for reducing tiger mobility
REWARD_BUBBLE_SPACE       =  0.02       # reward for creating safe goat regions
NEAR_LOCK_BONUS           =  0.3        # bonus when tiger has <= 2 moves left
REWARD_CLUSTER_TIGERS     =  0.02       # reward when tigers group closer

KEY_CENTERS               = [9, 10, 15, 16]  # high-value central positions
REWARD_CENTER_GOAT        =  0.05       # bonus for goats occupying centers
REWARD_CENTER_TIGER       = -0.03       # penalty when tigers hold centers

LATE_GAME_START_TURN      = 40          # turn when late-game scaling starts
MOBILITY_BACKSLIDE_SCALE  = 0.5         # weight for bad mobility changes



# ============================================================
#  PENALTIES – Invalid / Repetitive Behavior
# ============================================================

REWARD_INVALID_SOFT       = -0.05       # soft penalty for masked/invalid action
REWARD_INVALID_HARD       = -1.0        # hard penalty for illegal goat move
REWARD_REPEAT_STATE       = -0.5        # penalty for repeating a board state
MAX_REPEATS               = 6           # max times a state may repeat


# ============================================================
#  REWARD ABLATION – Term Weights
#     1.0 = keep as-is
#     0.0 = ablate (remove)
#     >1  = up-weight, <1 = down-weight
# ============================================================

DEFAULT_REWARD_WEIGHTS = {
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
#  TIGER AI – smart_tiger
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
#  BATTLE ENV – Goat reward heuristics + smart_tiger opponent
# ============================================================

class BattleEnv(FalconEnvCore, gym.Env):
    """
    Tigers & Goats Environment (Battle Variant).

    - Same observation / action structure as your main env.
    - Full goat-side reward shaping.
    - Tigers controlled by smart_tiger() instead of the simpler
      random/capture-bias heuristic.

    Observation (Box(25,)):
      - 23 board cells: 0=empty, 1=goat, 2=tiger
      - 1 value: goats eaten (0..6)
      - 1 value: phase (0=placing, 1=moving)

    Action (MultiDiscrete([23, 5])):
      - position index (0..22)
      - direction code (0..4), interpreted via move_map
    """

    def __init__(self, reward_weights = None):
        super().__init__()

        # Ablation weights (copy defaults, then override them if provided)
        self.reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
        if reward_weights is not None:
            self.reward_weights.update(reward_weights)

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
        # Action space (pos, dir)
        # -------------------------
        self.action_space = spaces.MultiDiscrete([BOARD_SIZE, DIR_CODES])

        # Game state (core: board pieces + valid_moves)
        self._init_board_state()
        self.last_move_desc = ""

        # Cached geometry for shaping
        tiger_opts = self._tiger_moves()  # core: tiger move generator
        self.prev_tiger_moves = len(tiger_opts)
        self.prev_unreachable = self._compute_unreachable_safe_cells()  # core: bubble metric
        self.prev_tiger_spread = self._tiger_spread()  # core: tiger spread metric

    # --------------------------------------------------------
    # 
    # --------------------------------------------------------
    def _w(self, term: str) -> float:
        """Return the ablation weight for a given reward term."""
        return self.reward_weights.get(term, 1.0)


    # --------------------------------------------------------
    #  Gym API: reset
    # --------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._init_board_state()  # core: board pieces + valid_moves
        self.last_move_desc = ""

        tiger_opts = self._tiger_moves()
        self.prev_tiger_moves = len(tiger_opts)
        self.prev_unreachable = self._compute_unreachable_safe_cells()  # core: bubble metric
        self.prev_tiger_spread = self._tiger_spread()  # core: tiger spread metric

        obs = self.get_state()
        info = {"action_mask": self.get_action_mask()}
        return obs, info

    # --------------------------------------------------------
    #  State + Rendering
    # --------------------------------------------------------
    def render(self, show_coords=True):
        """Render using the shared board renderer (from core)."""
        self.render_board(
            show_coords=show_coords,
            last_move_desc=getattr(self, "last_move_desc", ""),
        )

    # --------------------------------------------------------
    #  Gym API: step (full goat shaping + smart_tiger)
    # --------------------------------------------------------
    def step(self, action):
        if self.terminate:
            return self.get_state(), 0.0, True, False, {
                "action_mask": self.get_action_mask()
            }

        self.turns += 1

        pos, dir_code = self._decode_action(action)

        # Human readable move descriptions for this turn
        goat_desc = ""
        tiger_desc = ""

        # Soft invalid: out-of-bounds
        if not (0 <= pos < BOARD_SIZE) or not (0 <= dir_code < DIR_CODES):
            return self.get_state(), self._w("invalid_soft") * REWARD_INVALID_SOFT, False, False, {
                "invalid_action": True,
                "action_mask": self.get_action_mask(),
            }

        tiger_moves_before   = self.prev_tiger_moves
        bubble_before        = self.prev_unreachable
        spread_before        = self.prev_tiger_spread
        goats_eaten_before   = self.eaten

        # Base per-step penalty
        if self.phase == 0:
            shaped_reward = self._w("step") * REWARD_STEP
        else:
            step_pen = MOVE_STEP_BASE + MOVE_STEP_SLOPE * self.move_steps
            step_pen = max(step_pen, MOVE_STEP_MIN)
            shaped_reward = self._w("step") * step_pen

        # -------------------------
        # Goat turn
        # -------------------------
        if self.phase == 0:
            # Placing phase
            if dir_code != 0 or self.board[pos] != 0:
                if DEBUG_INVALID:
                    print("Invalid placement:", action)
                return self.get_state(), self._w("invalid_soft") * REWARD_INVALID_SOFT, False, False, {
                    "invalid_action": True,
                    "action_mask": self.get_action_mask(),
                }

            self.board[pos] = 1
            self.goats_placed += 1

            # record goat placement
            goat_desc =f"Goat placed at {idx_to_coord(pos)}"

            # Determine if game shifts to move phase
            if self.goats_placed >= TOTAL_GOATS_TO_PLACE:
                self.phase = 1

        else:
            # Moving phase
            dest = self.move_map[pos].get(dir_code, None)
            if dest is None or self.board[pos] != 1 or self.board[dest] != 0:
                if DEBUG_INVALID:
                    print("Invalid goat move:", action)
                self.terminate = True
                return self.get_state(), self._w("invalid_hard") * REWARD_INVALID_HARD, True, False, {
                    "error": "Invalid move",
                    "action_mask": self.get_action_mask(),
                }

            self.board[pos] = 0
            self.board[dest] = 1

            # record goat move (from -> to)
            goat_desc = f"Goat {idx_to_coord(pos)} -> {idx_to_coord(dest)}"

        if self.phase == 1:
            self.move_steps += 1

        # -------------------------
        # Recompute geometry after goat move
        # -------------------------
        tiger_options = self._tiger_moves()
        tiger_moves_after = len(tiger_options)
        self.prev_tiger_moves = tiger_moves_after

        # Near-lock bonus
        if 0 < tiger_moves_after <= 2:
            shaped_reward += self._w("near_lock") * NEAR_LOCK_BONUS

        late = min(
            max(self.turns - LATE_GAME_START_TURN, 0)
            / (MAX_TURNS - LATE_GAME_START_TURN), 
            1.0
            )

        # 1) Mobility shaping
        delta_moves = tiger_moves_before - tiger_moves_after
        if delta_moves > 0:
            shaped_reward += (
                self._w("block_tiger") 
                * REWARD_BLOCK_TIGER
                * delta_moves 
                * (1.0 + late)
                )
        elif delta_moves < 0:
            shaped_reward += (
                self._w("block_tiger") 
                * REWARD_BLOCK_TIGER 
                * MOBILITY_BACKSLIDE_SCALE 
                * delta_moves
                )

        # 2) Bubble shaping
        bubble_after = self._compute_unreachable_safe_cells()
        self.prev_unreachable = bubble_after
        delta_bubble = bubble_after - bubble_before
        if delta_bubble > 0:
            shaped_reward += (
                self._w("bubble") 
                * REWARD_BUBBLE_SPACE 
                * delta_bubble 
                * (1.0 + late)
                )

        # 3) Tiger clustering
        spread_after = self._tiger_spread()
        self.prev_tiger_spread = spread_after
        delta_spread = spread_before - spread_after
        if delta_spread > 0:
            shaped_reward += (
                self._w("cluster") 
                * REWARD_CLUSTER_TIGERS 
                * delta_spread 
                * (1.0 + late)
                )

        # 4) Center control
        center_score = 0.0
        for idx in KEY_CENTERS:
            if self.board[idx] == 1:
                center_score += REWARD_CENTER_GOAT
            elif self.board[idx] == 2:
                center_score += REWARD_CENTER_TIGER
        shaped_reward += self._w("center") * center_score

        # 5) Goat win terminal state (no tiger moves)
        if tiger_moves_after == 0:
            self.terminate = True
            shaped_reward = (
                self._w("goat_win") 
                * REWARD_GOAT_WIN - GOAT_WIN_TURN_DECAY 
                * self.turns
            )

            self.last_move_desc = (
                goat_desc or "Goat moved, then tigers had no moves"
            )

            return self.get_state(), shaped_reward, True, False, {
                "winner": "Goat",
                "action_mask": self.get_action_mask(),
            }

        # -------------------------
        # Tiger turn – uses smart_tiger instead of random heuristic
        # -------------------------
        chosen = smart_tiger(self.board, tiger_options, self.move_map, debug=False)

        if chosen is None:
            # Safety: if AI gives nothing, treat as goat win
            self.terminate = True
            # Goat win reward is less as game goes on
            shaped_reward = (
                self._w("goat_win") 
                * REWARD_GOAT_WIN - GOAT_WIN_TURN_DECAY 
                * self.turns
            )
            return self.get_state(), shaped_reward, True, False, {
                "winner": "Goat",
                "action_mask": self.get_action_mask(),
            }

        t_from, t_to, took_capture = chosen

        if took_capture:
            for d_code, neigh in self.move_map[t_from].items():
                jump = self.move_map.get(neigh, {}).get(d_code, None)
                if t_from == 0:
                    jump = self.move_map.get(neigh, {}).get(3, None)
                if jump == t_to and self.board[neigh] == 1:
                    self.board[neigh] = 0
                    self.eaten += 1
                    break

        self.board[t_to] = 2
        self.board[t_from] = 0

        # human readable tiger move
        if t_from is not None and t_to is not None:
            if took_capture:
                tiger_desc = (
                    f"Tiger {idx_to_coord(t_from)} -> {idx_to_coord(t_to)} capture"
                )
            else:
                tiger_desc = f"Tiger {idx_to_coord(t_from)} -> {idx_to_coord(t_to)}"

        # Goat eaten shaping
        goats_eaten_this_turn = self.eaten - goats_eaten_before
        if goats_eaten_this_turn > 0:
            shaped_reward += (
                self._w("goat_eaten") 
                * REWARD_GOAT_EATEN 
                * goats_eaten_this_turn
            )

        # store combined last-move description
        if goat_desc and tiger_desc:
            self.last_move_desc = f"{goat_desc} | {tiger_desc}"
        elif goat_desc:
            self.last_move_desc = goat_desc
        elif tiger_desc:
            self.last_move_desc = tiger_desc
        else:
            self.last_move_desc = ""

        # Tiger win
        if self.eaten >= GOATS_EATEN_FOR_TIGER_WIN:
            self.terminate = True
            shaped_reward += self._w("tiger_win") * REWARD_TIGER_WIN
            return self.get_state(), shaped_reward, True, False, {
                "winner": "Tiger",
                "action_mask": self.get_action_mask(),
            }

        # Max turn timeout
        if self.turns >= MAX_TURNS:
            self.terminate = True
            shaped_reward += (
                self._w("goat_timeout") 
                * REWARD_TIGER_WIN 
                * GOAT_TIMEOUT_SCALE
            )

            return self.get_state(), shaped_reward, True, False, {
                "winner": "GoatTimeout",
                "action_mask": self.get_action_mask(),
            }

        # Repetition penalty
        board_hash = (self.board.tobytes(), self.eaten, self.phase)
        prev_count = self.state_history.get(board_hash, 0)
        new_count = prev_count + 1
        self.state_history[board_hash] = new_count

        if new_count >= MAX_REPEATS:
            self.terminate = True
            shaped_reward += (
                self._w("goat_stall") 
                * REWARD_TIGER_WIN 
                * GOAT_STALL_SCALE
            )

            return self.get_state(), shaped_reward, True, False, {
                "winner": "StallTimeout",
                "action_mask": self.get_action_mask(),
            }

        if prev_count > 0:
            repeat_pen = REWARD_REPEAT_STATE * (prev_count ** 2)
            shaped_reward += self._w("repeat_state") * repeat_pen

        # Update goat legal moves
        self._update_valid_moves()

        return self.get_state(), shaped_reward, False, False, {
            "action_mask": self.get_action_mask()
        }


# ============================================================
# Flatten wrapper for Discrete(115) actions
# ============================================================

class FlattenBattleActionWrapper(gym.Wrapper):
    """
    Wraps BattleEnv so external action space is Discrete(115),
    but internally still uses MultiDiscrete([23, 5]).
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(BOARD_SIZE * DIR_CODES)
        self.observation_space = env.observation_space

    def get_action_mask(self):
        return self.env.get_action_mask()  # type: ignore[attr-defined]

    def step(self, action):
        flat = int(action)
        pos = flat // DIR_CODES
        dir_code = flat % DIR_CODES
        md_action = np.array([pos, dir_code], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.env.step(md_action)
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
