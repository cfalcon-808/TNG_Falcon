# ============================================================
#  Project    : Tigers & Goats – Falcon Branch
#  Module     : Tiger-Control Environment (Goat AI Opponent)
#  File       : env_tiger_falcon.py
#  Version    : env_tiger_1.0 (draft)
#  Last Update: 2025-12-30
#
#  Purpose:
#    Provide a lightweight Gymnasium environment where the *agent*
#    controls the tigers and plays against a built-in (or pluggable)
#    goat policy. Designed to pair with MaskablePPO using action masks
#    that expose only legal tiger moves.
#
#  Notes:
#    - Observation mirrors the goat env: board(23) + goats_eaten + phase.
#    - Action space is MultiDiscrete([23, 5]) by default; a Flatten wrapper
#      is included for Discrete(115) compatibility.
#    - Goat behavior defaults to random legal play; you can inject a goat
#      policy via goat_policy callable(goat_obs, mask) -> flat_action.
# ============================================================

import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

try:
    from sb3_contrib import MaskablePPO
except Exception:  # pragma: no cover - optional dependency
    MaskablePPO = None

# Game constants (mirrors env_tng_falcon)
BOARD_SIZE = 23
DIR_CODES = 5
TOTAL_GOATS_TO_PLACE = 15
GOATS_EATEN_FOR_TIGER_WIN = 6
TIGER_START_POSITIONS = [0, 3, 4]
MAX_TURNS = 100
MAX_REPEATS = 6

# Coordinate labels (same mapping as env_tng_falcon)
COORD_LABELS = [
    "b0",        # 0
    "a1", "b1", "c1", "d1", "e1", "f1",      # 1–6
    "a2", "b2", "c2", "d2", "e2", "f2",      # 7–12
    "a3", "b3", "c3", "d3", "e3", "f3",      # 13–18
    "b4", "c4", "d4", "e4",                  # 19–22
]


def idx_to_coord(idx: int) -> str:
    if 0 <= idx < len(COORD_LABELS):
        return COORD_LABELS[idx]
    return str(idx)


class TigerEnv(gym.Env):
    """
    Tigers-and-Goats environment where the agent controls the tigers.

    Observation (Box(25, int8)):
      - 23 board cells: 0=empty, 1=goat, 2=tiger
      - goats eaten (0..6)
      - phase: 0=goat placing, 1=goat moving

    Action (MultiDiscrete([23, 5])):
      - tiger_from position (0..22)
      - direction code (0..4), interpreted via move_map
    """

    metadata = {"render_modes": []}

    def __init__(self, goat_policy=None, goat_model_path=None, goat_device="cpu", goat_deterministic=True):
        super().__init__()
        self.goat_policy = goat_policy  # callable(obs, mask) -> flat action, optional
        self.goat_model = None
        self.goat_deterministic = goat_deterministic
        self._load_goat_model(goat_model_path, goat_device)

        self.observation_space = spaces.Box(
            low=np.array([0] * BOARD_SIZE + [0] + [0], dtype=np.int8),
            high=np.array([2] * BOARD_SIZE + [6] + [1], dtype=np.int8),
            shape=(BOARD_SIZE + 2,),
            dtype=np.int8,
        )
        self.action_space = spaces.MultiDiscrete([BOARD_SIZE, DIR_CODES])

        # Move map identical to env_tng_falcon
        self.move_map = {
            0: {1: 2, 2: 3, 3: 4, 4: 5},  # b0 special
            1: {2: 2, 3: 7},
            2: {1: 0, 2: 3, 3: 8, 4: 1},
            3: {1: 0, 2: 4, 3: 9, 4: 2},
            4: {1: 0, 2: 5, 3: 10, 4: 3},
            5: {1: 0, 2: 6, 3: 11, 4: 4},
            6: {3: 12, 4: 5},
            7: {1: 1, 2: 8, 3: 13},
            8: {1: 2, 2: 9, 3: 14, 4: 7},
            9: {1: 3, 2: 10, 3: 15, 4: 8},
            10: {1: 4, 2: 11, 3: 16, 4: 9},
            11: {1: 5, 2: 12, 3: 17, 4: 10},
            12: {1: 6, 3: 18, 4: 11},
            13: {1: 7, 2: 14},
            14: {1: 8, 2: 15, 3: 19, 4: 13},
            15: {1: 9, 2: 16, 3: 20, 4: 14},
            16: {1: 10, 2: 17, 3: 21, 4: 15},
            17: {1: 11, 2: 18, 3: 22, 4: 16},
            18: {1: 12, 4: 17},
            19: {1: 14, 2: 20},
            20: {1: 15, 2: 21, 4: 19},
            21: {1: 16, 2: 22, 4: 20},
            22: {1: 17, 4: 21},
        }

        self.reset()

    # ------------- Gym API -------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(BOARD_SIZE, dtype=np.int8)
        for idx in TIGER_START_POSITIONS:
            self.board[idx] = 2
        self.eaten = 0
        self.phase = 0
        self.goats_placed = 0
        self.turns = 0
        self.terminate = False
        self.state_history = {}
        return self._obs(), {"action_mask": self.get_action_mask()}

    def step(self, action):
        if self.terminate:
            return self._obs(), 0.0, True, False, {"action_mask": self.get_action_mask()}

        # Goat turn (auto policy)
        self._goat_turn()

        # Check goat win (tigers immobilized)
        tiger_moves = self._tiger_moves()
        if not tiger_moves:
            self.terminate = True
            return self._obs(), -1.0, True, False, {"winner": "Goat", "action_mask": self.get_action_mask()}

        # Tiger agent turn
        pos, dir_code = self._decode_action(action)
        legal = {(f, d): dest for (f, dest, is_cap, d) in tiger_moves_with_dir(tiger_moves, self.move_map)}

        if (pos, dir_code) not in legal:
            # Invalid: penalize and terminate
            self.terminate = True
            return self._obs(), -1.0, True, False, {"winner": "Goat", "invalid_action": True, "action_mask": self.get_action_mask()}

        dest = legal[(pos, dir_code)]
        took_capture = self.board[dest] == 1
        if took_capture:
            # remove jumped goat
            for d, neigh in self.move_map[pos].items():
                jump = self.move_map.get(neigh, {}).get(d, None)
                if pos == 0:
                    jump = self.move_map.get(neigh, {}).get(3, None)
                if jump == dest and self.board[neigh] == 1:
                    self.board[neigh] = 0
                    self.eaten += 1
                    break

        self.board[dest] = 2
        self.board[pos] = 0
        self.turns += 1

        # Repetition tracking (board hash includes eaten + phase)
        board_hash = (self.board.tobytes(), self.eaten, self.phase)
        prev_count = self.state_history.get(board_hash, 0)
        new_count = prev_count + 1
        self.state_history[board_hash] = new_count

        if new_count >= MAX_REPEATS:
            self.terminate = True
            return self._obs(), -1.0, True, False, {"winner": "StallTimeout", "action_mask": self.get_action_mask()}

        # Max turn timeout
        if self.turns >= MAX_TURNS:
            self.terminate = True
            return self._obs(), -1.0, True, False, {"winner": "Timeout", "action_mask": self.get_action_mask()}

        # Tiger win?
        if self.eaten >= GOATS_EATEN_FOR_TIGER_WIN:
            self.terminate = True
            return self._obs(), 1.0, True, False, {"winner": "Tiger", "action_mask": self.get_action_mask()}

        return self._obs(), 0.0, False, False, {"action_mask": self.get_action_mask()}
    #end def step()

    # ------------- Helpers -------------
    def _obs(self):
        return np.concatenate((self.board, [self.eaten, self.phase])).astype(np.int8)

    def get_action_mask(self):
        mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)
        for f, dest, is_cap, dir_code in tiger_moves_with_dir(self._tiger_moves(), self.move_map):
            flat = f * DIR_CODES + dir_code
            mask[flat] = True
        return mask

    def _decode_action(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) == 2:
            return int(action[0]), int(action[1])
        a = int(action)
        return a // DIR_CODES, a % DIR_CODES

    def _load_goat_model(self, path, device):
        if not path:
            return
        if MaskablePPO is None:
            print("[TigerEnv] sb3_contrib not available; cannot load goat model.")
            return
        if not os.path.isfile(path):
            print(f"[TigerEnv] Goat model not found at: {path}")
            return
        try:
            self.goat_model = MaskablePPO.load(path, device=device)
            print(f"[TigerEnv] Loaded goat model: {path}")
        except Exception as e:
            print(f"[TigerEnv] Failed to load goat model {path}: {e}")
            self.goat_model = None

        # Goat behaviors
    def _goat_turn(self):
        # If still placing, drop a goat; else move a goat.
        if self.phase == 0:
            empties = [i for i, v in enumerate(self.board) if v == 0]
            if not empties:
                return
            mask = self._goat_place_mask(empties)
            flat = self._goat_model_action(mask) or (self.goat_policy(self._obs(), mask) if self.goat_policy else None)
            if flat is not None and flat % DIR_CODES == 0:
                pos = flat // DIR_CODES
            else:
                pos = random.choice(empties)
            if pos not in empties:
                pos = random.choice(empties)
            self.board[pos] = 1
            self.goats_placed += 1
            if self.goats_placed >= TOTAL_GOATS_TO_PLACE:
                self.phase = 1
        else:
            moves = self._goat_moves()
            if not moves:
                return
            mask = self._goat_move_mask(moves)
            flat = self._goat_model_action(mask) or (self.goat_policy(self._obs(), mask) if self.goat_policy else None)
            if flat is not None:
                g_pos = flat // DIR_CODES
                g_dir = flat % DIR_CODES
                g_to = self.move_map.get(g_pos, {}).get(g_dir, None)
                if g_to is not None and self.board[g_pos] == 1 and self.board[g_to] == 0:
                    choice = (g_pos, g_to)
                else:
                    choice = random.choice(moves)
            else:
                choice = random.choice(moves)
            g_from, g_to = choice
            self.board[g_from] = 0
            self.board[g_to] = 1

    def _goat_place_mask(self, empties):
        mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)
        for e in empties:
            mask[e * DIR_CODES + 0] = True
        return mask

    def _goat_move_mask(self, moves):
        mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)
        for g_from, g_to in moves:
            for d, dest in self.move_map[g_from].items():
                if dest == g_to:
                    mask[g_from * DIR_CODES + d] = True
        return mask

    def _goat_model_action(self, mask):
        if self.goat_model is None:
            return None
        try:
            action, _ = self.goat_model.predict(
                self._obs(),
                deterministic=self.goat_deterministic,
                action_masks=mask,
            )
            return int(action)
        except Exception:
            return None

    # Move generators
    def _tiger_moves(self):
        moves = []
        for i in range(BOARD_SIZE):
            if self.board[i] == 2:
                for dir_code, dest in self.move_map[i].items():
                    if self.board[dest] == 0:
                        moves.append((i, dest, False))
                    elif self.board[dest] == 1:
                        if i in [2, 3, 4, 5] and dir_code == 1:
                            continue
                        jump = self.move_map.get(dest, {}).get(dir_code, None)
                        if i == 0:
                            jump = self.move_map.get(dest, {}).get(3, None)
                        if jump is not None and self.board[jump] == 0:
                            moves.append((i, jump, True))
        return moves

    def _goat_moves(self):
        moves = []
        for i in range(BOARD_SIZE):
            if self.board[i] == 1:
                for dir_code, dest in self.move_map[i].items():
                    if dest is not None and self.board[dest] == 0:
                        moves.append((i, dest))
        return moves


def tiger_moves_with_dir(moves, move_map):
    """
    Expand tiger moves to include dir_code (needed for masking/discrete mapping).
    """
    expanded = []
    for f, t, is_cap in moves:
        for d, dest in move_map.get(f, {}).items():
            if dest == t:
                expanded.append((f, t, is_cap, d))
    return expanded


class FlattenTigerActionWrapper(gym.Wrapper):
    """
    Wraps TigerEnv so external action space is Discrete(115).
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(BOARD_SIZE * DIR_CODES)
        self.observation_space = env.observation_space

    def get_action_mask(self):
        return self.env.get_action_mask()

    def step(self, action):
        flat = int(action)
        pos = flat // DIR_CODES
        dir_code = flat % DIR_CODES
        md_action = np.array([pos, dir_code], dtype=np.int64)
        return self.env.step(md_action)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
