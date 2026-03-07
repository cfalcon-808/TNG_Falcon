import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class TnGEnv(gym.Env):
    def __init__(self, tiger_policy: str = "greedy", max_turns: int = 400):
        super(TnGEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=np.array([0] * 23 + [0] + [0], dtype=np.int8),
            high=np.array([2] * 23 + [6] + [1], dtype=np.int8),
            shape=(25,),
            dtype=np.int8,
        )

        # Discrete(23*5) = 115 actions
        self.action_space = spaces.Discrete(23 * 5)

        # NEW: max turn truncation
        self.max_turns = int(max_turns)
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be > 0, got: {max_turns}")
        self.turn_counter = 0

        self.board = np.array([0] * 23, dtype=np.int8)
        self.board[0] = 2
        self.board[3] = 2
        self.board[4] = 2
        self.eaten = 0
        self.phase = 0
        self.goats_placed = 0
        self.move_map = {
            0: {1: 2, 2: 3, 3: 4, 4: 5},  # b0 special

            1: {2: 2, 3: 7},  # a1 → right=b1, down=a2
            2: {1: 0, 2: 3, 3: 8, 4: 1},  # b1 → right=c1, down=b2, left=a1
            3: {1: 0, 2: 4, 3: 9, 4: 2},  # c1 → up=b0, right=d1, down=c2, left=b1
            4: {1: 0, 2: 5, 3: 10, 4: 3},  # d1 → up=b0, right=e1, down=d2, left=c1
            5: {1: 0, 2: 6, 3: 11, 4: 4},  # e1 → up=b0, right=f1, down=e2, left=d1
            6: {3: 12, 4: 5},  # f1 → down=f2, left=e1

            7: {1: 1, 2: 8, 3: 13},  # a2 → up=a1, right=b2, down=a3
            8: {1: 2, 2: 9, 3: 14, 4: 7},  # b2 → up=b1, right=c2, down=b3, left=a2
            9: {1: 3, 2: 10, 3: 15, 4: 8},  # c2 → up=c1, right=d2, down=c3, left=b2
            10: {1: 4, 2: 11, 3: 16, 4: 9},  # d2 → up=d1, right=e2, down=d3, left=c2
            11: {1: 5, 2: 12, 3: 17, 4: 10},  # e2 → up=e1, right=f2, down=e3, left=d2
            12: {1: 6, 3: 18, 4: 11},  # f2 → up=f1, down=f3, left=e2

            13: {1: 7, 2: 14},  # a3 → up=a2, right=b3
            14: {1: 8, 2: 15, 3: 19, 4: 13},  # b3 → up=b2, right=c3, down=b4, left=a3
            15: {1: 9, 2: 16, 3: 20, 4: 14},  # c3 → up=c2, right=d3, down=c4, left=b3
            16: {1: 10, 2: 17, 3: 21, 4: 15},  # d3 → up=d2, right=e3, down=d4, left=c3
            17: {1: 11, 2: 18, 3: 22, 4: 16},  # e3 → up=e2, right=f3, down=e4, left=d3
            18: {1: 12, 4: 17},  # f3 → up=f2, left=e3

            19: {1: 14, 2: 20},  # b4 → up=b3, right=c4
            20: {1: 15, 2: 21, 4: 19},  # c4 → up=c3, right=d4, left=b4
            21: {1: 16, 2: 22, 4: 20},  # d4 → up=d3, right=e4, left=c4
            22: {1: 17, 4: 21},  # e4 → up=e3, left=d4
        }

        self.valid_moves = [self.encode_action(i, 0) for i in range(23) if self.board[i] == 0]
        if tiger_policy not in ("greedy", "smart"):
            raise ValueError(f"tiger_policy must be 'greedy' or 'smart', got: {tiger_policy}")
        self.tiger_policy = tiger_policy

    def encode_action(self, pos: int, dir_code: int) -> int:
        return int(pos) * 5 + int(dir_code)

    def decode_action(self, action: int) -> tuple[int, int]:
        a = int(action)
        return a // 5, a % 5

    def reset(self):
        super().reset()
        self.board = np.array([0] * 23, dtype=np.int8)
        self.board[0] = 2
        self.board[3] = 2
        self.board[4] = 2
        self.eaten = 0
        self.phase = 0
        self.goats_placed = 0
        self.turn_counter = 0
        self.valid_moves = [self.encode_action(i, 0) for i in range(23) if self.board[i] == 0]
        return self.get_state(), {}

    def get_state(self):
        return np.concatenate((self.board, [self.eaten, self.phase]))

    def render(self):
        symbols = {0: ".", 1: "G", 2: "T"}
        state = self.get_state()
        cells = [symbols[val] for val in state[:23]]

        goats_eaten = state[23]
        phase = "Place" if state[24] == 0 else "Move"

        board_str = f"""
       {cells[0]}(b0)
{cells[1]}(a1) {cells[2]}(b1) {cells[3]}(c1) {cells[4]}(d1) {cells[5]}(e1) {cells[6]}(f1)
{cells[7]}(a2) {cells[8]}(b2) {cells[9]}(c2) {cells[10]}(d2) {cells[11]}(e2) {cells[12]}(f2)
{cells[13]}(a3) {cells[14]}(b3) {cells[15]}(c3) {cells[16]}(d3) {cells[17]}(e3) {cells[18]}(f3)
   {cells[19]}(b4) {cells[20]}(c4) {cells[21]}(d4) {cells[22]}(e4)

Goats eaten: {goats_eaten} | Phase: {phase}
"""
        print(board_str)

    def step(self, action):
        prev_obs = self.get_state().copy()
        terminated = False
        truncated = False

        info = {
            "winner": None,          # "Goat" | "Tiger" | None
            "reason": None,          # "invalid_action" | "tigers_no_moves" | "goats_eaten_threshold" | "max_turns"
            "phase": int(self.phase),
            "goats_eaten": int(self.eaten),
            "goats_placed": int(self.goats_placed),
            "goat_action": int(action),
            "goat_decoded": None,
            "tiger_policy": self.tiger_policy,
            "tiger_move": None,
            "turn_counter": int(self.turn_counter),
        }

        pos, dir_code = self.decode_action(action)
        info["goat_decoded"] = (int(pos), int(dir_code))

        # 1) Goat turn
        invalid = False
        if self.phase == 0:
            if dir_code != 0 or self.board[pos] != 0:
                invalid = True
            else:
                self.board[pos] = 1
                self.goats_placed += 1
                if self.goats_placed >= 15:
                    self.phase = 1
        else:
            dest = self.move_map.get(pos, {}).get(dir_code, None)
            if dest is None or self.board[pos] != 1 or self.board[dest] != 0:
                invalid = True
            else:
                self.board[pos] = 0
                self.board[dest] = 1

        info["phase"] = int(self.phase)
        info["goats_placed"] = int(self.goats_placed)

        if invalid:
            truncated = True
            info["reason"] = "invalid_action"

        # 2) Tiger turn + goat win check
        if not truncated:
            tiger_options = self._tiger_moves()

            if not tiger_options:
                terminated = True
                info["winner"] = "Goat"
                info["reason"] = "tigers_no_moves"
            else:
                if self.tiger_policy == "smart":
                    out = self.smart_tiger(False)
                else:
                    out = self.greedy_tiger()

                if out is None:
                    info["reason"] = "tiger_policy_error"
                    raise RuntimeError(
                        f"Tiger policy '{self.tiger_policy}' returned None despite "
                        f"{len(tiger_options)} available tiger moves. "
                        f"State={self.get_state().tolist()}"
                    )

                t_from, t_to, capture = out
                info["tiger_move"] = (int(t_from), int(t_to), bool(capture))

                if capture:
                    for d, neigh in self.move_map[t_from].items():
                        if neigh is None:
                            continue
                        jump = self.move_map.get(neigh, {}).get(d, None)
                        if t_from == 0:
                            jump = self.move_map.get(neigh, {}).get(3, None)
                        if jump == t_to and self.board[neigh] == 1:
                            self.board[neigh] = 0
                            self.eaten += 1
                            break

                self.board[t_from] = 0
                self.board[t_to] = 2
                info["goats_eaten"] = int(self.eaten)

        # 3) Tiger win check
        if not terminated and not truncated:
            if self.eaten >= 6:
                terminated = True
                info["winner"] = "Tiger"
                info["reason"] = "goats_eaten_threshold"

        # 4) Update legal goat moves
        self._update_valid_moves()

        # 5) Max turns truncation (last thing before reward/return)
        self.turn_counter += 1
        info["turn_counter"] = int(self.turn_counter)
        if not terminated and not truncated and self.turn_counter >= self.max_turns:
            truncated = True
            info["reason"] = "max_turns"

        obs = self.get_state()
        reward = self.sparse_reward(prev_obs, action, obs, terminated, truncated, info)
        return obs, float(reward), terminated, truncated, info

    def _tiger_moves(self):
        moves = []
        for i in range(23):
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

    def _update_valid_moves(self):
        new_moves = []
        if self.phase == 0:
            for i in range(23):
                if self.board[i] == 0:
                    new_moves.append(self.encode_action(i, 0))
        else:
            for i in range(23):
                if self.board[i] == 1:
                    for dir_code, dest in self.move_map[i].items():
                        if dest is not None and self.board[dest] == 0:
                            new_moves.append(self.encode_action(i, dir_code))
        self.valid_moves = new_moves

    def debug_simulate(self,tiger_policy):
        env = TnGEnv(tiger_policy=tiger_policy)
        obs, _ = env.reset()
        terminated = False
        truncated = False
        print("=== DEBUG SIM START ===")
        print("tiger_policy:", tiger_policy)
        while not (terminated or truncated):
          a = random.choice(env.valid_moves)
          obs, reward, terminated, truncated, info = env.step(a)
          env.render()
          print("\nreturns:", (obs, reward, terminated, truncated))
          print("info:", info)
        print("\n=== DEBUG SIM END ===")

    # ----------------- default tiger policies -----------------

    def greedy_tiger(self):
        tiger_moves = self._tiger_moves()
        if not tiger_moves:
            return None
        captures = [m for m in tiger_moves if m[2]]
        return random.choice(captures if captures else tiger_moves)

    def smart_tiger(self, debug: bool = False):
        """
        Smart tiger:
        - If captures exist: choose the "best" capture by position value (ties broken randomly)
        - Else: try anchor strategy (0, and/or 9/10) using BFS shortest path
        - Else: roam to high-value squares
        - Else: random fallback
        - If no moves, return None
        Returns: (from_pos, to_pos, is_capture) or None
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
            Returns (first_step, dist) for shortest path from start_pos to any target in target_positions,
            walking through empty squares (0) and allowing stepping onto a target even if occupied check differs.
            """
            if start_pos in target_positions:
                return (None, 0)

            queue = []
            visited = {start_pos}

            # seed from neighbors, record their first step
            for _, neighbor in self.move_map[start_pos].items():
                if neighbor is None:
                    continue
                is_walkable = (self.board[neighbor] == 0) or (neighbor in target_positions)
                if is_walkable:
                    if neighbor in target_positions:
                        return (neighbor, 1)
                    queue.append((neighbor, neighbor, 1))
                    visited.add(neighbor)

            while queue:
                curr, first_step, dist = queue.pop(0)
                if curr in target_positions:
                    return (first_step, dist)

                for _, neighbor in self.move_map[curr].items():
                    if neighbor is None or neighbor in visited:
                        continue
                    is_walkable = (self.board[neighbor] == 0) or (neighbor in target_positions)
                    if is_walkable:
                        visited.add(neighbor)
                        queue.append((neighbor, first_step, dist + 1))

            return (None, None)

        tiger_moves = self._tiger_moves()
        if not tiger_moves:
            debug_print("No legal tiger moves.")
            return None

        # 1) CAPTURE LOGIC
        captures = [m for m in tiger_moves if m[2]]
        if captures:
            random.shuffle(captures)  # tie-breaking randomness
            best_capture = max(captures, key=lambda x: POS_VALS.get(x[1], 0))
            return best_capture

        # 2) ANCHOR LOGIC
        current_tiger_positions = [i for i in range(23) if self.board[i] == 2]
        anchor_0_occupied = (self.board[0] == 2)
        anchor_primary_occupied = (self.board[9] == 2 or self.board[10] == 2)

        need_anchor0 = not anchor_0_occupied
        need_primary = not anchor_primary_occupied

        if need_anchor0 or need_primary:
            anchor_groups = []
            if need_anchor0:
                anchor_groups.append(("secondary", [0]))
            if need_primary:
                primary_targets = [p for p in (9, 10) if self.board[p] == 0] or [9, 10]
                anchor_groups.append(("primary", primary_targets))

            best = None  # (dist, from_pos, first_step)

            for t_pos in current_tiger_positions:
                # don't move a tiger that's already fulfilling an occupied anchor role
                if anchor_0_occupied and t_pos == 0:
                    continue
                if anchor_primary_occupied and t_pos in (9, 10):
                    continue

                for _, targets in anchor_groups:
                    step, dist = bfs_find_next_step(t_pos, targets)
                    if step is None:
                        continue
                    if (best is None) or (dist < best[0]):
                        best = (dist, t_pos, step)

            if best is not None:
                _, from_pos, step = best
                chosen_move = next((m for m in tiger_moves if m[0] == from_pos and m[1] == step), None)
                if chosen_move is not None:
                    return chosen_move

        # 3) ROAMING LOGIC
        high_value_moves = []
        for move in tiger_moves:
            t_from, t_to, _ = move

            # preserve your "don't leave 0 once anchored" behavior
            if t_from == 0:
                continue

            # if we already have a primary anchor, try not to break it unnecessarily
            if (t_from in (9, 10)) and anchor_primary_occupied:
                if self.board[9] == 2 and self.board[10] != 2 and t_from == 9:
                    continue
                if self.board[10] == 2 and self.board[9] != 2 and t_from == 10:
                    continue

            if POS_VALS.get(t_to, 0) >= 5:
                high_value_moves.append(move)

        if high_value_moves:
            return random.choice(high_value_moves)
        # 4) FALLBACK
        return random.choice(tiger_moves)

    # ----------------- reward -----------------

    def sparse_reward(self, prev_obs, action, obs, terminated, truncated, info):
        if terminated:
            w = info.get("winner")
            if w == "Goat":
                return 1.0
            if w == "Tiger":
                return -1.0
            return 0.0
        if truncated:
            reason = info.get("reason")
            if reason == "invalid_action":
                return -10.0
            if reason == "max_turns":
                return -0.25
            return 0.0
        return 0.0
