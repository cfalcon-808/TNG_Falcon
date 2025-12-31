import numpy as np
from collections import deque

# ============================================================
#  Shared constants
# ============================================================
BOARD_SIZE = 23
DIR_CODES = 5
TOTAL_GOATS_TO_PLACE = 15
GOATS_EATEN_FOR_TIGER_WIN = 6
TIGER_START_POSITIONS = [0, 3, 4]
KEY_CENTERS = [9, 10, 15, 16]
DEBUG_INVALID = False

# Human-readable board labels
COORD_LABELS = [
    "b0",        # 0
    "a1", "b1", "c1", "d1", "e1", "f1",      # 1–6
    "a2", "b2", "c2", "d2", "e2", "f2",      # 7–12
    "a3", "b3", "c3", "d3", "e3", "f3",      # 13–18
    "b4", "c4", "d4", "e4",                  # 19–22
]


def idx_to_coord(idx: int) -> str:
    """Safely convert a board index to its coordinate label."""
    if 0 <= idx < len(COORD_LABELS):
        return COORD_LABELS[idx]
    return str(idx)


def build_move_map():
    """Static move graph for the Tigers & Goats board."""
    return {
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


class FalconEnvCore:
    """
    Shared board + geometry helpers for both env variants.
    Subclasses are responsible for reward logic; this mixin
    keeps the board mechanics in one place.
    """

    def __init__(self):
        super().__init__()
        self.move_map = build_move_map()
        self._dist_matrix = self._precompute_all_pairs_dist()
        self._visited_buffer = np.zeros(BOARD_SIZE, dtype=bool)

    # --------------------------------------------------------
    #  Core state helpers
    # --------------------------------------------------------
    def _init_board_state(self):
        """Reset board pieces and generic counters."""
        self.board = np.array([0] * BOARD_SIZE, dtype=np.int8)
        for idx in TIGER_START_POSITIONS:
            self.board[idx] = 2

        self.eaten = 0
        self.phase = 0
        self.goats_placed = 0
        self.terminate = False

        self.turns = 0
        self.move_steps = 0
        self.state_history = {}

        self.valid_moves = [
            [i, 0] for i in range(BOARD_SIZE) if self.board[i] == 0
        ]

    def get_state(self):
        """Flattened state: [board(23), eaten(1), phase(1)] as int8."""
        return np.concatenate(
            (self.board, [self.eaten, self.phase])
        ).astype(np.int8)

    def get_action_mask(self):
        """Boolean mask of shape (115,) built from valid_moves."""
        mask = np.zeros(BOARD_SIZE * DIR_CODES, dtype=bool)
        for pos, dir_code in self.valid_moves:
            flat = pos * DIR_CODES + dir_code
            mask[flat] = True
        return mask

    def _decode_action(self, action):
        """
        Accept either:
          - Discrete(115) scalar (int/np.int): a = pos*5 + dir
          - MultiDiscrete([23,5]) vector: [pos, dir]
        Returns (pos, dir) as ints.
        """
        if isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.item()

        if np.isscalar(action):
            a = int(action)
            pos = a // DIR_CODES
            dir_code = a % DIR_CODES
            return pos, dir_code

        if isinstance(action, (list, tuple, np.ndarray)):
            arr = np.asarray(action).reshape(-1)
            if arr.size == 2:
                return int(arr[0]), int(arr[1])

        raise ValueError(f"Unrecognized action format: {action} (type {type(action)})")

    # --------------------------------------------------------
    #  Move generation helpers
    # --------------------------------------------------------
    def _tiger_moves(self):
        """
        Compute all legal tiger moves.

        Returns:
          list of (from_pos, to_pos, is_capture)
        """
        moves = []

        for i in range(BOARD_SIZE):
            if self.board[i] == 2:  # tiger at this position
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
        """
        Recompute all legal goat actions and store in self.valid_moves.

        - During placing phase: any empty cell with dir_code=0
        - During moving phase: goats can move to empty neighbors
        """
        new_moves = []

        if self.phase == 0:
            for i in range(BOARD_SIZE):
                if self.board[i] == 0:
                    new_moves.append([i, 0])
        else:
            for i in range(BOARD_SIZE):
                if self.board[i] == 1:  # goat
                    for dir_code, dest in self.move_map[i].items():
                        if dest is not None and self.board[dest] == 0:
                            new_moves.append([i, dir_code])

        self.valid_moves = new_moves

    # ========================================================
    # Reward Shaping definitions
    # ========================================================
    def _compute_unreachable_safe_cells(self) -> int:
        """
        Count how many cells are unreachable by any tiger if tigers
        can only walk through empty cells and onto empty cells.
        (Goats act as walls for this reachability notion.)
        """
        tiger_starts = [i for i in range(BOARD_SIZE) if self.board[i] == 2]

        if not tiger_starts:
            return int(np.sum((self.board == 0) | (self.board == 1)))

        visited = self._visited_buffer
        visited[:] = False

        q = deque()
        for t in tiger_starts:
            visited[t] = True
            q.append(t)

        while q:
            cur = q.popleft()
            for _, dest in self.move_map.get(cur, {}).items():
                if self.board[dest] == 0 and not visited[dest]:
                    visited[dest] = True
                    q.append(dest)

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

    # --------------------------------------------------------
    #  Shared render helper
    # --------------------------------------------------------
    def render_board(self, show_coords: bool = True, last_move_desc: str = ""):
        """
        ASCII render for either env variant. Optionally includes the
        last move description if provided.
        """
        symbols = {0: "0", 1: "G", 2: "T"}

        cells = [symbols[v] for v in self.board]
        goats_eaten = self.eaten
        phase_str = "Place" if self.phase == 0 else "Move"

        coord = (
            [f" {label} " for label in COORD_LABELS]
            if show_coords else [""] * BOARD_SIZE
        )

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

        status = (
            f"-----------------------------------------\n"
            f" Phase: {phase_str:<5} | Goats Eaten: {goats_eaten} | Turn: {self.turns}\n"
            f"-----------------------------------------"
        )

        if last_move_desc:
            status += f"\n Last move: {last_move_desc}"

        print(
            "\n"
            "============== TIGERS & GOATS BOARD ==============\n"
            f"{line0}\n{line1}\n{line2}\n{line3}\n{line4}\n{status}\n"
        )
