"""Read the solved tables: exact value of any position, perfect moves, draw and repetition detection.

    from perfect_play.oracle import PerfectPlay, Position, RepetitionTracker
    pp = PerfectPlay()                                   # tables in perfect_play/tables/
    pos = Position.from_env(env, mover="goat")           # or Position(tm, gm, captured, placed, mover)
    pp.evaluate(pos)          -> ("win" | "loss" | "draw" for the mover, plies to the end or None)
    pp.evaluate_with_limit(pos, plies_played)  -> the same under TnGEnv's 200-ply turn limit
    pp.best_action(pos)       -> env action id of a perfect move
    pp.predict_fn("tiger")    -> TnGEnv model-opponent callback (obs, mask) -> action

Values: every position carries its exact value and plies to the end with optimal play (winner fastest,
loser slowest): uint16 movement tables (mv_k*.u16) and uint8 placement tables (pl_n*.u8, from
solve_placement_dtm.py). If the .u8 files are missing, the 2-bit placement tables written by solve.py
(pl_n*.u2) are used instead; they hold the exact win/draw/loss value but no ply counts.
A "draw" means neither side can force a win: with perfect play the game goes on forever, which the
environment ends as MaxTimeout or RepeatTimeout. RepetitionTracker finds those repeating sections.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import GOATS, goat_moves, placement_offsets, popcount, rank_board, tiger_moves  # noqa: E402

TABLES = Path(__file__).resolve().parent / "tables"
RESULT = {0: "draw", 1: "win", 2: "loss"}


@dataclass(frozen=True)
class Position:
    tm: int            # tiger squares (bit mask)
    gm: int            # goat squares (bit mask)
    captured: int
    placed: int        # goats placed so far (15 in the movement phase)
    mover: str         # "goat" or "tiger"

    @property
    def placing(self):
        return self.placed < GOATS

    def key(self):
        return (self.tm, self.gm, self.captured, self.placed, self.mover)

    @staticmethod
    def from_obs(obs, mover):
        board = [int(x) for x in obs[:23]]
        tm = sum(1 << p for p, v in enumerate(board) if v == 2)
        gm = sum(1 << p for p, v in enumerate(board) if v == 1)
        captured = int(obs[23])
        placed = GOATS if int(obs[24]) == 1 else popcount(gm) + captured
        return Position(tm, gm, captured, placed, mover)

    @staticmethod
    def from_env(env, mover):
        return Position.from_obs(env.unwrapped.get_state(), mover)


class PerfectPlay:
    def __init__(self, tables=TABLES):
        tables = Path(tables)
        self.meta = json.loads((tables / "metadata.json").read_text())
        self.mv = {(k, s): np.memmap(tables / f"mv_k{k}_{s}.u16", np.uint16, "r") for k in range(10, 16) for s in "GT"}
        u8 = {(n, s): tables / f"pl_n{n}_{s}.u8" for n in range(GOATS) for s in "GT"}
        self.pl8 = ({key: np.memmap(path, np.uint8, "r") for key, path in u8.items()}
                    if all(path.exists() for path in u8.values()) else None)
        # 2-bit fallback (value only), read only when the .u8 placement tables are absent
        self.pl = (None if self.pl8 is not None else
                   {(n, s): np.memmap(tables / f"pl_n{n}_{s}.u2", np.uint8, "r") for n in range(GOATS) for s in "GT"})
        self.off = {n: placement_offsets(n) for n in range(GOATS)}

    # ── values ──────────────────────────────────────────────────────────────────────────────────────
    def evaluate(self, pos):
        """(result for the mover, plies to the result or None if unknown/draw)."""
        side = "G" if pos.mover == "goat" else "T"
        k = popcount(pos.gm)
        if pos.placing:
            i = self.off[pos.placed][pos.captured] + rank_board(pos.tm, pos.gm, k)
            if self.pl8 is not None:
                code = int(self.pl8[(pos.placed, side)][i])
                if code == 0:
                    return "draw", None
                return ("win" if (code - 1) & 1 else "loss"), code - 1      # the winner makes the last move
            return RESULT[(int(self.pl[(pos.placed, side)][i >> 2]) >> (2 * (i & 3))) & 3], None
        code = int(self.mv[(k, side)][rank_board(pos.tm, pos.gm, k)])
        if code == 0:
            return "draw", None
        return ("loss" if code & 1 else "win"), code >> 1

    def is_draw(self, pos):
        return self.evaluate(pos)[0] == "draw"

    def evaluate_with_limit(self, pos, plies_played, ply_limit=200):
        """Value in TnGEnv's game, which ends after `ply_limit` plies (max_turns = 100 turns of two plies).

        A forced result in d plies still happens iff plies_played + d <= ply_limit: the winner's fastest line
        needs d plies against any defence and the loser can always last d. Otherwise the game is cut off and
        counts as a draw. The same moves (best_action) are optimal with or without the limit. TnGEnv's
        repeat rule never interrupts a perfect player's win: along it the plies-to-go fall every ply, so no
        position repeats, and no position before it had the same (winning) value.
        Placement positions are always reached after the same number of plies: 2n (goats to move) or
        2n - 1 (tigers to move) when the goats move first, 2n + 1 or 2n when the tigers do."""
        result, d = self.evaluate(pos)
        if result == "draw" or d is None or plies_played + d <= ply_limit:
            return result, d
        return "draw", None

    def successors(self, pos):
        """[(action, next position or None if the move wins at once)]."""
        if pos.mover == "goat":
            nxt_placed = pos.placed + 1 if pos.placing else GOATS
            return [(a, Position(pos.tm, g2, pos.captured, nxt_placed, "tiger"))
                    for a, g2 in goat_moves(pos.tm, pos.gm, pos.placing)]
        out = []
        for a, t2, g2, took in tiger_moves(pos.tm, pos.gm):
            c2 = pos.captured + took
            out.append((a, None if c2 >= 6 else Position(t2, g2, c2, pos.placed, "goat")))
        return out

    def move_values(self, pos):
        """[(action, result for the mover after this move, plies to it or None)], best first."""
        rows = []
        for a, nxt in self.successors(pos):
            if nxt is None:
                rows.append((a, "win", 1))
                continue
            res, d = self.evaluate(nxt)
            mine = {"win": "loss", "loss": "win", "draw": "draw"}[res]
            rows.append((a, mine, None if d is None else d + 1))
        rank = {"win": 0, "draw": 1, "loss": 2}
        # wins: fastest first; losses: slowest first; unknown distances (only without the .u8 placement
        # tables) after known ones
        rows.sort(key=lambda r: (rank[r[1]], (r[2] if r[2] is not None else 10**6) if r[1] == "win"
                                 else -(r[2] if r[2] is not None else -1)))
        return rows

    def best_action(self, pos, seen=None):
        """A perfect move. With `seen` (a set of position keys), equal-valued moves to unseen positions
        are preferred, so drawn games wander instead of repeating."""
        rows = self.move_values(pos)
        best = [r for r in rows if r[1] == rows[0][1] and (rows[0][1] == "draw" or r[2] == rows[0][2])]
        if seen is not None and best[0][1] == "draw":
            fresh = [r for r in best if self._next_key(pos, r[0]) not in seen]
            best = fresh or best
        return best[0][0]

    def _next_key(self, pos, action):
        for a, nxt in self.successors(pos):
            if a == action:
                return None if nxt is None else nxt.key()
        return None

    def predict_fn(self, role):
        """Callback for TnGEnv(tiger_model_predict_fn=...) or goat_model_predict_fn: perfect opponent."""
        def predict(obs, mask):
            return self.best_action(Position.from_obs(obs, role))
        return predict


class RepetitionTracker:
    """Find repeated positions and long repeating sections in one game.

    update(pos, action) is called once per ply with the position *before* the move.
      repeats            plies whose position already occurred earlier in the game
      longest_repeat_run longest run of consecutive plies that only revisit earlier positions
      longest_shuffle    longest run of plies in which each move undoes that side's previous move
      cycle              (first ply, period) of the first repetition, if any
      max_occurrences    most times any single position occurred
    """
    def __init__(self):
        self.first_seen, self.counts = {}, {}
        self.repeats = self.run = self.longest_repeat_run = 0
        self.shuffle = self.longest_shuffle = 0
        self.cycle = None
        self.last_move = {}
        self.ply = 0

    def update(self, pos, action, next_pos=None):
        key = pos.key()
        self.counts[key] = self.counts.get(key, 0) + 1
        if key in self.first_seen:
            self.repeats += 1
            self.run += 1
            if self.cycle is None:
                self.cycle = (self.first_seen[key], self.ply - self.first_seen[key])
        else:
            self.first_seen[key] = self.ply
            self.run = 0
        self.longest_repeat_run = max(self.longest_repeat_run, self.run)
        # a "shuffle" move takes a piece back to where that side's previous move took it from
        if not pos.placing and next_pos is not None:
            before = pos.gm if pos.mover == "goat" else pos.tm
            after = next_pos.gm if pos.mover == "goat" else next_pos.tm
            moved = (before & ~after, after & ~before)        # (from, to) squares
            prev = self.last_move.get(pos.mover)
            if prev is not None and moved == (prev[1], prev[0]):
                self.shuffle += 1
            else:
                self.shuffle = 0
            self.longest_shuffle = max(self.longest_shuffle, self.shuffle)
            self.last_move[pos.mover] = moved
        self.ply += 1

    @property
    def max_occurrences(self):
        return max(self.counts.values(), default=0)

    def summary(self):
        return dict(plies=self.ply, repeated_plies=self.repeats, longest_repeat_run=self.longest_repeat_run,
                    longest_shuffle=self.longest_shuffle, first_cycle=self.cycle,
                    max_occurrences=self.max_occurrences)
