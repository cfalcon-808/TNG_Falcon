"""Game model for the visualizer: moves, history, end-of-game rules, and perfect-play analysis.

Rules and move generation come from perfect_play/rules.py, which reads the board graph from env_tng.py
(proved equal to TnGEnv by perfect_play/prove_rules.py). Values come from the solved tables through
perfect_play/oracle.py.
"""

from __future__ import annotations

import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, REPO / "perfect_play"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import env_tng  # noqa: E402
from oracle import PerfectPlay, Position  # noqa: E402
from rules import (CAPTURES_TO_WIN, GOATS, L, MOVE_MAP, N, SQUARES, goat_moves, popcount,  # noqa: E402
                   tiger_moves)

GOAT, TIGER = "goat", "tiger"
OTHER = {GOAT: TIGER, TIGER: GOAT}
SIDE_NAME = {GOAT: "Goats", TIGER: "Tigers"}
COORD_LABELS = list(env_tng.COORD_LABELS)
TIGER_START = tuple(env_tng.TIGER_START_POSITIONS)
PLY_LIMIT = 2 * int(env_tng.DEFAULT_KNOBS["MAX_TURNS"])      # 100 turns of a goat move and a tiger move
MAX_REPEATS = int(env_tng.DEFAULT_KNOBS["MAX_REPEATS"])      # the 4th repeat of a board ends the game
DEFAULT_TABLES = REPO / "perfect_play" / "tables"
RANK = {"win": 0, "draw": 1, "loss": 2}
FLIP = {"win": "loss", "loss": "win", "draw": "draw"}


def square_name(sq: int, labels: str = "names") -> str:
    return str(sq) if labels == "numbers" else COORD_LABELS[sq]


def edges() -> List[Tuple[int, int]]:
    out = set()
    for a, nbrs in MOVE_MAP.items():
        for b in nbrs.values():
            if b is not None:
                out.add((min(a, b), max(a, b)))
    return sorted(out)


def squares(mask: int) -> List[int]:
    return [p for p in range(SQUARES) if (mask >> p) & 1]


def start_position(first: str = GOAT) -> Position:
    return Position(sum(1 << p for p in TIGER_START), 0, 0, 0, first)


# ── moves ───────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Move:
    action: int                    # TnGEnv action id: 5 * square + direction (0 for a placement)
    side: str
    frm: Optional[int]             # None for a placement
    to: int
    jumped: Optional[int]          # the captured goat's square
    nxt: Optional[Position]        # None: this move captures the 6th goat and wins at once

    def text(self, labels: str = "names") -> str:
        if self.frm is None:
            return "@" + square_name(self.to, labels)
        sep = "x" if self.jumped is not None else "-"
        return f"{square_name(self.frm, labels)}{sep}{square_name(self.to, labels)}"


def legal_moves(pos: Position) -> List[Move]:
    out = []
    if pos.mover == GOAT:
        placed = pos.placed + 1 if pos.placing else GOATS
        for a, g2 in goat_moves(pos.tm, pos.gm, pos.placing):
            p, d = divmod(a, 5)
            frm, to = (None, p) if pos.placing else (p, int(N[p, d]))
            out.append(Move(a, GOAT, frm, to, None, Position(pos.tm, g2, pos.captured, placed, TIGER)))
        return out
    for a, t2, g2, took in tiger_moves(pos.tm, pos.gm):
        p, d = divmod(a, 5)
        c2 = pos.captured + took
        nxt = None if c2 >= CAPTURES_TO_WIN else Position(t2, g2, c2, pos.placed, GOAT)
        if took:
            out.append(Move(a, TIGER, p, int(L[p, d]), int(N[p, d]), nxt))
        else:
            out.append(Move(a, TIGER, p, int(N[p, d]), None, nxt))
    return out


def plies_before(pos: Position, first: str) -> int:
    """Plies played before this position in a game from the standard start. Fixed in the placement phase;
    in the movement phase this is the earliest possible count (no movement moves yet)."""
    n = pos.placed
    if first == GOAT:
        p = 2 * n if pos.mover == GOAT else 2 * n - 1
    else:
        p = 2 * n if pos.mover == TIGER else 2 * n + 1
    return max(0, p)


# ── position strings ────────────────────────────────────────────────────────────────────────────────
def position_to_text(pos: Position) -> str:
    t = ",".join(map(str, squares(pos.tm)))
    g = ",".join(map(str, squares(pos.gm)))
    return f"tigers={t} goats={g} captured={pos.captured} mover={pos.mover}"


def _parse_squares(text: str) -> List[int]:
    out = []
    for tok in re.split(r"[,\s;]+", text.strip()):
        if not tok:
            continue
        if tok.isdigit():
            sq = int(tok)
        elif tok.lower() in COORD_LABELS:
            sq = COORD_LABELS.index(tok.lower())
        else:
            raise ValueError(f"unknown square {tok!r}")
        if not 0 <= sq < SQUARES:
            raise ValueError(f"square {sq} is off the board (0-{SQUARES - 1})")
        out.append(sq)
    return out


def parse_position(text: str) -> Position:
    """Parse 'tigers=0,3,4 goats=2,8 captured=0 mover=goat' (squares by number or name, keys in any
    order, 'mover' may be goat/goats/g/tiger/tigers/t). The goats placed follow from the goats on the
    board plus the captured goats."""
    text = text.strip().strip("`'\"")
    fields: Dict[str, str] = {}
    for m in re.finditer(r"(tigers?|goats?|captured|mover|to_move|placed)\s*[=:]\s*([^=:]*?)(?=\s+\w+\s*[=:]|$)",
                         text, flags=re.I):
        fields[m.group(1).lower().rstrip("s")] = m.group(2).strip()
    if "tiger" not in fields:
        raise ValueError("expected e.g. tigers=0,3,4 goats=2,8 captured=0 mover=goat")
    tigers = _parse_squares(fields["tiger"])
    goats = _parse_squares(fields.get("goat", ""))
    captured = int(fields.get("captured", "0") or 0)
    mover_text = (fields.get("mover") or fields.get("to_move") or "goat").lower()
    if mover_text in ("g", "goat", "goats"):
        mover = GOAT
    elif mover_text in ("t", "tiger", "tigers"):
        mover = TIGER
    else:
        raise ValueError(f"mover must be goat or tiger, not {mover_text!r}")
    if len(tigers) != 3 or len(set(tigers)) != 3:
        raise ValueError("need exactly 3 different tiger squares")
    if len(set(goats)) != len(goats) or set(goats) & set(tigers):
        raise ValueError("goat squares must be different and not under a tiger")
    if not 0 <= captured < CAPTURES_TO_WIN:
        raise ValueError(f"captured must be 0-{CAPTURES_TO_WIN - 1}")
    placed = len(goats) + captured
    if placed > GOATS:
        raise ValueError(f"goats on the board plus captured exceed {GOATS}")
    if "placed" in fields and int(fields["placed"]) != placed:
        raise ValueError(f"placed must equal goats on the board + captured ({placed})")
    return Position(sum(1 << p for p in tigers), sum(1 << p for p in goats), captured, placed, mover)


# ── the game ───────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Outcome:
    winner: Optional[str]          # GOAT, TIGER or None for a draw
    reason: str


class Game:
    """A root position plus the moves played from it; `ply` indexes the position shown."""

    def __init__(self, root: Position, first: str = GOAT) -> None:
        self.root = root
        self.first = first
        self.root_plies = plies_before(root, first)
        self.moves: List[Move] = []
        self.positions: List[Position] = [root]
        self.ply = 0

    @property
    def pos(self) -> Position:
        return self.positions[self.ply]

    def plies_played(self, i: Optional[int] = None) -> int:
        return self.root_plies + (self.ply if i is None else i)

    def push(self, move: Move) -> None:
        """Play `move` at the current ply, dropping any moves after it."""
        del self.moves[self.ply:]
        del self.positions[self.ply + 1:]
        self.moves.append(move)
        self.positions.append(move.nxt if move.nxt is not None else self._after_win(move))
        self.ply += 1

    def _after_win(self, move: Move) -> Position:
        """Board after the winning 6th capture (not in the tables; only drawn)."""
        pos = self.positions[self.ply]
        tm = pos.tm ^ (1 << move.frm) ^ (1 << move.to)
        return Position(tm, pos.gm ^ (1 << move.jumped), pos.captured + 1, pos.placed, GOAT)

    def seen(self, upto: Optional[int] = None) -> set:
        upto = self.ply if upto is None else upto
        return {p.key() for p in self.positions[:upto + 1]}

    def repeats(self, i: Optional[int] = None) -> int:
        """How often position i occurred so far, counting from the first move (TnGEnv's state history)."""
        i = self.ply if i is None else i
        key = self.positions[i].key()
        return sum(1 for p in self.positions[1:i + 1] if p.key() == key)

    def outcome(self, i: Optional[int] = None, limits: bool = True) -> Optional[Outcome]:
        i = self.ply if i is None else i
        if i > 0 and self.moves[i - 1].nxt is None:
            return Outcome(TIGER, f"the {CAPTURES_TO_WIN}th goat is captured")
        pos = self.positions[i]
        if not legal_moves(pos):
            return Outcome(OTHER[pos.mover], f"the {SIDE_NAME[pos.mover].lower()} have no legal move")
        if limits:
            if self.plies_played(i) >= PLY_LIMIT:
                return Outcome(None, f"turn limit ({PLY_LIMIT // 2} turns, {PLY_LIMIT} plies)")
            # TnGEnv (goats first) counts the board after every tiger reply; the 4th occurrence ends it
            if self.first == GOAT and pos.mover == GOAT and i > 0 and self.repeats(i) >= MAX_REPEATS:
                return Outcome(None, f"repeat rule ({MAX_REPEATS}th occurrence of this board)")
        return None


# ── analysis ───────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Line:
    move: Move
    result: str                    # for the side to move: win / draw / loss
    d: Optional[int]               # plies to the end with perfect play, from the current position
    over_limit: bool = False       # a forced result that TnGEnv's turn limit cuts off (a draw there)
    pv: Optional[List[Move]] = None  # perfect continuation after `move`, filled in lazily
    traps: int = 0                 # for a drawing move: opponent replies that lose
    replies: int = 0               # the opponent's legal replies

    @property
    def limited_result(self) -> str:
        return "draw" if self.over_limit else self.result


@dataclass
class Analysis:
    pos: Position
    result: str                    # value for the side to move
    d: Optional[int]
    over_limit: bool
    lines: List[Line] = field(default_factory=list)   # every legal move, best first

    @property
    def best(self) -> List[Line]:
        """All moves that are exactly as good as the best one."""
        if not self.lines:
            return []
        top = self.lines[0]
        return [ln for ln in self.lines if ln.result == top.result and (top.result == "draw" or ln.d == top.d)]

    def counts(self) -> Dict[str, int]:
        out = {"win": 0, "draw": 0, "loss": 0}
        for ln in self.lines:
            out[ln.result] += 1
        return out


def sort_key(ln: Line):
    """Fastest win first, slowest loss last-but-best; draws by the share of the opponent's replies that lose."""
    if ln.result == "win":
        return (0, ln.d, 0.0, ln.move.action)
    if ln.result == "loss":
        return (2, -ln.d, 0.0, ln.move.action)
    return (1, 0, -(ln.traps / ln.replies if ln.replies else 0.0), ln.move.action)


class Analyzer:
    """Exact values from the solved tables, cached per position."""

    def __init__(self, tables: Path = DEFAULT_TABLES) -> None:
        self.pp = PerfectPlay(tables)
        if self.pp.pl8 is None:
            raise FileNotFoundError(f"the placement ply-count tables (pl_n*.u8) are missing from {tables}")
        self.cache: Dict[tuple, Tuple[str, Optional[int]]] = {}

    def value(self, pos: Position) -> Tuple[str, Optional[int]]:
        key = pos.key()
        hit = self.cache.get(key)
        if hit is None:
            if len(self.cache) > 500_000:
                self.cache.clear()
            hit = self.cache[key] = self.pp.evaluate(pos)
        return hit

    def lines(self, pos: Position, plies_played: int, limits: bool, traps: bool = True) -> List[Line]:
        """Every legal move with its exact result, best first. With `traps`, drawing moves also count the
        opponent replies that would lose, and draws are ordered by that share."""
        out = []
        for mv in legal_moves(pos):
            if mv.nxt is None:
                result, d = "win", 1
            else:
                r, dn = self.value(mv.nxt)
                result, d = FLIP[r], (None if dn is None else dn + 1)
            over = limits and d is not None and plies_played + d > PLY_LIMIT
            ln = Line(mv, result, d, over)
            if traps and result == "draw":
                replies = legal_moves(mv.nxt)
                ln.replies = len(replies)
                ln.traps = sum(1 for r in replies if r.nxt is not None and self.value(r.nxt)[0] == "win")
            out.append(ln)
        out.sort(key=sort_key)
        return out

    def analyse(self, pos: Position, plies_played: int, limits: bool) -> Analysis:
        result, d = self.value(pos)
        over = limits and d is not None and plies_played + d > PLY_LIMIT
        return Analysis(pos, result, d, over, self.lines(pos, plies_played, limits))

    def choose(self, analysis: Analysis, seen: set, rng: random.Random) -> Move:
        """A perfect move: the fastest win, the slowest loss, or a draw. Among draws it prefers positions not
        visited yet, then the largest share of opponent replies that lose."""
        best = analysis.best
        if best and best[0].result == "draw":
            fresh = [ln for ln in best if ln.move.nxt is not None and ln.move.nxt.key() not in seen]
            best = fresh or best
            share = max(ln.traps / ln.replies if ln.replies else 0.0 for ln in best)
            best = [ln for ln in best if (ln.traps / ln.replies if ln.replies else 0.0) == share]
        return rng.choice(best).move

    def pv(self, line: Line, plies_played: int, seen: set, length: int, rng: random.Random) -> Iterator[None]:
        """Fill line.pv with the perfect continuation after line.move, yielding between plies."""
        pv: List[Move] = []
        line.pv = pv
        seen = set(seen)
        pos = line.move.nxt
        ply = plies_played + 1
        while pos is not None and len(pv) < length:
            an = Analysis(pos, *self.value(pos), False, self.lines(pos, ply, False, traps=False))
            if not an.lines:
                break
            mv = self.choose(an, seen, rng)
            pv.append(mv)
            seen.add(pos.key())
            pos = mv.nxt
            ply += 1
            yield


def describe(result: str, d: Optional[int], mover: str) -> str:
    """'Goats win in 37 plies' style text for a value seen from the side to move."""
    if result == "draw":
        return "Draw with perfect play"
    winner = mover if result == "win" else OTHER[mover]
    return f"{SIDE_NAME[winner]} can force a win in {d} {'ply' if d == 1 else 'plies'}"


def pill_text(result: str, d: Optional[int]) -> str:
    if result == "draw":
        return "="
    return f"{'W' if result == 'win' else 'L'}{d}"


def goat_count_text(pos: Position) -> str:
    on_board = popcount(pos.gm)
    if pos.placing:
        return f"placement · {pos.placed} of {GOATS} goats placed · {pos.captured} captured"
    return f"movement · {on_board} goats on the board · {pos.captured} captured"


__all__ = [
    "Analysis", "Analyzer", "COORD_LABELS", "DEFAULT_TABLES", "GOAT", "GOATS", "Game", "Line", "MOVE_MAP",
    "Move", "OTHER", "Outcome", "PLY_LIMIT", "Position", "SIDE_NAME", "TIGER", "describe", "edges",
    "goat_count_text", "legal_moves", "parse_position", "pill_text", "plies_before", "position_to_text",
    "square_name", "squares", "start_position",
]

