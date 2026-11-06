"""Exhaustive proof that the solver's rules (rules.py) are TnGEnv's rules, on every board.

usage: python perfect_play/prove_rules.py            (about 11 minutes on 16 threads; writes tables/rules_proof.json)

The move generators of both programs are loops over pieces. Inside one piece's iteration the code reads a few
board squares and appends that piece's moves; iterations share no state except the list they append to
(see TnGEnv._tiger_moves, TnGEnv._update_valid_moves, rules.tiger_moves, rules.goat_moves). So:

1. Locality, checked mechanically. Every board read is recorded with the piece being processed: TnGEnv's
   board is replaced by an array that logs each square read together with the loop variable `i` of the
   calling frame, and rules.py's bit masks by an int that logs each bit tested with the loop variable `p`.
   For every square p and EVERY assignment of {empty, goat, tiger} to the squares near p (graph distance 2 for
   a tiger, plus the hub landings from square 0; distance 1 for a goat), with at most 3 tigers and the rest of
   the board empty, we check that piece p's iteration reads only squares inside that ball. A deterministic iteration that only ever reads squares in a set S computes
   the same thing on any two boards that agree on S (induction over its reads). So piece p's moves are a
   function of the ball's contents, on every board.
2. Equality on every local pattern: for every such assignment, piece p's moves in TnGEnv and in rules.py
   are compared: action id, landing square, capture flag and the goat removed (TnGEnv._find_jumped_goat).
   With 1, every board's full move list is the union of these, so the two agree on every board.
3. State and transitions: the constants, every line of env_tng.py that writes the board, the captured
   count, the goats-placed count or the phase (found by parsing the file), and the terminal rules.
4. Indexing, every index: rules.rank_board / unrank_board (the same Python functions, compiled with numba
   with math.comb replaced by a checked table) round-trip every index of every block k = 0..15 to a valid
   board. Unrank is then injective into the valid boards, and the block holds exactly C(23,3) * C(20,k) of
   them, so the indexing is a bijection.
5. The independent checker (check_all.py, written from env_tng.py alone) has the same indexing and the same
   moves and resulting boards as rules.py on every one of the 1.85 billion boards, so its full check of the
   tables is a check under TnGEnv's rules.
"""
import ast
import itertools
import json
import sys
import time
import types
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402
from rules import env_tng  # noqa: E402

ENV_FILE = Path(env_tng.__file__)
report = {}


# ── tracing ─────────────────────────────────────────────────────────────────────────────────────────
class TracedBoard(np.ndarray):
    """A board that logs every scalar read as (function, value of its loop variable `i`, square)."""
    log = None

    def __getitem__(self, key):
        if TracedBoard.log is not None and isinstance(key, (int, np.integer)):
            f = sys._getframe(1)
            TracedBoard.log.append((f.f_code.co_name, f.f_locals.get("i"), int(key)))
        return super().__getitem__(key)


class TracedMask(int):
    """A bit mask that logs every bit tested with `mask >> s` as (function, loop variable `p`, s)."""
    log = None

    def __rshift__(self, s):
        if TracedMask.log is not None:
            f = sys._getframe(1)
            TracedMask.log.append((f.f_code.co_name, f.f_locals.get("p"), int(s)))
        return int(self) >> s

    def __or__(self, o):
        return TracedMask(int(self) | int(o))

    def __xor__(self, o):
        return TracedMask(int(self) ^ int(o))

    def __and__(self, o):
        return TracedMask(int(self) & int(o))

    __ror__, __rxor__, __rand__ = __or__, __xor__, __and__


env = env_tng.TnGEnv(learner_role="goat")
MOVE_MAP = {p: dict(n) for p, n in env.move_map.items()}
HUB = dict(env_tng.HUB0_CAPTURE_JUMP)


def ball(p, radius=2):
    seen, frontier = {p}, {p}
    for _ in range(radius):
        frontier = {q for x in frontier for q in MOVE_MAP[x].values() if q is not None} - seen
        seen |= frontier
    if p == 0:
        seen |= set(HUB.values())
    return sorted(seen - {p})


def local_patterns(p, piece, max_other_tigers, radius=2):
    cells = ball(p, radius)
    for values in itertools.product((0, 1, 2), repeat=len(cells)):
        if values.count(2) > max_other_tigers:
            continue
        board = np.zeros(23, np.int8)
        board[p] = piece
        board[cells] = values
        yield cells, board


def masks(board):
    tm = sum(1 << q for q in range(23) if board[q] == 2)
    gm = sum(1 << q for q in range(23) if board[q] == 1)
    return tm, gm


def bit(x):
    assert x and not x & (x - 1)
    return x.bit_length() - 1


def env_reads(fn, p):
    return {sq for name, i, sq in TracedBoard.log if name == fn and i == p}


def rules_reads(fn, p):
    return {s for name, q, s in TracedMask.log if name == fn and q == p}


# ── 1 + 2. every local pattern of every piece ───────────────────────────────────────────────────────
t0 = time.time()
summary = {"tiger": {}, "goat_slide": {}, "goat_place": {}}
patterns = 0
discrepancies = {}
for p in range(23):
    allowed = set(ball(p)) | {p}
    allowed_goat = set(ball(p, 1)) | {p}
    # tigers
    read_env, read_rules = set(), set()
    for cells, board in local_patterns(p, 2, 2):
        patterns += 1
        env.board = board.copy().view(TracedBoard)
        TracedBoard.log = []
        env_moves = []
        for f, to, cap, d in env._tiger_moves(include_dir=True):
            if f != p:
                continue
            # what both TnGEnv transitions do with a capture: remove _find_jumped_goat(from, to) if a goat is there
            j = env._find_jumped_goat(f, to) if cap else None
            removed = j if j is not None and board[j] == 1 else None
            env_moves.append((5 * f + d, to, bool(cap), removed))
        env_moves.sort()
        reads = env_reads("_tiger_moves", p)
        TracedBoard.log = None
        assert reads <= allowed, (p, reads - allowed)
        read_env |= reads
        tm, gm = masks(board)
        TracedMask.log = []
        out = rules.tiger_moves(TracedMask(tm), TracedMask(gm))
        reads = rules_reads("tiger_moves", p)
        TracedMask.log = None
        assert reads <= allowed, (p, reads - allowed)
        read_rules |= reads
        rule_moves = []
        for a, t2, g2, took in out:
            if a // 5 != p:
                continue
            t2, g2 = int(t2), int(g2)
            assert t2 & ~int(tm) and (int(tm) & ~t2) == 1 << p
            rule_moves.append((a, bit(t2 & ~tm), bool(took), bit(gm & ~g2) if took else None))
            assert (g2 == gm) != took and (not took or g2 | (gm & ~g2) == gm)
        rule_moves.sort()
        # legality (action, landing, capture flag) and the capture's effect (goat removed) are compared apart
        assert [m[:3] for m in env_moves] == [m[:3] for m in rule_moves], (p, board.tolist(), env_moves, rule_moves)
        for e, r in zip(env_moves, rule_moves):
            if e != r:
                key = f"tiger {p} action {e[0]} (to {e[1]}): TnGEnv removes {e[3]}, rules.py removes {r[3]}"
                row = discrepancies.setdefault(key, dict(patterns=0, example=None))
                row["patterns"] += 1
                row["example"] = row["example"] or board.tolist()
    summary["tiger"][p] = dict(env_reads=sorted(read_env - {p}), rules_reads=sorted(read_rules - {p}))
    # goats, movement phase
    read_env, read_rules = set(), set()
    for cells, board in local_patterns(p, 1, 3, radius=1):
        patterns += 1
        env.board = board.copy().view(TracedBoard)
        env.phase = 1
        TracedBoard.log = []
        env._update_valid_moves()
        reads = env_reads("_update_valid_moves", p)
        TracedBoard.log = None
        assert reads <= allowed_goat, (p, reads - allowed_goat)
        read_env |= reads
        env_moves = sorted((a, MOVE_MAP[p][a % 5]) for a in env.valid_moves if a // 5 == p)
        assert sorted(int(a) for a in np.flatnonzero(env.get_action_mask()) if a // 5 == p) == [a for a, _ in env_moves]
        tm, gm = masks(board)
        TracedMask.log = []
        out = rules.goat_moves(TracedMask(tm), TracedMask(gm), False)
        reads = rules_reads("goat_moves", p)
        TracedMask.log = None
        assert reads <= allowed_goat, (p, reads - allowed_goat)
        read_rules |= reads
        rule_moves = []
        for a, g2 in out:
            if a // 5 != p:
                continue
            g2 = int(g2)
            assert (gm & ~g2) == 1 << p
            dest = bit(g2 & ~gm)
            assert g2 == gm ^ (1 << p) ^ (1 << dest)
            rule_moves.append((a, dest))
        assert env_moves == sorted(rule_moves), (p, board.tolist(), env_moves, rule_moves)
    summary["goat_slide"][p] = dict(env_reads=sorted(read_env - {p}), rules_reads=sorted(read_rules - {p}))
    # goats, placement phase: square p's own content decides whether 5p is legal
    read_env, read_rules = set(), set()
    for cells, board in local_patterns(p, 0, 3, radius=1):
        for own in (0, 1, 2):
            patterns += 1
            board[p] = own
            env.board = board.copy().view(TracedBoard)
            env.phase = 0
            TracedBoard.log = []
            env._update_valid_moves()
            reads = env_reads("_update_valid_moves", p)
            TracedBoard.log = None
            assert reads <= {p}, (p, reads)
            read_env |= reads
            env_moves = sorted((a, p) for a in env.valid_moves if a // 5 == p)
            tm, gm = masks(board)
            TracedMask.log = []
            out = rules.goat_moves(TracedMask(tm), TracedMask(gm), True)
            reads = rules_reads("goat_moves", p)
            TracedMask.log = None
            read_rules |= reads
            rule_moves = [(a, bit(int(g2) & ~gm)) for a, g2 in out if a // 5 == p]
            assert all(int(g2) == gm | (1 << p) for a, g2 in out if a // 5 == p)
            assert env_moves == sorted(rule_moves), (p, board.tolist(), env_moves, rule_moves)
        break                      # the ball's contents cannot matter here: p's iteration reads only p
    summary["goat_place"][p] = dict(env_reads=sorted(read_env - {p}), rules_reads=sorted(read_rules - {p}))
    print(f"square {p:2d}: ball of {len(ball(p))} squares; tiger reads {summary['tiger'][p]['env_reads']}", flush=True)

# both programs read exactly the same squares for each piece
for kind in summary:
    for p, s in summary[kind].items():
        assert s["env_reads"] == s["rules_reads"], (kind, p, s)
n_disc = sum(d["patterns"] for d in discrepancies.values())
report["local_patterns"] = dict(patterns=patterns, legality_mismatches=0, capture_effect_mismatches=n_disc,
                                discrepancies=discrepancies, seconds=round(time.time() - t0, 1),
                                squares_read_per_piece={k: {p: v["env_reads"] for p, v in d.items()}
                                                        for k, d in summary.items()})
print(f"1-2. {patterns:,} local patterns: reads stay inside the ball; legal moves agree on all of them; "
      f"capture effects differ in {n_disc:,} ({time.time() - t0:.0f}s)", flush=True)
for key, row in discrepancies.items():
    print("   DISCREPANCY", key, f"in {row['patterns']:,} local patterns, e.g. board {row['example']}")

# ── 3. constants, state writes and terminal rules ───────────────────────────────────────────────────
assert (env_tng.BOARD_SIZE, env_tng.DIR_CODES, env_tng.TOTAL_GOATS_TO_PLACE, env_tng.GOATS_EATEN_FOR_TIGER_WIN) == \
       (rules.SQUARES, 5, rules.GOATS, rules.CAPTURES_TO_WIN)
assert list(env_tng.TIGER_START_POSITIONS) == [0, 3, 4]
for p in range(23):                                       # the graph rules.py builds is the env's graph
    for d in range(5):
        q = MOVE_MAP[p].get(d)
        assert rules.N[p, d] == (-1 if q is None else q)
assert all(env.encode_action(p, d) == 5 * p + d for p in range(23) for d in range(5))

tree = ast.parse(ENV_FILE.read_text())
writes = []
for fn in ast.walk(tree):
    if not isinstance(fn, ast.FunctionDef):
        continue
    aliases = {t.id for node in ast.walk(fn) if isinstance(node, ast.Assign) for t in node.targets
               if isinstance(t, ast.Name) and ast.unparse(node.value) == "self.board"}
    for node in ast.walk(fn):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        for t in targets:
            text = ast.unparse(t)
            if text in ("self.eaten", "self.goats_placed", "self.phase") or (
                    isinstance(t, ast.Subscript) and ast.unparse(t.value) in {"self.board", *aliases}):
                writes.append((fn.name, node.lineno, ast.unparse(node)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                ast.unparse(node.func.value) in {"self.board", *aliases} and node.func.attr not in ("tobytes", "copy"):
            writes.append((fn.name, node.lineno, ast.unparse(node)))
expected_functions = {"__init__", "reset", "_step_goat_transition", "_step_tiger_transition"}
assert {w[0] for w in writes} == expected_functions, writes
report["state_writes"] = [f"{name} line {line}: {code}" for name, line, code in writes]
report["transition_and_terminal_rules"] = [
    "goat placement (both learner paths): board[pos] = 1, goats_placed += 1, phase = 1 once goats_placed >= 15 "
    "-> rules: gm | 1 << p, placed + 1 (placing while placed < 15)",
    "goat slide: dest = move_map[pos][dir]; board[pos] = 0, board[dest] = 1 -> rules: gm ^ 1 << p ^ 1 << N[p, d]",
    "tiger move: (dest, capture) from _tiger_moves keyed by (from, dir); capture removes _find_jumped_goat(from, dest)"
    " (checked above to be the goat the rules remove) and eaten += 1",
    "tigers win when eaten >= 6 right after the capture (both paths) -> rules: the 6th capture wins at once",
    "goats win when the tigers have no legal move on their turn (goat path: after the goat move; tiger path: after "
    "the goat reply) -> rules: a mover with no legal move loses",
    "tigers win when the goats have no legal move (tiger path: goat_random_action returns None) -> same rule; the "
    "goat-learner path has no such check (its mask is empty and any action is invalid)",
    "not in the solved game: the turn limit (MaxTimeout after max_turns) and RepeatTimeout (goat path, 4th "
    "occurrence of a board after the tiger reply); see README for how ply counts handle the turn limit",
    "observation -> position: placed = 15 in the movement phase, else goats on board + eaten, because the only "
    "writes (listed) keep goats_placed = goats on board + eaten during placement",
]
print(f"3. constants, graph and {len(writes)} state-writing lines checked", flush=True)

# ── 4. indexing: every index of every block ─────────────────────────────────────────────────────────
import numba  # noqa: E402

BIN = np.array([[comb(n, k) for k in range(24)] for n in range(24)], np.int64)


@numba.njit(cache=False)
def table_comb(n, k):
    return BIN[n, k]


def compiled(fn):
    g = dict(fn.__globals__)
    g["comb"] = table_comb
    return numba.njit(types.FunctionType(fn.__code__, g, fn.__name__, fn.__defaults__, fn.__closure__))


rank_nb, unrank_nb = compiled(rules.rank_board), compiled(rules.unrank_board)


@numba.njit(parallel=True)
def roundtrip(k, size):
    bad = np.zeros(numba.config.NUMBA_NUM_THREADS * 64, np.int64)
    chunk = (size + bad.size - 1) // bad.size
    for c in numba.prange(bad.size):
        for i in range(c * chunk, min(size, (c + 1) * chunk)):
            tm, gm = unrank_nb(i, k)
            ok = tm < (1 << 23) and gm < (1 << 23) and (tm & gm) == 0
            t, g, n_t, n_g = tm, gm, 0, 0
            while t:
                t &= t - 1
                n_t += 1
            while g:
                g &= g - 1
                n_g += 1
            if not (ok and n_t == 3 and n_g == k and rank_nb(tm, gm, k) == i):
                bad[c] += 1
    return bad.sum()


t0 = time.time()
assert all(table_comb(n, k) == comb(n, k) for n in range(24) for k in range(24))
for _ in range(2000):                              # the compiled functions are the Python functions
    k = np.random.randint(0, 16)
    i = np.random.randint(0, rules.block_size(k))
    assert unrank_nb(i, k) == rules.unrank_board(i, k) and rank_nb(*rules.unrank_board(i, k), k) == i
index_rows = {}
for k in range(16):
    size = rules.block_size(k)
    assert size == comb(23, 3) * comb(20, k)
    bad = int(roundtrip(k, size))
    assert bad == 0, (k, bad)
    index_rows[k] = size
total = sum(index_rows.values())
report["indexing"] = dict(blocks=index_rows, indices_checked=total, failures=0, seconds=round(time.time() - t0, 1))
print(f"4. indexing: all {total:,} indices of blocks k = 0..15 round-trip to valid boards ({time.time() - t0:.0f}s)")

# ── 5. the independent checker (check_all.py) uses the same rules and indexing, on every board ─────────
# check_all.py was written from env_tng.py without reading rules.py or the GPU code. Its verdict covers
# TnGEnv's game once its move generator and indexing equal rules.py's (proved equal to TnGEnv above) on every
# board. rules.py's own functions are compiled with numba here, unchanged.
import check_all as ca  # noqa: E402

tiger_nb, goat_nb = compiled(rules.tiger_moves), compiled(rules.goat_moves)


@numba.njit
def same_moves(a, b):
    """a: rules.py rows (action, tm, gm[, took]); b: check_all rows (action, tm, gm, took)."""
    if len(a) != b.shape[0]:
        return False
    acts = np.empty(len(a), np.int64)
    for j in range(len(a)):
        acts[j] = a[j][0]
    oa, ob = np.argsort(acts), np.argsort(b[:, 0])
    for j in range(len(a)):
        if acts[oa[j]] != b[ob[j], 0]:
            return False
    return True


@numba.njit
def same_tiger(tm, gm):
    a, b = tiger_nb(tm, gm), ca.tiger_successors(tm, gm)
    if not same_moves(a, b):
        return False
    acts = np.empty(len(a), np.int64)
    for j in range(len(a)):
        acts[j] = a[j][0]
    oa, ob = np.argsort(acts), np.argsort(b[:, 0])
    for j in range(len(a)):
        x, y = a[oa[j]], b[ob[j]]
        if x[1] != y[1] or x[2] != y[2] or (1 if x[3] else 0) != y[3]:
            return False
    return True


@numba.njit
def same_goat(tm, gm, placing):
    a, b = goat_nb(tm, gm, placing), ca.goat_successors(tm, gm, placing)
    if not same_moves(a, b):
        return False
    acts = np.empty(len(a), np.int64)
    for j in range(len(a)):
        acts[j] = a[j][0]
    oa, ob = np.argsort(acts), np.argsort(b[:, 0])
    for j in range(len(a)):
        x, y = a[oa[j]], b[ob[j]]
        if y[1] != tm or x[1] != y[2]:
            return False
    return True


@numba.njit(parallel=True)
def every_board(k, size):
    bad = np.zeros((numba.config.NUMBA_NUM_THREADS * 64, 3), np.int64)
    chunk = (size + bad.shape[0] - 1) // bad.shape[0]
    for c in numba.prange(bad.shape[0]):
        for i in range(c * chunk, min(size, (c + 1) * chunk)):
            tm, gm = unrank_nb(i, k)
            t2, g2 = ca.unrank(i, k)
            if t2 != tm or g2 != gm or ca.rank(tm, gm, k) != i:
                bad[c, 0] += 1
            if not same_tiger(tm, gm):
                bad[c, 1] += 1
            if (k <= 14 and not same_goat(tm, gm, True)) or (k >= 10 and not same_goat(tm, gm, False)):
                bad[c, 2] += 1
    return bad.sum(axis=0)


t0 = time.time()
rows = {}
for k in range(16):
    b = every_board(k, rules.block_size(k))
    rows[k] = dict(boards=rules.block_size(k), index_mismatches=int(b[0]), tiger_move_mismatches=int(b[1]),
                   goat_move_mismatches=int(b[2]))
    assert b.sum() == 0, (k, rows[k])
    print(f"   k={k:2d}: {rules.block_size(k):,} boards, check_all == rules.py", flush=True)
report["checker_vs_rules_every_board"] = dict(blocks=rows, boards=sum(r["boards"] for r in rows.values()),
                                              mismatches=0, seconds=round(time.time() - t0, 1))
print(f"5. check_all.py: same index, tiger moves and goat moves (placement for k <= 14, sliding for k >= 10) as "
      f"rules.py on all {report['checker_vs_rules_every_board']['boards']:,} boards ({time.time() - t0:.0f}s)")

out = Path(__file__).resolve().parent / "tables" / "rules_proof.json"
out.write_text(json.dumps(report, indent=2))
print("wrote", out)
sys.exit(1 if discrepancies else 0)
