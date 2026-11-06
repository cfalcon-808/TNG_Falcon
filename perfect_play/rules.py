"""Board tables and position indexing for the Tigers & Goats solver.

The board graph and capture rules come from env_tng.py (TnGEnv.move_map, HUB0_CAPTURE_JUMP, the
"no capture up through the hub from squares 2-5" rule), so the solver and the environment share one
source of truth. prove_rules.py proves that move generation, captures and indexing equal TnGEnv's on
every board.

A position is (tiger mask, goat mask, captured c, goats placed n, mover). Positions are grouped in layers:
  movement layer k       : n = 15, k = 15 - c goats on the board, one array per mover
  placement layer (n, m) : n < 15 goats placed, mover m, sub-blocks for c = 0..min(n, 5)
Inside a k-goat block the index is  rank3(tigers) * C(20, k) + rank_k(goats among the 20 non-tiger squares).
"""
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import env_tng  # noqa: E402

SQUARES = 23
TIGERS = 3
GOATS = 15
CAPTURES_TO_WIN = 6

_env = env_tng.TnGEnv()
MOVE_MAP = {p: dict(nbrs) for p, nbrs in _env.move_map.items()}
HUB_JUMP = dict(env_tng.HUB0_CAPTURE_JUMP)
_env.close()

# N[p, d]: neighbour of p in direction d (1..4), -1 if none.  L[p, d]: capture landing, -1 if not allowed.
N = -np.ones((SQUARES, 5), np.int32)
L = -np.ones((SQUARES, 5), np.int32)
for p, nbrs in MOVE_MAP.items():
    for d, q in nbrs.items():
        N[p, d] = q
for p in range(SQUARES):
    for d in range(1, 5):
        q = N[p, d]
        if q < 0 or (p in (2, 3, 4, 5) and d == 1):
            continue
        L[p, d] = HUB_JUMP.get(d, -1) if p == 0 else N[q, d]
# ADJ[q]: squares a piece on q can have slid from (the graph is undirected; checked in verify.py)
ADJ = -np.ones((SQUARES, 4), np.int32)
for q in range(SQUARES):
    nbrs = sorted({int(x) for x in N[q, 1:] if x >= 0})
    ADJ[q, :len(nbrs)] = nbrs
BIN = np.array([[comb(n, k) for k in range(24)] for n in range(24)], np.int64)
TIGER_SETS = comb(SQUARES, TIGERS)                     # 1771


def block_size(k):
    """Boards with 3 tigers and k goats."""
    return TIGER_SETS * comb(SQUARES - TIGERS, k)


def placement_offsets(n):
    """Start of each captured-count sub-block c = 0..min(n, 5) in placement layer n (plus the end)."""
    offs, total = [], 0
    for c in range(min(n, CAPTURES_TO_WIN - 1) + 1):
        offs.append(total)
        total += block_size(n - c)
    offs.append(total)
    return offs


# ── Pure-Python rank / unrank, identical to the CUDA kernels ────────────────────────────────────────
def rank_board(tm, gm, k):
    rt = rg = ti = gi = j = 0
    for p in range(SQUARES):
        if (tm >> p) & 1:
            ti += 1
            rt += comb(p, ti)
        else:
            if (gm >> p) & 1:
                gi += 1
                rg += comb(j, gi)
            j += 1
    return rt * comb(20, k) + rg


def unrank_board(index, k):
    rt, rg = divmod(index, comb(20, k))
    tm, x = 0, 22
    for i in range(TIGERS, 0, -1):
        while comb(x, i) > rt:
            x -= 1
        tm |= 1 << x
        rt -= comb(x, i)
        x -= 1
    slots, x = 0, 19
    for i in range(k, 0, -1):
        while comb(x, i) > rg:
            x -= 1
        slots |= 1 << x
        rg -= comb(x, i)
        x -= 1
    gm, j = 0, 0
    for p in range(SQUARES):
        if (tm >> p) & 1:
            continue
        if (slots >> j) & 1:
            gm |= 1 << p
        j += 1
    return tm, gm


def goat_moves(tm, gm, placing):
    """Legal goat moves as (action, new goat mask). Placement: a = 5p; slide: a = 5p + d."""
    occ = tm | gm
    if placing:
        return [(5 * p, gm | (1 << p)) for p in range(SQUARES) if not (occ >> p) & 1]
    out = []
    for p in range(SQUARES):
        if (gm >> p) & 1:
            for d in range(1, 5):
                q = int(N[p, d])
                if q >= 0 and not (occ >> q) & 1:
                    out.append((5 * p + d, gm ^ (1 << p) ^ (1 << q)))
    return out


def tiger_moves(tm, gm):
    """Legal tiger moves as (action, new tiger mask, new goat mask, captured)."""
    occ = tm | gm
    out = []
    for p in range(SQUARES):
        if not (tm >> p) & 1:
            continue
        for d in range(1, 5):
            q = int(N[p, d])
            if q < 0:
                continue
            if not (occ >> q) & 1:
                out.append((5 * p + d, tm ^ (1 << p) ^ (1 << q), gm, False))
            elif (gm >> q) & 1:
                land = int(L[p, d])
                if land >= 0 and not (occ >> land) & 1:
                    out.append((5 * p + d, tm ^ (1 << p) ^ (1 << land), gm ^ (1 << q), True))
    return out


def popcount(x):
    return bin(x).count("1")
