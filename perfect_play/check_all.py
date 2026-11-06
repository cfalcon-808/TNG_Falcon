#!/usr/bin/env python
"""Independent exhaustive verifier for the Tigers & Goats endgame tables.

Written only from the game rules in ``env_tng.py`` (repo root) and the table
file-format spec; it shares no code with the solver in ``perfect_play/``.

For every position of every table (both movers) the stored value is recomputed
from the stored values of all successors with the minimax recursion

    no legal move                           -> loss in 0
    some move wins (6th capture = win in 1,
      or opponent loses in d')              -> win in min(d') + 1
    else some successor is a draw           -> draw
    else                                    -> loss in max(opponent win d') + 1

and compared exactly.

Why this is a proof (given the move generator is right):
  * Movement tables carry exact distances.  A table that is a fixed point of the
    recursion above is sound by induction on the stored distance (every stored
    distance points to successors with strictly smaller distance) and complete
    by induction on the true distance, so it equals the true game value and the
    true optimal distance.  Infinite play is a draw.
  * Placement positions form a DAG (every goat turn increases the number of
    goats placed, so no cycle is possible) ending in the movement tables, so a
    win/draw/loss fixed point (.u2) is already exact; .u8 tables are checked
    with distances as well.  The .u2 tables are optional once the .u8 tables
    exist: when present, every .u2 entry is also compared with its .u8 entry.

Index of a board with tiger squares t1<t2<t3 and k goats:
    rt * C(20, k) + rg,   rt = C(t1,1) + C(t2,2) + C(t3,3),
    rg = sum_i C(j_i, i) over the goat positions j_1<..<j_k in the list of the
    20 non-tiger squares (i = 1..k).  Equivalently: rg is the colex rank of the
    20-bit "relative" goat mask, so consecutive indices of one tiger set are
    consecutive integers with popcount k (Gosper's hack).

Usage:
    python -I perfect_play/check_all.py --selftest
    python -I perfect_play/check_all.py --layers mv15,mv14,pl0-8
    python -I perfect_play/check_all.py            # everything
Results go to perfect_play/tables/check_all.json; exit code 1 on any mismatch.
"""
import argparse
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import numba  # noqa: E402
from numba import njit, prange  # noqa: E402

import env_tng  # noqa: E402

TABLE_DIR = os.path.join(_HERE, "tables")

# ============================================================================
#  Rules, taken from env_tng.py
# ============================================================================
NSQ = int(env_tng.BOARD_SIZE)                       # 23
NDIR = int(env_tng.DIR_CODES)                       # 5 (action = 5*square + dir)
TOTAL_GOATS = int(env_tng.TOTAL_GOATS_TO_PLACE)     # 15
WIN_CAPTURES = int(env_tng.GOATS_EATEN_FOR_TIGER_WIN)  # 6
assert NSQ == 23 and NDIR == 5 and TOTAL_GOATS == 15 and WIN_CAPTURES == 6, \
    "index/table spec assumes 23 squares, 15 goats, 6 captures"
NFREE = NSQ - 3                                     # 20 non-tiger squares
FULL23 = (1 << NSQ) - 1
FULL20 = (1 << NFREE) - 1
MIN_MV_K = TOTAL_GOATS - (WIN_CAPTURES - 1)         # 10 goats on board at c=5


def _build_rule_tables():
    env = env_tng.TnGEnv()
    nb = np.full((NSQ, NDIR), -1, np.int64)
    for s, dd in env.move_map.items():
        for d, q in dd.items():
            if q is not None:
                nb[s, d] = q
    # capture landing square from tiger square i in direction d (over nb[i,d]);
    # mirrors TnGEnv._tiger_moves exactly.
    land = np.full((NSQ, NDIR), -1, np.int64)
    for i in range(NSQ):
        for d, q in env.move_map.get(i, {}).items():
            if q is None:
                continue
            # "row-1 nodes (b1..e1) cannot capture 'up' through b0"
            if i in (2, 3, 4, 5) and d == 1:
                continue
            if i == 0:
                jump = env_tng.HUB0_CAPTURE_JUMP.get(d, None)
            else:
                jump = env.move_map.get(q, {}).get(d, None)
            if jump is not None:
                land[i, d] = jump
    return env.move_map, nb, land


MOVE_MAP, NB, LAND = _build_rule_tables()

# ---------------------------------------------------------------- binomials
BIN = np.zeros((NSQ + 2, NSQ + 2), np.int64)
for _n in range(NSQ + 2):
    BIN[_n, 0] = 1
    for _r in range(1, _n + 1):
        BIN[_n, _r] = BIN[_n - 1, _r - 1] + BIN[_n - 1, _r]
NTS = int(BIN[NSQ, 3])                              # 1771 tiger sets

# ---------------------------------------------------------------- tiger sets
TSQ = np.full((NTS, 3), -1, np.int64)               # sorted tiger squares
TMASK = np.zeros(NTS, np.int64)
POS20 = np.full((NTS, NSQ), -1, np.int64)           # square -> position in non-tiger list
SQ20 = np.full((NTS, NFREE), -1, np.int64)          # position -> square
for _t3 in range(NSQ):
    for _t2 in range(_t3):
        for _t1 in range(_t2):
            _rt = int(BIN[_t1, 1] + BIN[_t2, 2] + BIN[_t3, 3])
            assert TSQ[_rt, 0] == -1
            TSQ[_rt] = (_t1, _t2, _t3)
            TMASK[_rt] = (1 << _t1) | (1 << _t2) | (1 << _t3)
            _j = 0
            for _s in range(NSQ):
                if _s not in (_t1, _t2, _t3):
                    POS20[_rt, _s] = _j
                    SQ20[_rt, _j] = _s
                    _j += 1
assert (TSQ[:, 0] >= 0).all()

# ---------------------------------------------------------------- rank20 tables
POP10 = np.array([bin(x).count("1") for x in range(1024)], np.int64)
LO10 = np.zeros(1024, np.int64)
HI10 = np.zeros((1024, 11), np.int64)
for _x in range(1024):
    _bits = [b for b in range(10) if (_x >> b) & 1]
    LO10[_x] = sum(BIN[b, i + 1] for i, b in enumerate(_bits))
    for _c in range(11):
        HI10[_x, _c] = sum(BIN[b + 10, i + 1 + _c] for i, b in enumerate(_bits))

# de Bruijn count-trailing-zeros for values < 2**32 with one bit set
_DB_MUL = 0x077CB531
CTZ32 = np.full(32, -1, np.int64)
for _i in range(32):
    CTZ32[(((1 << _i) * _DB_MUL) & 0xFFFFFFFF) >> 27] = _i
assert (CTZ32 >= 0).all()

KIND_U16, KIND_U8, KIND_U2 = 0, 1, 2
BIGD = 1 << 30
MAXS = 128      # max successors of one position (<= 20 placements / 12 tiger / 60 goat moves)
MAXEX = 4       # mismatch examples kept per chunk


# ============================================================================
#  Index primitives
# ============================================================================
@njit(cache=False, inline="always")
def _ctz(low):
    return CTZ32[((low * 0x077CB531) & 0xFFFFFFFF) >> 27]


@njit(cache=False, inline="always")
def _pop20(m):
    return POP10[m & 1023] + POP10[(m >> 10) & 1023]


@njit(cache=False, inline="always")
def _pop23(m):
    return POP10[m & 1023] + POP10[(m >> 10) & 1023] + POP10[(m >> 20) & 1023]


@njit(cache=False, inline="always")
def _rank20(m):
    lo = m & 1023
    return LO10[lo] + HI10[m >> 10, POP10[lo]]


@njit(cache=False)
def _unrank20(r, k):
    m = 0
    j = NFREE - 1
    for i in range(k, 0, -1):
        while BIN[j, i] > r:
            j -= 1
        m |= 1 << j
        r -= BIN[j, i]
        j -= 1
    return m


@njit(cache=False, inline="always")
def _gosper(m):
    c = m & -m
    r = m + c
    return (((r ^ m) >> 2) // c) | r


@njit(cache=False, inline="always")
def _remove_bit(x, t):
    return (x & ((1 << t) - 1)) | ((x >> (t + 1)) << t)


@njit(cache=False, inline="always")
def _insert_zero(x, t):
    return (x & ((1 << t) - 1)) | ((x >> t) << (t + 1))


@njit(cache=False, inline="always")
def _compress23(gm, t1, t2, t3):
    """23-bit goat mask -> 20-bit mask over the non-tiger squares (t1<t2<t3)."""
    return _remove_bit(_remove_bit(_remove_bit(gm, t3), t2), t1)


@njit(cache=False, inline="always")
def _expand20(m, t1, t2, t3):
    """20-bit relative goat mask -> 23-bit goat mask (t1<t2<t3)."""
    return _insert_zero(_insert_zero(_insert_zero(m, t1), t2), t3)


@njit(cache=False, inline="always")
def _rank3(a, b, c):
    """rt of the tiger set {a,b,c} (any order)."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
    if a > b:
        a, b = b, a
    return BIN[a, 1] + BIN[b, 2] + BIN[c, 3], a, b, c


@njit(cache=False)
def _tiger_rt(tm):
    """rt of a 23-bit tiger mask with exactly 3 bits."""
    t = np.empty(3, np.int64)
    n = 0
    for s in range(NSQ):
        if (tm >> s) & 1:
            if n == 3:
                raise ValueError("tiger mask must have exactly 3 bits")
            t[n] = s
            n += 1
    if n != 3:
        raise ValueError("tiger mask must have exactly 3 bits")
    return BIN[t[0], 1] + BIN[t[1], 2] + BIN[t[2], 3]


@njit(cache=False)
def rank(tm, gm, k):
    """Index of the board (tm, gm) in the k-goat block: rt*C(20,k) + rg.

    tm, gm: 23-bit masks (bit s = square s) of the 3 tigers and the k goats.
    Fast path (table based); see rank_reference for the literal spec version.
    """
    rt = _tiger_rt(tm)
    if (tm & gm) != 0 or _pop23(gm) != k:
        raise ValueError("bad goat mask")
    m = _compress23(gm, TSQ[rt, 0], TSQ[rt, 1], TSQ[rt, 2])
    return rt * BIN[NFREE, k] + _rank20(m)


@njit(cache=False)
def rank_reference(tm, gm, k):
    """Literal implementation of the index spec (slow; used to test rank)."""
    rt = 0
    n = 0          # tigers seen so far: rt = C(t1,1) + C(t2,2) + C(t3,3)
    for s in range(NSQ):
        if (tm >> s) & 1:
            n += 1
            rt += BIN[s, n]
    if n != 3:
        return -1
    j = 0          # position in the list of non-tiger squares
    i = 0          # number of goats seen so far
    rg = 0
    for s in range(NSQ):
        if (tm >> s) & 1:
            continue
        if (gm >> s) & 1:
            i += 1
            rg += BIN[j, i]
        j += 1
    if i != k:
        return -1
    return rt * BIN[NFREE, k] + rg


@njit(cache=False)
def unrank(idx, k):
    """Inverse of rank: index in the k-goat block -> (tiger mask, goat mask)."""
    ck = BIN[NFREE, k]
    rt = idx // ck
    rg = idx - rt * ck
    m = _unrank20(rg, k)
    return TMASK[rt], _expand20(m, TSQ[rt, 0], TSQ[rt, 1], TSQ[rt, 2])


# ============================================================================
#  Move generation (the single source used by the checker and the wrappers)
# ============================================================================
@njit(cache=False)
def _tiger_moves_core(t1, t2, t3, tm, gm, act, frm, dst, cap):
    """Legal tiger moves; mirrors TnGEnv._tiger_moves.  Writes action id
    (5*from+dir), from square, landing square and captured square (-1 for a
    slide) into the given arrays; returns the number of moves."""
    occ = tm | gm
    n = 0
    for w in range(3):
        a = t1 if w == 0 else (t2 if w == 1 else t3)
        for d in range(1, NDIR):
            q = NB[a, d]
            if q < 0:
                continue
            if ((occ >> q) & 1) == 0:                       # slide
                act[n] = NDIR * a + d
                frm[n] = a
                dst[n] = q
                cap[n] = -1
                n += 1
            elif ((gm >> q) & 1) != 0:                      # capture candidate
                land = LAND[a, d]
                if land >= 0 and ((occ >> land) & 1) == 0:
                    act[n] = NDIR * a + d
                    frm[n] = a
                    dst[n] = land
                    cap[n] = q
                    n += 1
    return n


@njit(cache=False)
def _goat_moves_core(tm, gm, placing, act, frm, dst):
    """Legal goat moves; mirrors TnGEnv._update_valid_moves.  Placement:
    action 5*p, from -1, to p for every empty p.  Movement: action 5*i+d for a
    goat on i sliding to the empty move_map[i][d]."""
    occ = tm | gm
    n = 0
    if placing:
        e = (~occ) & FULL23
        while e != 0:
            low = e & -e
            e ^= low
            p = _ctz(low)
            act[n] = NDIR * p
            frm[n] = -1
            dst[n] = p
            n += 1
    else:
        x = gm
        while x != 0:
            low = x & -x
            x ^= low
            a = _ctz(low)
            for d in range(1, NDIR):
                b = NB[a, d]
                if b >= 0 and ((occ >> b) & 1) == 0:
                    act[n] = NDIR * a + d
                    frm[n] = a
                    dst[n] = b
                    n += 1
    return n


@njit(cache=False)
def tiger_successors(tm, gm):
    """All tiger moves from (tm, gm) as rows (action, new_tm, new_gm, captured).

    captured = 1 for a capture (the jumped goat is removed), 0 for a slide."""
    rt = _tiger_rt(tm)
    act = np.empty(MAXS, np.int64)
    frm = np.empty(MAXS, np.int64)
    dst = np.empty(MAXS, np.int64)
    cap = np.empty(MAXS, np.int64)
    n = _tiger_moves_core(TSQ[rt, 0], TSQ[rt, 1], TSQ[rt, 2], tm, gm, act, frm, dst, cap)
    out = np.empty((n, 4), np.int64)
    for s in range(n):
        out[s, 0] = act[s]
        out[s, 1] = (tm & ~(1 << frm[s])) | (1 << dst[s])
        if cap[s] >= 0:
            out[s, 2] = gm & ~(1 << cap[s])
            out[s, 3] = 1
        else:
            out[s, 2] = gm
            out[s, 3] = 0
    return out


@njit(cache=False)
def goat_successors(tm, gm, placing):
    """All goat moves from (tm, gm) as rows (action, new_tm, new_gm, 0).

    placing=True: placement phase (any empty square); else movement phase."""
    act = np.empty(MAXS, np.int64)
    frm = np.empty(MAXS, np.int64)
    dst = np.empty(MAXS, np.int64)
    n = _goat_moves_core(tm, gm, placing, act, frm, dst)
    out = np.empty((n, 4), np.int64)
    for s in range(n):
        out[s, 0] = act[s]
        out[s, 1] = tm
        if placing:
            out[s, 2] = gm | (1 << dst[s])
        else:
            out[s, 2] = (gm & ~(1 << frm[s])) | (1 << dst[s])
        out[s, 3] = 0
    return out


# ---------------------------------------------------------------- successor indices
@njit(cache=False)
def _goat_succ_idx(rt, m, gm, k, placing, act, frm, dst, out):
    """Successor indices (inside the successor block) of a goat-to-move board
    (tiger set rt, relative mask m == compress(gm)).  Placement successors live
    in the (k+1)-goat block, movement successors in the k-goat block."""
    n = _goat_moves_core(TMASK[rt], gm, placing, act, frm, dst)
    if placing:
        base = rt * BIN[NFREE, k + 1]
        for s in range(n):
            out[s] = base + _rank20(m | (1 << POS20[rt, dst[s]]))
    else:
        base = rt * BIN[NFREE, k]
        for s in range(n):
            out[s] = base + _rank20(m ^ (1 << POS20[rt, frm[s]]) ^ (1 << POS20[rt, dst[s]]))
    return n


@njit(cache=False)
def _tiger_succ_idx(rt, gm, k, cap_wins, act, frm, dst, cap, kind, out):
    """Successors of a tiger-to-move board.  kind[s] = 0: slide, index in the
    k-goat block; 1: capture, index in the (k-1)-goat block; 2: capture that is
    the winning capture (cap_wins), no index."""
    t1 = TSQ[rt, 0]
    t2 = TSQ[rt, 1]
    t3 = TSQ[rt, 2]
    n = _tiger_moves_core(t1, t2, t3, TMASK[rt], gm, act, frm, dst, cap)
    for s in range(n):
        a = frm[s]
        if a == t1:
            x, y = t2, t3
        elif a == t2:
            x, y = t1, t3
        else:
            x, y = t1, t2
        rt2, s1, s2, s3 = _rank3(x, y, dst[s])
        if cap[s] < 0:
            kind[s] = 0
            out[s] = rt2 * BIN[NFREE, k] + _rank20(_compress23(gm, s1, s2, s3))
        elif cap_wins:
            kind[s] = 2
            out[s] = -1
        else:
            kind[s] = 1
            g2 = gm & ~(1 << cap[s])
            out[s] = rt2 * BIN[NFREE, k - 1] + _rank20(_compress23(g2, s1, s2, s3))
    return n


@njit(cache=False)
def goat_successor_indices(tm, gm, k, placing):
    """Python-callable: successor indices (as the checker computes them) of a
    goat-to-move board with k goats; see _goat_succ_idx."""
    rt = _tiger_rt(tm)
    m = _compress23(gm, TSQ[rt, 0], TSQ[rt, 1], TSQ[rt, 2])
    act = np.empty(MAXS, np.int64)
    frm = np.empty(MAXS, np.int64)
    dst = np.empty(MAXS, np.int64)
    out = np.empty(MAXS, np.int64)
    n = _goat_succ_idx(rt, m, gm, k, placing, act, frm, dst, out)
    return out[:n].copy()


@njit(cache=False)
def tiger_successor_indices(tm, gm, k, cap_wins):
    """Python-callable: rows (kind, index) of the successors of a tiger-to-move
    board with k goats, as the checker computes them; see _tiger_succ_idx."""
    rt = _tiger_rt(tm)
    act = np.empty(MAXS, np.int64)
    frm = np.empty(MAXS, np.int64)
    dst = np.empty(MAXS, np.int64)
    cap = np.empty(MAXS, np.int64)
    kind = np.empty(MAXS, np.int64)
    out = np.empty(MAXS, np.int64)
    n = _tiger_succ_idx(rt, gm, k, cap_wins, act, frm, dst, cap, kind, out)
    res = np.empty((n, 2), np.int64)
    for s in range(n):
        res[s, 0] = kind[s]
        res[s, 1] = out[s]
    return res


# ============================================================================
#  Table value codes.  Internally every value is a "u16 style" code:
#  0 draw, 2d win in d plies, 2d+1 loss in d plies, -1 invalid (.u2 value 3).
# ============================================================================
@njit(cache=False, inline="always")
def _read_code(arr, kind, idx):
    if kind == KIND_U16:
        j = idx * 2
        return np.int64(arr[j]) | (np.int64(arr[j + 1]) << 8)
    elif kind == KIND_U8:
        e = np.int64(arr[idx])
        if e == 0:
            return np.int64(0)
        d = e - 1
        if d & 1:
            return 2 * d          # d odd: mover wins
        return 2 * d + 1          # d even: mover loses
    else:
        v = (np.int64(arr[idx >> 2]) >> ((idx & 3) << 1)) & 3
        if v == 0:
            return np.int64(0)
        if v == 1:
            return np.int64(2)    # win (distance unknown)
        if v == 2:
            return np.int64(1)    # loss (distance unknown)
        return np.int64(-1)


@njit(cache=False, inline="always")
def _read_raw(arr, kind, idx):
    if kind == KIND_U16:
        return np.int64(arr[2 * idx]) | (np.int64(arr[2 * idx + 1]) << 8)
    elif kind == KIND_U8:
        return np.int64(arr[idx])
    return (np.int64(arr[idx >> 2]) >> ((idx & 3) << 1)) & 3


@njit(cache=False, inline="always")
def _wdl(code):
    """0 draw, 1 win, 2 loss, 3 invalid (matches the .u2 encoding)."""
    if code < 0:
        return 3
    if code == 0:
        return 0
    if code & 1:
        return 2
    return 1


@njit(cache=False, inline="always")
def _combine(has_move, best_win, any_draw, max_opp_win):
    if not has_move:
        return np.int64(1)                    # loss in 0
    if best_win < BIGD:
        return 2 * (best_win + 1)             # win in min d' + 1
    if any_draw:
        return np.int64(0)
    return 2 * (max_opp_win + 1) + 1          # loss in max d' + 1


# ============================================================================
#  The checking kernel: one block of boards (tiger sets x k goats)
# ============================================================================
@njit(parallel=True, cache=False)
def _check_block(tiger_to_move, placing, k, chunk,
                 cur, cur_kind, cur_base,
                 sa, sa_kind, sa_base,
                 sb, sb_kind, sb_base, cap_wins,
                 dist_mode, aux, aux_kind, aux_base, has_aux,
                 o_mism, o_aux, o_cnt, o_maxd, o_exn, o_ex):
    ck = BIN[NFREE, k]
    total = NTS * ck
    nch = (total + chunk - 1) // chunk
    for ch in prange(nch):
        i0 = ch * chunk
        i1 = min(total, i0 + chunk)
        rt = i0 // ck
        rg = i0 - rt * ck
        m = _unrank20(rg, k)
        t1 = TSQ[rt, 0]
        t2 = TSQ[rt, 1]
        t3 = TSQ[rt, 2]
        act = np.empty(MAXS, np.int64)
        frm = np.empty(MAXS, np.int64)
        dst = np.empty(MAXS, np.int64)
        cap = np.empty(MAXS, np.int64)
        knd = np.empty(MAXS, np.int64)
        sidx = np.empty(MAXS, np.int64)
        mism = 0
        naux = 0
        cdraw = 0
        cwin = 0
        closs = 0
        cbad = 0
        maxd = -1
        nex = 0
        for i in range(i0, i1):
            gm = _expand20(m, t1, t2, t3)
            if tiger_to_move:
                ns = _tiger_succ_idx(rt, gm, k, cap_wins, act, frm, dst, cap, knd, sidx)
            else:
                ns = _goat_succ_idx(rt, m, gm, k, placing, act, frm, dst, sidx)
            best = BIGD
            anyd = False
            mx = -1
            for s in range(ns):
                if tiger_to_move:
                    kd = knd[s]
                    if kd == 2:
                        code = np.int64(1)        # opponent "lost in 0" -> win in 1
                    elif kd == 0:
                        code = _read_code(sa, sa_kind, sa_base + sidx[s])
                    else:
                        code = _read_code(sb, sb_kind, sb_base + sidx[s])
                else:
                    code = _read_code(sa, sa_kind, sa_base + sidx[s])
                if code <= 0:
                    anyd = True                   # draw (invalid entries flagged at their own layer)
                elif code & 1:
                    d = code >> 1
                    if d < best:
                        best = d
                else:
                    d = code >> 1
                    if d > mx:
                        mx = d
            exp = _combine(ns > 0, best, anyd, mx)
            st = _read_code(cur, cur_kind, cur_base + i)
            if st < 0:
                cbad += 1
            elif st == 0:
                cdraw += 1
            elif st & 1:
                closs += 1
            else:
                cwin += 1
            if st > 0 and (st >> 1) > maxd:
                maxd = st >> 1
            if dist_mode:
                ok = st == exp
            else:
                ok = _wdl(st) == _wdl(exp)
            if not ok:
                mism += 1
                if nex < MAXEX:
                    o_ex[ch, nex, 0] = cur_base + i
                    o_ex[ch, nex, 1] = 0
                    o_ex[ch, nex, 2] = _read_raw(cur, cur_kind, cur_base + i)
                    o_ex[ch, nex, 3] = exp
                    nex += 1
            if has_aux:
                if _wdl(_read_code(aux, aux_kind, aux_base + i)) != _wdl(st):
                    naux += 1
                    if nex < MAXEX:
                        o_ex[ch, nex, 0] = cur_base + i
                        o_ex[ch, nex, 1] = 1
                        o_ex[ch, nex, 2] = _read_raw(cur, cur_kind, cur_base + i)
                        o_ex[ch, nex, 3] = _read_raw(aux, aux_kind, aux_base + i)
                        nex += 1
            # next board in index order
            rg += 1
            if rg == ck:
                rg = 0
                rt += 1
                if rt < NTS:
                    t1 = TSQ[rt, 0]
                    t2 = TSQ[rt, 1]
                    t3 = TSQ[rt, 2]
                m = (1 << k) - 1
            else:
                m = _gosper(m)
        o_mism[ch] = mism
        o_aux[ch] = naux
        o_cnt[ch, 0] = cdraw
        o_cnt[ch, 1] = cwin
        o_cnt[ch, 2] = closs
        o_cnt[ch, 3] = cbad
        o_maxd[ch] = maxd
        o_exn[ch] = nex


# ============================================================================
#  Table files
# ============================================================================
def placement_offsets(n):
    """Block offsets of placement layer n (blocks c = 0..min(n,5)) + total."""
    offs = [0]
    for c in range(min(n, WIN_CAPTURES - 1) + 1):
        offs.append(offs[-1] + NTS * int(BIN[NFREE, n - c]))
    return offs


def mv_boards(k):
    return NTS * int(BIN[NFREE, k])


_DUMMY = np.zeros(16, np.uint8)
_DUMMY.flags.writeable = False


class Tables:
    """Read-only memory maps of the table files (opened lazily, kept per job)."""

    def __init__(self, table_dir):
        self.dir = table_dir
        self.cache = {}
        self.used = {}

    def _open(self, name, nbytes, required=True):
        if name in self.cache:
            return self.cache[name]
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(path)
            return None
        size = os.path.getsize(path)
        if size != nbytes:
            if required:
                raise ValueError(f"{path}: size {size} != expected {nbytes}")
            print(f"  note: {name} has size {size}, expected {nbytes} -> treated as absent")
            return None
        arr = np.memmap(path, dtype=np.uint8, mode="r").view(np.ndarray)
        st = os.stat(path)
        self.used[name] = {"bytes": size, "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                                                 time.localtime(st.st_mtime))}
        self.cache[name] = arr
        return arr

    def mv(self, k, mover):
        return self._open(f"mv_k{k}_{mover}.u16", 2 * mv_boards(k))

    def u2(self, n, mover, required=True):
        return self._open(f"pl_n{n}_{mover}.u2", (placement_offsets(n)[-1] + 3) // 4, required)

    def u8(self, n, mover, required=False):
        return self._open(f"pl_n{n}_{mover}.u8", placement_offsets(n)[-1], required)

    def release(self):
        self.cache = {}


# ============================================================================
#  Running one layer
# ============================================================================
def _decode(code):
    if code < 0:
        return "invalid"
    if code == 0:
        return "draw"
    return ("loss in %d" if code & 1 else "win in %d") % (code >> 1)


def _decode_raw(raw, kind):
    """Human-readable stored value; .u2 entries carry no distance."""
    if kind == KIND_U2:
        return {0: "draw", 1: "win", 2: "loss"}.get(int(raw), "invalid")
    return _decode(_raw_to_code(raw, kind))


def _raw_to_code(raw, kind):
    if kind == KIND_U16:
        return int(raw)
    if kind == KIND_U8:
        if raw == 0:
            return 0
        d = raw - 1
        return 2 * d if d & 1 else 2 * d + 1
    return {0: 0, 1: 2, 2: 1}.get(int(raw), -1)


def _code_to_raw(code, kind):
    if kind == KIND_U16:
        return int(code)
    if kind == KIND_U8:
        return 0 if code == 0 else int(code >> 1) + 1
    return 0 if code == 0 else (2 if code & 1 else 1)


def _squares(mask):
    return [s for s in range(NSQ) if (mask >> s) & 1]


def run_block(blk, chunk):
    """Run the kernel on one block description; returns aggregated numbers."""
    k = blk["k"]
    total = NTS * int(BIN[NFREE, k])
    nch = (total + chunk - 1) // chunk
    o_mism = np.zeros(nch, np.int64)
    o_aux = np.zeros(nch, np.int64)
    o_cnt = np.zeros((nch, 4), np.int64)
    o_maxd = np.full(nch, -1, np.int64)
    o_exn = np.zeros(nch, np.int64)
    o_ex = np.zeros((nch, MAXEX, 4), np.int64)
    _check_block(blk["tiger"], blk["placing"], k, chunk,
                 blk["cur"], blk["cur_kind"], blk["cur_base"],
                 blk["sa"], blk["sa_kind"], blk["sa_base"],
                 blk["sb"], blk["sb_kind"], blk["sb_base"], blk["cap_wins"],
                 blk["dist"], blk["aux"], blk["aux_kind"], blk["aux_base"], blk["has_aux"],
                 o_mism, o_aux, o_cnt, o_maxd, o_exn, o_ex)
    ex = []
    for ch in np.flatnonzero(o_exn):
        for e in range(o_exn[ch]):
            ex.append(tuple(int(v) for v in o_ex[ch, e]))
            if len(ex) >= 5:
                break
        if len(ex) >= 5:
            break
    return {
        "positions": total,
        "mismatches": int(o_mism.sum()),
        "aux_mismatches": int(o_aux.sum()),
        "counts": o_cnt.sum(axis=0),
        "max_d": int(o_maxd.max()) if nch else -1,
        "examples": ex,
    }


def _example_record(ex, blk, n_placed, c, cur_name, aux_name):
    gidx, typ, raw, other = ex
    k = blk["k"]
    tm, gm = unrank(gidx - blk["cur_base"], k)
    rec = {
        "index": gidx,
        "tigers": _squares(tm),
        "goats": _squares(gm),
        "captured": c,
        "goats_placed": n_placed,
    }
    if typ == 0:
        rec["kind"] = "recursion"
        rec["stored"] = raw
        rec["stored_decoded"] = _decode_raw(raw, blk["cur_kind"])
        rec["expected"] = _code_to_raw(other, blk["cur_kind"])
        rec["expected_decoded"] = _decode(other) if blk["dist"] else \
            ["draw", "win", "loss"][0 if other == 0 else (2 if other & 1 else 1)]
    else:
        rec["kind"] = f"{cur_name} vs {aux_name} value"
        rec["stored"] = raw
        rec["stored_decoded"] = _decode_raw(raw, blk["cur_kind"])
        rec["other"] = other
        rec["other_decoded"] = _decode_raw(other, blk["aux_kind"])
    return rec


def build_job(tabs, kind, num, mover):
    """Describe the blocks of one layer/mover.  Returns (name, mode, blocks)."""
    blocks = []
    tiger = mover == "T"
    if kind == "mv":
        k = num
        c = TOTAL_GOATS - k
        cur = tabs.mv(k, mover)
        if not tiger:
            sa, sb, cap_wins = tabs.mv(k, "T"), _DUMMY, False
        else:
            sa = tabs.mv(k, "G")
            cap_wins = c + 1 == WIN_CAPTURES
            sb = _DUMMY if cap_wins else tabs.mv(k - 1, "G")
        blocks.append(dict(c=c, n=TOTAL_GOATS, k=k, tiger=tiger, placing=False,
                           cur=cur, cur_kind=KIND_U16, cur_base=0,
                           sa=sa, sa_kind=KIND_U16, sa_base=0,
                           sb=sb, sb_kind=KIND_U16, sb_base=0, cap_wins=cap_wins,
                           dist=True, aux=_DUMMY, aux_kind=KIND_U2, aux_base=0, has_aux=False))
        return f"mv{k}_{mover}", "u16 (value+distance)", f"mv_k{k}_{mover}.u16", None, blocks

    n = num
    offs = placement_offsets(n)
    u2 = tabs.u2(n, mover, required=False)      # optional once the .u8 tables exist
    u8 = tabs.u8(n, mover)
    # successor tables: (array, kind, offsets or None for mv)
    if not tiger:
        if n + 1 == TOTAL_GOATS:
            succ_dist = True
            s8 = None
        else:
            s8 = tabs.u8(n + 1, "T")
            succ_dist = s8 is not None
            s2 = tabs.u2(n + 1, "T", required=False)
            soffs = placement_offsets(n + 1)
    else:
        s8 = tabs.u8(n, "G")
        succ_dist = s8 is not None
        s2 = tabs.u2(n, "G", required=False)
        soffs = offs
    dist = u8 is not None and succ_dist
    if dist:
        cur, cur_kind, cur_name = u8, KIND_U8, f"pl_n{n}_{mover}.u8"
        if u2 is not None:
            aux, aux_kind, has_aux, aux_name = u2, KIND_U2, True, f"pl_n{n}_{mover}.u2"
            mode = "u8 (value+distance) + u2 consistency"
        else:
            aux, aux_kind, has_aux, aux_name = _DUMMY, KIND_U2, False, None
            mode = "u8 (value+distance)"
    else:
        if u2 is None:
            u2 = tabs.u2(n, mover)                  # raises: neither table can be checked
        if not (not tiger and n + 1 == TOTAL_GOATS) and s2 is None:
            s2 = tabs.u2(n + 1, "T") if not tiger else tabs.u2(n, "G")   # raises
        cur, cur_kind = u2, KIND_U2
        cur_name = f"pl_n{n}_{mover}.u2"
        if u8 is not None:
            aux, aux_kind, has_aux = u8, KIND_U8, True
            aux_name = f"pl_n{n}_{mover}.u8"
            mode = "u2 (value) + u8 consistency (successor u8 missing)"
        else:
            aux, aux_kind, has_aux = _DUMMY, KIND_U2, False
            aux_name = None
            mode = "u2 (value)"
    for c in range(min(n, WIN_CAPTURES - 1) + 1):
        k = n - c
        blk = dict(c=c, n=n, k=k, tiger=tiger, placing=not tiger,
                   cur=cur, cur_kind=cur_kind, cur_base=offs[c],
                   dist=dist, aux=aux, aux_kind=aux_kind, aux_base=offs[c], has_aux=has_aux,
                   sb=_DUMMY, sb_kind=KIND_U2, sb_base=0, cap_wins=False)
        if not tiger:
            if n + 1 == TOTAL_GOATS:
                blk.update(sa=tabs.mv(TOTAL_GOATS - c, "T"), sa_kind=KIND_U16, sa_base=0)
            elif dist:
                blk.update(sa=s8, sa_kind=KIND_U8, sa_base=soffs[c])
            else:
                blk.update(sa=s2, sa_kind=KIND_U2, sa_base=soffs[c])
        else:
            arr, akind = (s8, KIND_U8) if dist else (s2, KIND_U2)
            blk.update(sa=arr, sa_kind=akind, sa_base=offs[c])
            if c + 1 == WIN_CAPTURES:
                blk["cap_wins"] = True
            elif k > 0:
                blk.update(sb=arr, sb_kind=akind, sb_base=offs[c + 1])
        blocks.append(blk)
    return f"pl{n}_{mover}", mode, cur_name, aux_name, blocks


def run_job(tabs, kind, num, mover, chunk, meta):
    t0 = time.time()
    name, mode, cur_name, aux_name, blocks = build_job(tabs, kind, num, mover)
    res = {"layer": name, "mode": mode, "positions": 0, "mismatches": 0,
           "aux_mismatches": 0, "counts": {"draw": 0, "win": 0, "loss": 0, "invalid": 0},
           "max_d": None, "examples": [], "blocks": []}
    maxd = -1
    for blk in blocks:
        tb = time.time()
        r = run_block(blk, chunk)
        res["positions"] += r["positions"]
        res["mismatches"] += r["mismatches"]
        res["aux_mismatches"] += r["aux_mismatches"]
        for key, v in zip(("draw", "win", "loss", "invalid"), r["counts"]):
            res["counts"][key] += int(v)
        maxd = max(maxd, r["max_d"])
        for ex in r["examples"]:
            if len(res["examples"]) < 5:
                res["examples"].append(_example_record(ex, blk, blk["n"], blk["c"],
                                                       cur_name, aux_name))
        res["blocks"].append({"c": blk["c"], "k": blk["k"], "positions": r["positions"],
                              "mismatches": r["mismatches"],
                              "aux_mismatches": r["aux_mismatches"],
                              "seconds": round(time.time() - tb, 2)})
    if blocks[0]["dist"]:
        res["max_d"] = maxd
    dt = time.time() - t0
    res["seconds"] = round(dt, 2)
    res["positions_per_second"] = round(res["positions"] / dt) if dt > 0 else None
    res["metadata_counts_match"] = _meta_match(meta, kind, num, mover, res)
    return res


def _meta_match(meta, kind, num, mover, res):
    try:
        if kind == "mv":
            md = meta["movement"][str(num)][mover]
            ok = all(md[key] == res["counts"][key] for key in ("win", "draw", "loss"))
            if res["max_d"] is not None and "max_dtm" in md:
                ok = ok and md["max_dtm"] == res["max_d"]
        else:
            md = meta["placement"][str(num)][mover]
            ok = all(md[key] == res["counts"][key] for key in ("win", "draw", "loss"))
        return bool(ok)
    except (KeyError, TypeError):
        return None


# ============================================================================
#  CLI layer selection
# ============================================================================
MV_RANGE = list(range(MIN_MV_K, TOTAL_GOATS + 1))     # 10..15
PL_RANGE = list(range(0, TOTAL_GOATS))                # 0..14


def parse_layers(spec):
    jobs = set()
    for tok in spec.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok == "all":
            for k in MV_RANGE:
                jobs.update({("mv", k, "G"), ("mv", k, "T")})
            for n in PL_RANGE:
                jobs.update({("pl", n, "G"), ("pl", n, "T")})
            continue
        mt = re.fullmatch(r"(mv|pl)(\d+)(?:-(\d+))?([gt])?", tok)
        if not mt:
            raise SystemExit(f"bad layer spec {tok!r} (e.g. mv15, mv10-15, pl0-8, pl13T, all)")
        kind = mt.group(1)
        lo = int(mt.group(2))
        hi = int(mt.group(3)) if mt.group(3) else lo
        lo, hi = min(lo, hi), max(lo, hi)
        valid = MV_RANGE if kind == "mv" else PL_RANGE
        for v in range(lo, hi + 1):
            if v not in valid:
                raise SystemExit(f"{kind}{v} does not exist (mv{MV_RANGE[0]}-{MV_RANGE[-1]}, "
                                 f"pl{PL_RANGE[0]}-{PL_RANGE[-1]})")
            movers = [mt.group(4).upper()] if mt.group(4) else ["G", "T"]
            for mv in movers:
                jobs.add((kind, v, mv))
    return sorted(jobs, key=lambda j: (0 if j[0] == "mv" else 1, -j[1], j[2]))


# ============================================================================
#  Self tests
# ============================================================================
@njit(parallel=True, cache=False)
def _selftest_full_index(out_err):
    """Every tiger set x every 20-bit mask: expand/compress, fast rank vs the
    literal spec rank, unrank round trip."""
    for rt in prange(NTS):
        tm = TMASK[rt]
        t1 = TSQ[rt, 0]
        t2 = TSQ[rt, 1]
        t3 = TSQ[rt, 2]
        err = 0
        for m in range(1 << NFREE):
            k = _pop20(m)
            gm = _expand20(m, t1, t2, t3)
            if (gm & tm) != 0 or _pop23(gm) != k or gm > FULL23:
                err += 1
            if _compress23(gm, t1, t2, t3) != m:
                err += 1
            idx = rt * BIN[NFREE, k] + _rank20(m)
            if rank_reference(tm, gm, k) != idx:
                err += 1
            tm2, gm2 = unrank(idx, k)
            if tm2 != tm or gm2 != gm:
                err += 1
        out_err[rt] = err


@njit(cache=False)
def _selftest_rank20():
    """rank20 is a bijection {k-subsets of 20} -> [0, C(20,k)) for every k,
    unrank20 inverts it, and Gosper's hack enumerates masks in rank order."""
    err = 0
    for m in range(1 << NFREE):
        k = _pop20(m)
        r = _rank20(m)
        # literal definition
        rr = 0
        i = 0
        for j in range(NFREE):
            if (m >> j) & 1:
                i += 1
                rr += BIN[j, i]
        if rr != r or r < 0 or r >= BIN[NFREE, k]:
            err += 1
        if _unrank20(r, k) != m:
            err += 1
    # per-k bijection + Gosper order
    for k in range(NFREE + 1):
        ck = BIN[NFREE, k]
        hit = np.zeros(ck, np.int64)
        m = (1 << k) - 1
        for r in range(ck):
            if _pop20(m) != k or m > FULL20:
                err += 1
                break
            rr = _rank20(m)
            if rr != r:
                err += 1
            hit[rr] += 1
            if r + 1 < ck:
                m = _gosper(m)
        for r in range(ck):
            if hit[r] != 1:
                err += 1
    return err


def selftest_index():
    print("[selftest] index: rank3 bijection, rank20, unrank, Gosper order ...", flush=True)
    t = time.time()
    rts = set()
    for t3 in range(NSQ):
        for t2 in range(t3):
            for t1 in range(t2):
                rts.add(int(BIN[t1, 1] + BIN[t2, 2] + BIN[t3, 3]))
    ok3 = rts == set(range(NTS))
    e20 = int(_selftest_rank20())
    out = np.zeros(NTS, np.int64)
    _selftest_full_index(out)
    efull = int(out.sum())
    # spot checks of the Python-callable wrappers
    rng = np.random.default_rng(1)
    ewrap = 0
    for _ in range(2000):
        ts = rng.choice(NSQ, 3, replace=False)
        tm = int(sum(1 << int(s) for s in ts))
        free = [s for s in range(NSQ) if not (tm >> s) & 1]
        k = int(rng.integers(0, TOTAL_GOATS + 1))
        gs = rng.choice(free, k, replace=False)
        gm = int(sum(1 << int(s) for s in gs))
        idx = rank(tm, gm, k)
        if idx != rank_reference(tm, gm, k) or tuple(unrank(idx, k)) != (tm, gm):
            ewrap += 1
    res = {"rank3_bijection": ok3, "rank20_errors": e20,
           "boards_checked": NTS * (1 << NFREE), "full_index_errors": efull,
           "wrapper_errors": ewrap, "seconds": round(time.time() - t, 1)}
    print(f"   {res}", flush=True)
    return res, ok3 and e20 == 0 and efull == 0 and ewrap == 0


def _random_board(rng, k):
    ts = rng.choice(NSQ, 3, replace=False)
    tm = int(sum(1 << int(s) for s in ts))
    free = [s for s in range(NSQ) if not (tm >> s) & 1]
    gs = rng.choice(free, k, replace=False)
    gm = int(sum(1 << int(s) for s in gs))
    return tm, gm


def _env_set(env, tm, gm, c, n):
    board = np.zeros(NSQ, dtype=np.int8)
    for s in range(NSQ):
        if (tm >> s) & 1:
            board[s] = 2
        elif (gm >> s) & 1:
            board[s] = 1
    env.board = board
    env.eaten = c
    env.goats_placed = n
    env.phase = 0 if n < TOTAL_GOATS else 1
    env._update_valid_moves()


def _env_goat_moves(env, tm, gm):
    """(action, new_gm) via TnGEnv.get_action_mask on a goat-learner env."""
    out = set()
    for a in np.flatnonzero(env.get_action_mask()):
        pos, d = int(a) // NDIR, int(a) % NDIR
        if env.phase == 0:
            out.add((int(a), gm | (1 << pos)))
        else:
            dest = env.move_map[pos][d]
            out.add((int(a), (gm & ~(1 << pos)) | (1 << dest)))
    return out


def _env_tiger_moves(env, tm, gm, semantics="spec"):
    """(action, new_tm, new_gm, captured) for every move of TnGEnv._tiger_moves.

    semantics="spec": a capture over q = move_map[from][dir] removes the goat
      on q (the model the tables solve; the q that _tiger_moves tests).
    semantics="env_step": the goat is resolved like TnGEnv.step does, with
      _find_jumped_goat(from, to); removed only if a goat is on that square.
    """
    out = set()
    for f, t, capf, d in env._tiger_moves(include_dir=True):
        ntm = (tm & ~(1 << f)) | (1 << t)
        if not capf:
            out.add((f * NDIR + d, ntm, gm, 0))
        elif semantics == "spec":
            q = env.move_map[f][d]
            out.add((f * NDIR + d, ntm, gm & ~(1 << q), 1))
        else:
            j = env._find_jumped_goat(f, t)
            if j is not None and env.board[j] == 1:
                out.add((f * NDIR + d, ntm, gm & ~(1 << j), 1))
            else:
                out.add((f * NDIR + d, ntm, gm, 0))     # env: tiger moves, nothing eaten
    return out


def env_capture_divergences():
    """Captures where TnGEnv._find_jumped_goat(from, to) is not the square q
    that _tiger_moves jumped over (i.e. where TnGEnv.step deviates from the
    capture rule of _tiger_moves)."""
    env = env_tng.TnGEnv()
    out = []
    for i in range(NSQ):
        for d in range(1, NDIR):
            if NB[i, d] >= 0 and LAND[i, d] >= 0:
                j = env._find_jumped_goat(i, int(LAND[i, d]))
                if j != NB[i, d]:
                    out.append({"from": i, "dir": d, "over": int(NB[i, d]),
                                "to": int(LAND[i, d]), "env_find_jumped_goat": j})
    return out


def selftest_moves(nboards, seed=12345):
    """Move generation (and successor indexing) against TnGEnv on random boards."""
    print(f"[selftest] move generation vs TnGEnv on {nboards} random boards ...", flush=True)
    t = time.time()
    rng = np.random.default_rng(seed)
    env_g = env_tng.TnGEnv()
    env_t = env_tng.TnGEnv(learner_role=env_tng.TIGER_LEARNER)
    err_g = err_t = err_ti = err_gi = err_tmask = 0
    nmov_g = nmov_t = ncap = 0
    n_div = 0
    div_ex = []
    for it in range(nboards):
        placing = bool(rng.integers(0, 2))
        if placing:
            k = int(rng.integers(0, TOTAL_GOATS))                # 0..14 on board
            c = int(rng.integers(0, min(WIN_CAPTURES - 1, TOTAL_GOATS - 1 - k) + 1))
            n = k + c
        else:
            c = int(rng.integers(0, WIN_CAPTURES))
            k = TOTAL_GOATS - c
            n = TOTAL_GOATS
        tm, gm = _random_board(rng, k)
        _env_set(env_g, tm, gm, c, n)
        _env_set(env_t, tm, gm, c, n)
        # --- goats: actions and resulting boards
        ref = _env_goat_moves(env_g, tm, gm)
        rows = goat_successors(tm, gm, placing)
        mine = {(int(r[0]), int(r[2])) for r in rows}
        if len(rows) != len(mine) or mine != ref or any(int(r[1]) != tm for r in rows):
            err_g += 1
        nmov_g += len(ref)
        # goat successor indices vs literal rank of the env successor boards
        kk = k + 1 if placing else k
        ref_idx = sorted(rank_reference(tm, g2, kk) for _, g2 in ref)
        if sorted(int(x) for x in goat_successor_indices(tm, gm, k, placing)) != ref_idx:
            err_gi += 1
        # --- tigers: (action, from, landing, is_capture) exactly as _tiger_moves
        env_list = env_t._tiger_moves(include_dir=True)
        ref_moves = {(f * NDIR + d, f, to, int(cp)) for f, to, cp, d in env_list}
        rows = tiger_successors(tm, gm)
        mine_moves = set()
        for r in rows:
            a, ntm, ngm, cp = (int(v) for v in r)
            moved_to = ntm & ~tm
            land = moved_to.bit_length() - 1
            mine_moves.add((a, a // NDIR, land, cp))
        if len(rows) != len(mine_moves) or len(env_list) != len(ref_moves) or mine_moves != ref_moves:
            err_t += 1
        # resulting boards (captured goat = the q _tiger_moves jumped over)
        reft = _env_tiger_moves(env_t, tm, gm, "spec")
        if {tuple(int(v) for v in r) for r in rows} != reft:
            err_t += 1
        mask_t = env_t.get_action_mask()
        if set(int(a) for a in np.flatnonzero(mask_t)) != {r[0] for r in reft}:
            err_tmask += 1
        nmov_t += len(reft)
        ncap += sum(r[3] for r in reft)
        # how TnGEnv.step would resolve the same captures (informational)
        envt = _env_tiger_moves(env_t, tm, gm, "env_step")
        if envt != reft:
            n_div += 1
            if len(div_ex) < 3:
                diff = sorted(reft - envt)
                div_ex.append({"tigers": _squares(tm), "goats": _squares(gm),
                               "moves": [(a // NDIR, a % NDIR) for a, _, _, _ in diff]})
        cap_wins = c + 1 == WIN_CAPTURES
        ref_ti = []
        for _, t2, g2, cp in reft:
            if not cp:
                ref_ti.append((0, rank_reference(t2, g2, k)))
            elif cap_wins:
                ref_ti.append((2, -1))
            else:
                ref_ti.append((1, rank_reference(t2, g2, k - 1)))
        got_ti = sorted((int(r[0]), int(r[1])) for r in tiger_successor_indices(tm, gm, k, cap_wins))
        if got_ti != sorted(ref_ti):
            err_ti += 1
    # undirected-graph sanity (informational)
    undirected = all(any(q2 == s for q2 in MOVE_MAP[q].values())
                     for s, dd in MOVE_MAP.items() for q in dd.values() if q is not None)
    res = {"boards": nboards, "goat_moves_compared": nmov_g, "tiger_moves_compared": nmov_t,
           "tiger_captures_compared": int(ncap),
           "goat_move_errors": err_g, "tiger_move_errors": err_t,
           "tiger_action_mask_errors": err_tmask,
           "goat_successor_index_errors": err_gi, "tiger_successor_index_errors": err_ti,
           "graph_undirected": undirected,
           "env_step_capture_divergent_captures": env_capture_divergences(),
           "boards_where_env_step_capture_differs": n_div,
           "env_step_divergence_examples": div_ex,
           "seconds": round(time.time() - t, 1)}
    print(f"   {res}", flush=True)
    return res, err_g == 0 and err_t == 0 and err_ti == 0 and err_gi == 0 and err_tmask == 0


def _py_read_code(arr, kind, idx):
    return int(_read_code(arr, kind, idx))


def _py_combine(codes):
    if not codes:
        return 1
    losses = [cd >> 1 for cd in codes if cd > 0 and cd & 1]
    wins = [cd >> 1 for cd in codes if cd > 0 and not cd & 1]
    if losses:
        return 2 * (min(losses) + 1)
    if any(cd <= 0 for cd in codes):
        return 0
    return 2 * (max(wins) + 1) + 1


def selftest_sample(tabs, per_layer, seed=777):
    """Pure-Python recomputation of random positions of every existing table,
    using TnGEnv itself as the move generator and rank_reference for indices
    (independent of the kernel's move/index code paths).  Also counts sampled
    positions whose recomputed value would change if captures were resolved
    the way TnGEnv.step does (informational)."""
    print(f"[selftest] python/TnGEnv recomputation of {per_layer} random positions per table ...",
          flush=True)
    t = time.time()
    rng = np.random.default_rng(seed)
    env_g = env_tng.TnGEnv()
    env_t = env_tng.TnGEnv(learner_role=env_tng.TIGER_LEARNER)
    total = bad = 0
    n_env_diff_value = n_env_diff_any = 0
    bad_ex = []
    env_ex = []
    skipped = []

    def lookup_T(tm, gm, c, n_after):           # tigers to move after a goat move
        k = n_after - c if n_after < TOTAL_GOATS else TOTAL_GOATS - c
        if n_after == TOTAL_GOATS:
            return _py_read_code(tabs.mv(k, "T"), KIND_U16, rank_reference(tm, gm, k)), True
        offs = placement_offsets(n_after)
        a8 = tabs.u8(n_after, "T")
        idx = offs[c] + rank_reference(tm, gm, k)
        if a8 is not None:
            return _py_read_code(a8, KIND_U8, idx), True
        return _py_read_code(tabs.u2(n_after, "T"), KIND_U2, idx), False

    def lookup_G(tm, gm, c, n):                  # goats to move after a tiger move
        if n == TOTAL_GOATS:
            k = TOTAL_GOATS - c
            return _py_read_code(tabs.mv(k, "G"), KIND_U16, rank_reference(tm, gm, k)), True
        offs = placement_offsets(n)
        idx = offs[c] + rank_reference(tm, gm, n - c)
        a8 = tabs.u8(n, "G")
        if a8 is not None:
            return _py_read_code(a8, KIND_U8, idx), True
        return _py_read_code(tabs.u2(n, "G"), KIND_U2, idx), False

    for kind, num, mover in parse_layers("all"):
        try:
            for _ in range(per_layer):
                if kind == "mv":
                    n = TOTAL_GOATS
                    k = num
                    c = TOTAL_GOATS - k
                else:
                    n = num
                    c = int(rng.integers(0, min(n, WIN_CAPTURES - 1) + 1))
                    k = n - c
                tm, gm = _random_board(rng, k)
                if kind == "mv":
                    cur = _py_read_code(tabs.mv(k, mover), KIND_U16, rank_reference(tm, gm, k))
                    cur_dist = True
                else:
                    idx = placement_offsets(n)[c] + rank_reference(tm, gm, k)
                    a8 = tabs.u8(n, mover)
                    if a8 is not None:
                        cur, cur_dist = _py_read_code(a8, KIND_U8, idx), True
                    else:
                        cur, cur_dist = _py_read_code(tabs.u2(n, mover), KIND_U2, idx), False
                dist_ok = cur_dist
                exp_env = None
                if mover == "G":
                    codes = []
                    _env_set(env_g, tm, gm, c, n)
                    for _, g2 in _env_goat_moves(env_g, tm, gm):
                        code, dd = lookup_T(tm, g2, c, n + 1 if n < TOTAL_GOATS else n)
                        codes.append(code)
                        dist_ok = dist_ok and dd
                    exp = _py_combine(codes)
                else:
                    _env_set(env_t, tm, gm, c, n)
                    exps = {}
                    for sem in ("spec", "env_step"):
                        codes = []
                        for _, t2, g2, cp in _env_tiger_moves(env_t, tm, gm, sem):
                            if cp and c + 1 == WIN_CAPTURES:
                                codes.append(1)
                                continue
                            code, dd = lookup_G(t2, g2, c + cp, n)
                            codes.append(code)
                            dist_ok = dist_ok and dd
                        exps[sem] = _py_combine(codes)
                    exp, exp_env = exps["spec"], exps["env_step"]
                total += 1
                ok = (cur == exp) if dist_ok else (_wdl(cur) == _wdl(exp))
                if not ok:
                    bad += 1
                    if len(bad_ex) < 5:
                        bad_ex.append({"layer": f"{kind}{num}_{mover}", "tigers": _squares(tm),
                                       "goats": _squares(gm), "c": c, "n": n,
                                       "stored": _decode(cur), "expected": _decode(exp)})
                if exp_env is not None and exp_env != exp:
                    n_env_diff_any += 1
                    if _wdl(exp_env) != _wdl(exp):
                        n_env_diff_value += 1
                    if len(env_ex) < 3:
                        env_ex.append({"layer": f"{kind}{num}_{mover}", "tigers": _squares(tm),
                                       "goats": _squares(gm), "c": c, "n": n,
                                       "spec_rule": _decode(exp),
                                       "env_step_rule": _decode(exp_env)})
        except (FileNotFoundError, ValueError) as e:
            skipped.append(f"{kind}{num}_{mover}: {e}")
    res = {"positions": total, "mismatches": bad, "examples": bad_ex, "skipped": skipped,
           "env_step_rule_changes_value_or_distance": n_env_diff_any,
           "env_step_rule_changes_wdl_value": n_env_diff_value,
           "env_step_rule_examples": env_ex,
           "seconds": round(time.time() - t, 1)}
    print(f"   {res}", flush=True)
    return res, bad == 0


def selftest_mutation(tabs, chunk, ncorrupt=50, seed=99):
    """Non-vacuity: corrupt entries of an in-memory copy of a table and check
    the kernel reports exactly those positions."""
    print(f"[selftest] mutation test (corrupt {ncorrupt} entries, expect them reported) ...",
          flush=True)
    t = time.time()
    rng = np.random.default_rng(seed)
    out = {}
    allok = True
    cases = [("mv", 15, "G"), ("mv", 14, "T"), ("pl", 6, "T"), ("pl", 7, "G")]
    for kind, num, mover in cases:
        try:
            name, mode, cur_name, aux_name, blocks = build_job(tabs, kind, num, mover)
        except (FileNotFoundError, ValueError) as e:
            out[f"{kind}{num}_{mover}"] = f"skipped: {e}"
            continue
        clean = sum(run_block(b, chunk)["mismatches"] for b in blocks)
        cur = blocks[0]["cur"]
        ckind = blocks[0]["cur_kind"]
        npos = (cur.size // 2) if ckind == KIND_U16 else (
            cur.size if ckind == KIND_U8 else placement_offsets(num)[-1])
        bad = np.array(cur)            # writable copy
        picks = rng.choice(npos, ncorrupt, replace=False)
        for p in picks:
            p = int(p)
            if ckind == KIND_U16:
                v = int(bad[2 * p]) | (int(bad[2 * p + 1]) << 8)
                v2 = v + 2 if v < 400 else v - 2          # different code
                bad[2 * p], bad[2 * p + 1] = v2 & 255, v2 >> 8
            elif ckind == KIND_U8:
                bad[p] = (int(bad[p]) + 2) % 256
            else:
                sh = (p & 3) * 2
                v = (int(bad[p >> 2]) >> sh) & 3
                v2 = (v + 1) % 3
                bad[p >> 2] = (int(bad[p >> 2]) & ~(3 << sh) & 255) | (v2 << sh)
        bad.flags.writeable = False
        got = 0
        exidx = set()
        for b in blocks:
            b2 = dict(b, cur=bad, has_aux=False)
            r = run_block(b2, chunk)
            got += r["mismatches"]
            exidx.update(e[0] for e in r["examples"])
        ok = got == clean + ncorrupt and exidx <= set(int(p) for p in picks)
        allok = allok and ok
        out[name] = {"mode": mode, "clean_mismatches": clean, "corrupted": ncorrupt,
                     "reported": got, "ok": ok}
    out["seconds"] = round(time.time() - t, 1)
    print(f"   {out}", flush=True)
    return out, allok


# ============================================================================
#  main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layers", default="all",
                    help="comma list, e.g. mv15,mv10-14,pl0-8,pl13T,all (default all)")
    ap.add_argument("--selftest", action="store_true",
                    help="move generation vs TnGEnv + exhaustive rank/unrank + mutation test; "
                         "no full table check")
    ap.add_argument("--boards", type=int, default=50000, help="random boards for --selftest")
    ap.add_argument("--sample", type=int, default=300,
                    help="random positions per table for the --selftest python recomputation")
    ap.add_argument("--threads", type=int, default=0, help="numba threads (default: all)")
    ap.add_argument("--chunk", type=int, default=1 << 16, help="positions per parallel task")
    ap.add_argument("--tables", default=TABLE_DIR)
    ap.add_argument("--out", default=None,
                    help="result JSON (default <tables>/check_all.json)")
    args = ap.parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    out_path = args.out or os.path.join(args.tables, "check_all.json")
    tabs = Tables(args.tables)
    meta_path = os.path.join(args.tables, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    t_start = time.time()
    report = {"checker": os.path.abspath(__file__), "tables": os.path.abspath(args.tables),
              "rules_file": os.path.abspath(env_tng.__file__),
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "threads": numba.get_num_threads(), "threading_layer": None}
    failed = False

    # block offsets agree with metadata?
    off_ok = True
    for n in PL_RANGE:
        mo = meta.get("placement", {}).get(str(n), {}).get("block_offsets")
        if mo is not None and mo != placement_offsets(n):
            off_ok = False
            print(f"  WARNING: metadata block_offsets for n={n} differ from computed offsets")
    report["metadata_block_offsets_match"] = off_ok
    div = env_capture_divergences()
    report["env_step_capture_divergences"] = div
    for dv in div:
        print(f"WARNING: TnGEnv.step resolves the capture {dv['from']}->{dv['to']} (dir {dv['dir']}, "
              f"over {dv['over']}) with _find_jumped_goat -> {dv['env_find_jumped_goat']}; "
              f"this checker (and the table spec) removes the goat on {dv['over']}.", flush=True)

    if args.selftest:
        st = {}
        r, ok1 = selftest_index()
        st["index"] = r
        r, ok2 = selftest_moves(args.boards)
        st["moves"] = r
        r, ok3 = selftest_mutation(tabs, args.chunk)
        st["mutation"] = r
        tabs.release()
        r, ok4 = selftest_sample(tabs, args.sample)
        st["sample"] = r
        tabs.release()
        report["selftest"] = st
        report["selftest_passed"] = ok1 and ok2 and ok3 and ok4
        failed = not report["selftest_passed"]
        print(f"[selftest] {'PASSED' if not failed else 'FAILED'}", flush=True)
        out_path = args.out          # selftest writes JSON only when --out is given
    else:
        jobs = parse_layers(args.layers)
        report["layers_requested"] = args.layers
        report["results"] = []
        tot_pos = tot_bad = 0
        for kind, num, mover in jobs:
            try:
                res = run_job(tabs, kind, num, mover, args.chunk, meta)
            except (FileNotFoundError, ValueError) as e:
                print(f"{kind}{num}_{mover}: ERROR {e}", flush=True)
                report["results"].append({"layer": f"{kind}{num}_{mover}", "error": str(e)})
                failed = True
                continue
            if report["threading_layer"] is None:
                try:
                    report["threading_layer"] = numba.threading_layer()
                except ValueError:
                    pass
            report["results"].append(res)
            tot_pos += res["positions"]
            tot_bad += res["mismatches"] + res["aux_mismatches"]
            failed = failed or res["mismatches"] > 0 or res["aux_mismatches"] > 0 \
                or res["counts"]["invalid"] > 0
            md = "" if res["max_d"] is None else f" max_d={res['max_d']}"
            print(f"{res['layer']:>9}  {res['mode']:<38} positions={res['positions']:>13,}"
                  f"  mismatches={res['mismatches']}"
                  f"{'  aux_mismatches=%d' % res['aux_mismatches'] if 'consistency' in res['mode'] else ''}"
                  f"{md}  W/D/L={res['counts']['win']}/{res['counts']['draw']}/"
                  f"{res['counts']['loss']}  meta={res['metadata_counts_match']}"
                  f"  {res['seconds']:.1f}s ({res['positions_per_second'] / 1e6:.1f} M/s)",
                  flush=True)
            for ex in res["examples"]:
                print(f"      {ex}", flush=True)
            tabs.release()
        report["total_positions"] = tot_pos
        report["total_mismatches"] = tot_bad
        report["files_used"] = tabs.used
        print(f"TOTAL positions={tot_pos:,} mismatches={tot_bad}  "
              f"{time.time() - t_start:.0f}s  -> {'FAIL' if failed else 'OK'}", flush=True)
    report["seconds_total"] = round(time.time() - t_start, 1)
    report["ok"] = not failed
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=1,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        print(f"wrote {out_path}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
