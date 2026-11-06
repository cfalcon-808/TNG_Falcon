"""Throughput benchmark for a Tigers & Goats retrograde solver: successor generation + indexing + table reads.

Uses the exact move map and capture rules of env_tng.py (v1.1). Times one "pass" over random positions of the
12-goat movement layer, reading each successor's value from a full-size uint16 table, as a solver pass would.
usage (needs numba): python tools/dp_bench.py env_tng.py 5000000 16
Measured 2026-11-05 on a Ryzen 7 9800X3D (16 threads, machine partly loaded): 4.2M positions/s, 82M successor reads/s.
"""
import re
import sys
import time

import numba as nb
import numpy as np

env_src, n_pos, threads = open(sys.argv[1]).read(), int(sys.argv[2]), int(sys.argv[3])
nb.set_num_threads(threads)
start = env_src.index("self.move_map = {")
move_map = eval(env_src[start + len("self.move_map = "):env_src.index("\n        }", start) + 10].strip())
hub_jump = {1: 8, 2: 9, 3: 10, 4: 11}

N = -np.ones((23, 5), np.int64)   # neighbour in direction d (1..4)
L = -np.ones((23, 5), np.int64)   # capture landing in direction d, -1 when not allowed
for p, nbrs in move_map.items():
    for d, q in nbrs.items():
        N[p, d] = q
for p in range(23):
    for d in range(1, 5):
        q = N[p, d]
        if q < 0 or (p in (2, 3, 4, 5) and d == 1):
            continue
        L[p, d] = hub_jump[d] if p == 0 else N[q, d]
C = np.zeros((24, 24), np.int64)
for n in range(24):
    C[n, 0] = 1
    for k in range(1, n + 1):
        C[n, k] = C[n - 1, k - 1] + C[n - 1, k]


@nb.njit(cache=True)
def index(tmask, gmask, C):
    """Rank (tiger 3-set of 23, goat k-set of the other 20) -> dense index within the k-goat layer."""
    rt, i, j, rg, k = 0, 0, 0, 0, 0
    for p in range(23):
        if (tmask >> p) & 1:
            i += 1
            rt += C[p, i]
        else:
            if (gmask >> p) & 1:
                k += 1
                rg += C[j, k]
            j += 1
    return rt * C[20, k] + rg


@nb.njit(parallel=True, cache=True)
def one_pass(tmasks, gmasks, table, N, L, C, out_acc, out_cnt):
    for s in nb.prange(tmasks.size):
        tm, gm = tmasks[s], gmasks[s]
        occ = tm | gm
        acc = 0
        cnt = 0
        for p in range(23):                      # goat slides -> tiger to move, same layer
            if (gm >> p) & 1:
                for d in range(1, 5):
                    q = N[p, d]
                    if q >= 0 and not (occ >> q) & 1:
                        acc += table[index(tm, gm ^ (1 << p) ^ (1 << q), C)]
                        cnt += 1
        for p in range(23):                      # tiger slides and captures -> goat to move
            if (tm >> p) & 1:
                for d in range(1, 5):
                    q = N[p, d]
                    if q < 0:
                        continue
                    if not (occ >> q) & 1:
                        acc += table[index(tm ^ (1 << p) ^ (1 << q), gm, C)]
                        cnt += 1
                    elif (gm >> q) & 1:
                        land = L[p, d]
                        if land >= 0 and not (occ >> land) & 1:
                            # capture lands in the 11-goat layer; index range is smaller, reuse table for timing
                            acc += table[index(tm ^ (1 << p) ^ (1 << land), gm ^ (1 << q), C) % table.size]
                            cnt += 1
        out_acc[s] = acc
        out_cnt[s] = cnt


rng = np.random.default_rng(0)
layer = int(C[23, 3] * C[20, 12])
table = rng.integers(0, 65535, layer, dtype=np.uint16)          # full-size 12-goat layer, 446 MB
tm = np.zeros(n_pos, np.int64)
gm = np.zeros(n_pos, np.int64)
for s in range(n_pos):
    cells = rng.permutation(23)
    tm[s] = sum(1 << int(c) for c in cells[:3])
    gm[s] = sum(1 << int(c) for c in cells[3:15])
acc_out = np.zeros(n_pos, np.int64)
cnt_out = np.zeros(n_pos, np.int64)
one_pass(tm[:1000], gm[:1000], table, N, L, C, acc_out[:1000], cnt_out[:1000])   # compile
t0 = time.perf_counter()
one_pass(tm, gm, table, N, L, C, acc_out, cnt_out)
dt = time.perf_counter() - t0
reads = int(cnt_out.sum())
print(f"threads={threads} positions={n_pos:,} successors={reads:,} ({reads / n_pos:.1f}/position) "
      f"time={dt:.2f}s -> {n_pos / dt / 1e6:.1f}M positions/s, {reads / dt / 1e6:.0f}M successor reads/s")
