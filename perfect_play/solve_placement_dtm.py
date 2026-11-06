"""Exact plies-to-the-end for every placement position, on the GPU.

usage: python perfect_play/solve_placement_dtm.py [--tables perfect_play/tables] [--delete-u2]
Needs the movement tables written by solve.py (mv_k*_{G|T}.u16). Writes pl_n{n}_{G|T}.u8, one byte per position
of placement layer n (blocks by captured count c = 0..min(n, 5) at metadata.json placement.{n}.block_offsets,
each indexed like a movement layer with k = n - c):
  0 = draw, otherwise d + 1 where d = plies to the end with optimal play (winner fastest, loser slowest).
  d odd: the mover wins; d even: the mover loses (the winner always makes the last move).
The placement phase cannot cycle, so one backward pass from 14 goats placed down to 0 is exact. Each layer's
win/draw/loss is compared with the 2-bit table from solve.py (pl_n*_{G|T}.u2) when that file exists; the run
stops on any disagreement. --delete-u2 removes the 2-bit tables (4.2 GB) once every layer has passed, since the
.u8 tables hold the same values. Results are added to metadata.json["placement_dtm"].
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import BIN, GOATS, L, N, block_size, placement_offsets, rank_board  # noqa: E402

THREADS = 256
PLY_LIMIT = 200          # TnGEnv default: 100 turns, each a learner move plus the opponent's reply


def launch(kernel, total, *args):
    kernel(((int(total) + THREADS - 1) // THREADS,), (THREADS,), args)


def plies_before(n, side, first):
    """Plies already played when a placement position is reached: fixed, since every goat turn places."""
    if first == "goat":
        return 2 * n if side == "G" else 2 * n - 1
    return 2 * n if side == "T" else 2 * n + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default=str(Path(__file__).resolve().parent / "tables"))
    parser.add_argument("--delete-u2", action="store_true",
                        help="delete the 2-bit placement tables after every layer matched them")
    args = parser.parse_args()
    tables = Path(args.tables)
    log_file = open(tables / "solve_log.txt", "a")

    def log(message):
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    started = time.time()
    module = cp.RawModule(code=(Path(__file__).resolve().parent / "kernels.cu").read_text(), options=("-std=c++17",))
    for name, table, dtype in (("NB", N, np.int32), ("LD", L, np.int32), ("BIN", BIN, np.int64)):
        ptr = module.get_global(name)
        cp.ndarray(table.size, dtype, ptr)[...] = cp.asarray(table.ravel().astype(dtype))
    k_conv, k_goat, k_tiger, k_cmp = (module.get_function(n) for n in
                                      ("mv_to_dtm8", "pl_goat_dtm", "pl_tiger_dtm", "cmp_u2"))
    meta_path = tables / "metadata.json"
    meta = json.loads(meta_path.read_text())
    out = dict(encoding="uint8 per position of placement layer n (n goats placed), blocks c = 0..min(n,5) at "
                        "placement.{n}.block_offsets, each indexed like a movement layer with k = n - c; 0 draw, "
                        "d + 1 otherwise, d = plies to the end (odd: mover wins, even: mover loses)",
               ply_limit=PLY_LIMIT, layers={})
    log("placement ply counts: start")

    # movement tigers-to-move values as ply codes, one block per captured count (the n = 15 "layer")
    t_off = placement_offsets(GOATS)
    t_next = cp.zeros(t_off[-1], cp.uint8)
    parity_bad = cp.zeros(1, cp.uint64)
    for c in range(GOATS - 9):
        k = GOATS - c
        v = cp.asarray(np.fromfile(tables / f"mv_k{k}_T.u16", np.uint16))
        assert v.size == block_size(k)
        launch(k_conv, v.size, np.int64(v.size), v, t_next[t_off[c]:t_off[c + 1]], parity_bad)
        del v
    for k in range(10, 16):                        # parity rule on the goats-to-move movement tables too
        v = cp.asarray(np.fromfile(tables / f"mv_k{k}_G.u16", np.uint16))
        tmp = cp.empty(v.size, cp.uint8)
        launch(k_conv, v.size, np.int64(v.size), v, tmp, parity_bad)
        del v, tmp
    assert int(parity_bad.get()[0]) == 0, "movement tables break the win-in-odd-plies rule"
    log("movement tables: win/loss parity rule holds for every entry")

    longest = (0, None)
    checked = []                                   # (n, side) layers compared with the 2-bit table
    for n in range(GOATS - 1, -1, -1):
        t0 = time.time()
        off = placement_offsets(n)
        total = off[-1]
        off_d, next_off_d = cp.asarray(np.array(off, np.int64)), cp.asarray(np.array(t_off, np.int64))
        max_code = cp.zeros(1, cp.uint32)
        g = cp.empty(total, cp.uint8)
        launch(k_goat, total, np.int32(n), np.int64(total), off_d, np.int32(len(off) - 1), next_off_d, t_next, g,
               max_code)
        t = cp.empty(total, cp.uint8)
        launch(k_tiger, total, np.int32(n), np.int64(total), off_d, np.int32(len(off) - 1), g, t, max_code)
        assert int(max_code.get()[0]) <= 255, f"ply code overflow in layer {n}"
        layer = {}
        for side, v in (("G", g), ("T", t)):
            u2_path = tables / f"pl_n{n}_{side}.u2"
            if u2_path.exists():
                packed = cp.asarray(np.fromfile(u2_path, np.uint8))
                bad = cp.zeros(1, cp.uint64)
                launch(k_cmp, total, np.int64(total), v, packed, bad)
                n_bad = int(bad.get()[0])
                del packed
                if n_bad:
                    log(f"placement n={n} {side}: {n_bad:,} positions disagree with the 2-bit table; stopping")
                    sys.exit(1)
                checked.append((n, side))
            hist = cp.bincount(v, minlength=256).get()
            d = np.arange(255)                                   # hist[d + 1] = positions with d plies to go
            win, loss = int(hist[1:][d % 2 == 1].sum()), int(hist[1:][d % 2 == 0].sum())
            max_d = int(np.nonzero(hist)[0].max()) - 1 if hist[1:].any() else None
            over = {first: int(hist[1:][d > PLY_LIMIT - plies_before(n, side, first)].sum())
                    for first in ("goat", "tiger") if plies_before(n, side, first) >= 0}
            layer[side] = dict(win=win, loss=loss, draw=int(hist[0]), max_dtm=max_d,
                               beyond_ply_limit=over,
                               dtm_histogram={str(i - 1): int(x) for i, x in enumerate(hist) if i and x})
            if max_d is not None and max_d > longest[0]:
                longest = (max_d, dict(placed=n, mover="goats" if side == "G" else "tigers"))
            cp.asnumpy(v).tofile(tables / f"pl_n{n}_{side}.u8")
        out["layers"][str(n)] = layer
        check = "2-bit check OK" if (n, "G") in checked and (n, "T") in checked else "no 2-bit table to check"
        log(f"placement n={n}: {total:,} positions per side, {time.time() - t0:.0f}s, {check} | "
            f"max plies goats-to-move {layer['G']['max_dtm']}, tigers-to-move {layer['T']['max_dtm']}")
        if n == 0:
            start = rank_board(sum(1 << p for p in (0, 3, 4)), 0, 0)
            out["opening"] = dict(goats_to_move=int(g[start].get()), tigers_to_move=int(t[start].get()))
        del g, t_next
        t_next, t_off = t, off

    out["longest_placement_dtm"] = dict(plies=longest[0], **(longest[1] or {}))
    out["seconds"] = round(time.time() - started, 1)
    meta["placement_dtm"] = out
    meta["encoding"]["pl_n{n}_{G|T}.u8"] = out["encoding"]
    if args.delete_u2:
        if len(checked) == 2 * GOATS:
            for n in range(GOATS):
                for side in "GT":
                    (tables / f"pl_n{n}_{side}.u2").unlink()
            meta["encoding"]["pl_n{n}_{G|T}.u2"] += " (deleted after matching the .u8 tables; solve.py rewrites them)"
            log("2-bit placement tables matched on every layer and were deleted")
        else:
            log("--delete-u2: not every layer had a 2-bit table to check, nothing deleted")
    meta_path.write_text(json.dumps(meta, indent=2))
    log(f"placement ply counts done in {out['seconds']:.0f}s; longest {out['longest_placement_dtm']}; "
        f"opening codes {out['opening']}")


if __name__ == "__main__":
    main()
