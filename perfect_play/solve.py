"""Solve Tigers & Goats exactly on the GPU and write the tables to perfect_play/tables/.

usage: python perfect_play/solve.py [--out perfect_play/tables]
Needs CuPy with a CUDA GPU. Movement layers are solved by retrograde analysis (cyclic), placement layers by
backward induction (acyclic). Every movement layer is re-verified on the GPU before it is written.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rules import ADJ, BIN, GOATS, L, N, REPO, block_size, placement_offsets, rank_board  # noqa: E402

THREADS = 256


def log(message, stream):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def launch(kernel, total, *args):
    kernel(((int(total) + THREADS - 1) // THREADS,), (THREADS,), args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "tables"))
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    stream = open(out / "solve_log.txt", "a")
    started = time.time()

    module = cp.RawModule(code=(Path(__file__).resolve().parent / "kernels.cu").read_text(), options=("-std=c++17",))
    for name, table, dtype in (("NB", N, np.int32), ("LD", L, np.int32), ("ADJ", ADJ, np.int32), ("BIN", BIN, np.int64)):
        ptr = module.get_global(name)
        cp.ndarray(table.size, dtype, ptr)[...] = cp.asarray(table.ravel().astype(dtype))
    k_init, k_level, k_ext, k_check, k_to_u8, k_goat, k_tiger, k_pack = (
        module.get_function(n) for n in ("mv_init", "mv_level", "mv_ext", "mv_check", "mv_to_u8", "pl_goat",
                                          "pl_tiger", "pack4"))
    meta = dict(rules_file="env_tng.py", rules_sha256=hashlib.sha256((REPO / "env_tng.py").read_bytes()).hexdigest(),
                gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(), movement={}, placement={})
    log(f"GPU {meta['gpu']}; output {out}", stream)

    # ── Movement phase: layers k = 10 (5 captured) .. 15 (none captured) ──────────────────────────────
    t15_off = placement_offsets(GOATS)                      # tigers-to-move, n = 15, blocks c = 0..5
    t15 = cp.zeros(t15_off[-1], cp.uint8)                    # movement tiger values, for the placement phase
    low_g = cp.zeros(1, cp.uint16)
    for k in range(10, 16):
        t0 = time.time()
        c = GOATS - k
        B = block_size(k)
        pad = (B + 3) // 4 * 4
        vG, vT = cp.zeros(B, cp.uint16), cp.zeros(B, cp.uint16)
        cG, cT = cp.zeros(pad, cp.uint8), cp.zeros(pad, cp.uint8)
        extW, extM = cp.zeros(B, cp.uint16), cp.zeros(B, cp.uint16)
        launch(k_init, B, np.int32(k), np.int64(B), vG, vT, cG, cT, extW, extM, low_g)
        bound = max(int(extW.max()), int(extM.max()) + 1, int((vG >> 1).max()), int((vT >> 1).max()))
        ext_max = int(extW.max())
        cG32, cT32 = cG.view(cp.uint32), cT.view(cp.uint32)
        level = 0
        while True:
            # positions resolved at distance `level` (code 0 = unresolved also has code >> 1 == 0)
            frontier = int(cp.count_nonzero((vG >> 1) == level) + cp.count_nonzero((vT >> 1) == level))
            if level == 0:
                frontier -= int(cp.count_nonzero(vG == 0) + cp.count_nonzero(vT == 0))
            if frontier:
                launch(k_level, B, np.int32(k), np.int64(B), np.uint32(level), vG, vT, cG32, cT32, extW, extM)
            if level + 1 <= ext_max:
                launch(k_ext, B, np.int64(B), np.uint32(level), vT, extW)
            if frontier == 0 and level > bound:
                break
            level += 1
        cp.cuda.Device().synchronize()
        bad = cp.zeros(B, cp.uint8)
        launch(k_check, B, np.int32(k), np.int64(B), vG, vT, low_g, bad)
        n_bad = int(cp.count_nonzero(bad))
        del bad, cG, cT, cG32, cT32, extW, extM
        if n_bad:
            log(f"layer k={k}: recursion check FAILED at {n_bad:,} boards; stopping", stream)
            sys.exit(1)

        stats = {}
        for side, v in (("G", vG), ("T", vT)):
            win = int(cp.count_nonzero((v != 0) & ((v & 1) == 0)))
            loss = int(cp.count_nonzero(v & 1))
            stats[side] = dict(win=win, loss=loss, draw=B - win - loss, max_dtm=int((v >> 1).max()))
            cp.asnumpy(v).tofile(out / f"mv_k{k}_{side}.u16")
        launch(k_to_u8, B, np.int64(B), vT, t15[t15_off[c]:t15_off[c + 1]])
        meta["movement"][str(k)] = dict(captured=c, boards=B, levels=level, seconds=round(time.time() - t0, 1), **stats)
        log(f"movement k={k} (c={c}): {B:,} boards, {level} levels, check OK, {time.time() - t0:.0f}s | "
            f"goats-to-move W/L/D {stats['G']['win']:,}/{stats['G']['loss']:,}/{stats['G']['draw']:,} | "
            f"tigers-to-move W/L/D {stats['T']['win']:,}/{stats['T']['loss']:,}/{stats['T']['draw']:,} | "
            f"max DTM {max(stats['G']['max_dtm'], stats['T']['max_dtm'])}", stream)
        del vT
        low_g = vG

    del low_g
    # ── Placement phase: n = 14 .. 0 goats placed ────────────────────────────────────────────────────
    t_next, t_next_off = t15, t15_off
    for n in range(GOATS - 1, -1, -1):
        t0 = time.time()
        off = placement_offsets(n)
        total = off[-1]
        off_d, next_off_d = cp.asarray(np.array(off, np.int64)), cp.asarray(np.array(t_next_off, np.int64))
        g = cp.empty(total, cp.uint8)
        launch(k_goat, total, np.int32(n), np.int64(total), off_d, np.int32(len(off) - 1), next_off_d, t_next, g)
        t = cp.empty(total, cp.uint8)
        launch(k_tiger, total, np.int32(n), np.int64(total), off_d, np.int32(len(off) - 1), g, t)
        stats = {}
        for side, v in (("G", g), ("T", t)):
            counts = cp.bincount(v, minlength=3).get()
            stats[side] = dict(draw=int(counts[0]), win=int(counts[1]), loss=int(counts[2]))
            packed = cp.zeros((total + 3) // 4, cp.uint8)
            launch(k_pack, packed.size, np.int64(total), v, packed)
            cp.asnumpy(packed).tofile(out / f"pl_n{n}_{side}.u2")
            del packed
        meta["placement"][str(n)] = dict(positions=total, block_offsets=off, seconds=round(time.time() - t0, 1), **stats)
        log(f"placement n={n}: {total:,} positions per side, {time.time() - t0:.0f}s | goats-to-move W/L/D "
            f"{stats['G']['win']:,}/{stats['G']['loss']:,}/{stats['G']['draw']:,} | tigers-to-move W/L/D "
            f"{stats['T']['win']:,}/{stats['T']['loss']:,}/{stats['T']['draw']:,}", stream)
        if n == 0:
            start = rank_board(sum(1 << p for p in (0, 3, 4)), 0, 0)
            names = {0: "draw", 1: "mover wins", 2: "mover loses"}
            meta["opening"] = dict(goats_to_move=names[int(g[start].get())], tigers_to_move=names[int(t[start].get())])
        del g, t_next
        t_next, t_next_off = t, off

    meta["seconds_total"] = round(time.time() - started, 1)
    meta["encoding"] = {
        "mv_k{k}_{G|T}.u16": "uint16 per board of the k-goat movement layer (15 - k captured), index "
                             "rank3(tigers)*C(20,k) + rank_k(goats); 0 draw, 2d mover wins in d plies, "
                             "2d+1 mover loses in d plies",
        "pl_n{n}_{G|T}.u2": "2 bits per position of placement layer n (n goats placed), blocks c = 0..min(n,5) "
                            "at block_offsets, each indexed like a movement layer with k = n - c; "
                            "0 draw, 1 mover wins, 2 mover loses; position i in bits 2*(i%4) of byte i//4",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['seconds_total'] / 60:.1f} min; opening: {meta['opening']}", stream)


if __name__ == "__main__":
    main()
