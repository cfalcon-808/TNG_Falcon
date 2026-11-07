"""Positions reachable from the opening, and the longest forced wins a real game can reach.

usage: python perfect_play/reachable.py [--first goat tiger] [--tables perfect_play/tables]

The tables hold every arrangement of the pieces, including many that no game can reach: the tigers start on
b0, c1 and d1, and in the placement phase they have made only n or n + 1 moves after n goats are placed.
This script finds the reachable positions for games where the goats move first and where the tigers do:
- placement phase: one forward pass per layer (every goat turn places a goat, so it cannot cycle);
- movement phase: breadth-first search, which also records the fewest plies from the opening.
For every reachable position it reads the exact value from the tables. It reports the longest forced win
for each winner, with and without TnGEnv's 200-ply limit (judged at the fewest plies that reach the
position), and rebuilds a full game for each: a shortest route from the opening to the position, then
perfect play (winner fastest, loser slowest) to the end. Results go to tables/reachable.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oracle import PerfectPlay, Position  # noqa: E402
from rules import (BIN, GOATS, L, N, block_size, goat_moves, placement_offsets, popcount,  # noqa: E402
                   rank_board, tiger_moves)

THREADS = 256
UNREACHED = 0xFFFF
PLY_LIMIT = 200
START_TM = sum(1 << p for p in (0, 3, 4))
KS = range(10, 16)
SIDE = {"G": "goat", "T": "tiger"}
CHUNK = 1 << 28
CATEGORIES = [(w, lim) for w in ("goat", "tiger") for lim in (False, True)]


def launch(kernel, total, *args):
    kernel(((int(total) + THREADS - 1) // THREADS,), (THREADS,), args)


def plies_before(n, side, first):
    """Plies played before a placement position (n goats placed, side to move); fixed in that phase."""
    if first == "goat":
        p = 2 * n if side == "G" else 2 * n - 1
    else:
        p = 2 * n if side == "T" else 2 * n + 1
    return max(p, 0)


class Best:
    """Longest forced win per (winner, within the limit?); ties go to the position reached soonest."""
    def __init__(self):
        self.best = {cat: None for cat in CATEGORIES}

    def offer(self, cat, d, depth, where):
        cur = self.best[cat]
        if cur is None or (d, -depth) > (cur["plies_to_end"], -cur["reached_at_ply"]):
            self.best[cat] = dict(plies_to_end=int(d), reached_at_ply=int(depth), **where)


class Reach:
    def __init__(self, tables, first, module, log):
        self.tables, self.first, self.log = tables, first, log
        self.k_goat, self.k_tiger, self.k_mvg, self.k_mvt = (module.get_function(n) for n in (
            "fw_pl_goat", "fw_pl_tiger", "fw_mv_goat", "fw_mv_tiger"))
        self.off = {n: placement_offsets(n) for n in range(GOATS + 1)}
        self.pl_bits = {}            # (n, side) -> packed reachable flags (host)
        self.depth = {}              # (k, side) -> uint16 plies from the opening (host)
        self.best = Best()
        self.counts = {}

    # ── placement ─────────────────────────────────────────────────────────────────────────────────────
    def _goat_step(self, n, rG):
        offT = self.off[n + 1]
        rT = cp.zeros(offT[-1], cp.uint8)
        launch(self.k_goat, rG.size, np.int32(n), np.int64(rG.size), cp.asarray(np.array(self.off[n], np.int64)),
               np.int32(len(self.off[n]) - 1), cp.asarray(np.array(offT, np.int64)), rG, rT)
        return rT

    def _tiger_step(self, n, rT):
        rG = cp.zeros(self.off[n][-1], cp.uint8)
        off = cp.asarray(np.array(self.off[n], np.int64))
        launch(self.k_tiger, rT.size, np.int32(n), np.int64(rT.size), off, np.int32(len(self.off[n]) - 1), rT, rG)
        return rG

    def _placement_stats(self, n, side, flags):
        values = np.memmap(self.tables / f"pl_n{n}_{side}.u8", np.uint8, "r")
        depth = plies_before(n, side, self.first)
        cnt = dict(reachable=0, mover_wins=0, mover_loses=0, draws=0, at_ply=depth)
        for lo in range(0, flags.size, CHUNK):                 # chunks keep GPU memory small
            r = flags[lo:lo + CHUNK].astype(cp.bool_)
            v = cp.asarray(values[lo:lo + CHUNK])
            forced = r & (v > 0)
            d = v.astype(cp.int16) - 1
            mover_wins = (d & 1) == 1
            cnt["reachable"] += int(r.sum())
            cnt["mover_wins"] += int((forced & mover_wins).sum())
            cnt["mover_loses"] += int((forced & ~mover_wins).sum())
            cnt["draws"] += int((r & (v == 0)).sum())
            for winner, lim in CATEGORIES:
                mask = forced & (mover_wins if SIDE[side] == winner else ~mover_wins)
                if lim:
                    mask &= d <= PLY_LIMIT - depth
                if not bool(mask.any()):
                    continue
                dd = cp.where(mask, d, cp.int16(-1))
                i = int(dd.argmax())
                self.best.offer((winner, lim), int(dd[i]), depth,
                                dict(phase="placement", placed=n, side=side, index=lo + i))
        self.counts[f"pl_n{n}_{side}"] = cnt
        self.pl_bits[(n, side)] = cp.packbits(flags).get()

    def placement(self):
        start = rank_board(START_TM, 0, 0)
        empty0 = cp.zeros(self.off[0][-1], cp.uint8)
        if self.first == "goat":
            rG = empty0.copy()
            rG[start] = 1
            self._placement_stats(0, "T", empty0)
        else:
            rT = empty0.copy()
            rT[start] = 1
            self._placement_stats(0, "T", rT)
            rG = self._tiger_step(0, rT)
            del rT
        for n in range(GOATS):
            self._placement_stats(n, "G", rG)
            rT = self._goat_step(n, rG)
            del rG
            if n + 1 == GOATS:
                return rT                                   # tigers to move, movement phase, by blocks c = 0..5
            self._placement_stats(n + 1, "T", rT)
            rG = self._tiger_step(n + 1, rT)
            del rT
            self.log(f"  {self.first} first: placement layer {n + 1} done")

    # ── movement ──────────────────────────────────────────────────────────────────────────────────────
    def movement(self, holder):
        entry_flags = holder.pop()
        entry = plies_before(GOATS, "T", self.first)
        dG = {k: cp.full(block_size(k), UNREACHED, cp.uint16) for k in KS}
        dT = {k: cp.full(block_size(k), UNREACHED, cp.uint16) for k in KS}
        off15 = self.off[GOATS]
        for c in range(6):
            k = GOATS - c
            dT[k][entry_flags[off15[c]:off15[c + 1]].astype(cp.bool_)] = entry
        del entry_flags
        cp.get_default_memory_pool().free_all_blocks()
        added = cp.zeros(1, cp.uint64)
        r = entry
        while True:
            added[...] = 0
            for k in KS:
                B = block_size(k)
                launch(self.k_mvg, B, np.int32(k), np.int64(B), np.uint32(r), dG[k], dT[k], added)
                launch(self.k_mvt, B, np.int32(k), np.int64(B), np.uint32(r), dT[k], dG[k], dG[max(k - 1, 10)], added)
            if int(added.get()[0]) == 0:
                break
            r += 1
            assert r < UNREACHED - 1
        self.log(f"  {self.first} first: movement search done, deepest position first reached at ply {r}")
        for k in KS:
            for side, arr in (("G", dG[k]), ("T", dT[k])):
                values = np.memmap(self.tables / f"mv_k{k}_{side}.u16", np.uint16, "r")
                cnt = dict(reachable=0, mover_wins=0, mover_loses=0, draws=0, fewest_plies_max=0)
                for lo in range(0, arr.size, CHUNK // 4):      # chunks keep GPU memory small
                    v = cp.asarray(values[lo:lo + CHUNK // 4])
                    depth = arr[lo:lo + CHUNK // 4].astype(cp.int32)
                    reached = depth != UNREACHED
                    forced = reached & (v > 0)
                    d = (v >> 1).astype(cp.int32)
                    mover_wins = (v & 1) == 0
                    cnt["reachable"] += int(reached.sum())
                    cnt["mover_wins"] += int((forced & mover_wins).sum())
                    cnt["mover_loses"] += int((forced & ~mover_wins).sum())
                    cnt["draws"] += int((reached & (v == 0)).sum())
                    if bool(reached.any()):
                        cnt["fewest_plies_max"] = max(cnt["fewest_plies_max"], int(depth[reached].max()))
                    for winner, lim in CATEGORIES:
                        mask = forced & (mover_wins if SIDE[side] == winner else ~mover_wins)
                        if lim:
                            mask &= depth + d <= PLY_LIMIT
                        if not bool(mask.any()):
                            continue
                        key = cp.where(mask, d * 65536 + (65535 - depth), -1)
                        i = int(key.argmax())
                        self.best.offer((winner, lim), int(d[i]), int(depth[i]),
                                        dict(phase="movement", k=k, side=side, index=lo + i))
                    del v, depth, reached, forced, d, mover_wins
                self.counts[f"mv_k{k}_{side}"] = cnt
                self.depth[(k, side)] = arr.get()
        del dG, dT
        return r

    # ── rebuilding games ──────────────────────────────────────────────────────────────────────────────
    def _pl_reached(self, pos):
        side = "G" if pos.mover == "goat" else "T"
        c, n = pos.captured, pos.placed
        idx = self.off[n][c] + rank_board(pos.tm, pos.gm, n - c)
        return (self.pl_bits[(n, side)][idx >> 3] >> (7 - (idx & 7))) & 1

    def _mv_depth(self, pos):
        k = popcount(pos.gm)
        return int(self.depth[(k, "G" if pos.mover == "goat" else "T")][rank_board(pos.tm, pos.gm, k)])

    def route(self, target):
        """Moves from the opening to `target`, walking back through reachable predecessors."""
        start = Position(START_TM, 0, 0, 0, self.first)
        moves, pos = [], target
        while pos != start:
            if pos.placing:
                ok = lambda p: p.placing and self._pl_reached(p)  # noqa: E731
            else:
                d = self._mv_depth(pos)
                ok = (lambda p, d=d: (not p.placing and self._mv_depth(p) == d - 1)
                      or (p.placing and d == plies_before(GOATS, "T", self.first) and self._pl_reached(p)))
            for pred in candidate_predecessors(pos):
                if ok(pred):
                    mv = [m for m in moves_of(pred) if m[2] == pos]
                    if mv:
                        moves.append(mv[0])
                        pos = pred
                        break
            else:
                raise RuntimeError(f"no reachable predecessor for {pos}")
        return moves[::-1]

    def target_position(self, where):
        if where["phase"] == "placement":
            n, side, i = where["placed"], where["side"], where["index"]
            c = max(j for j in range(len(self.off[n]) - 1) if self.off[n][j] <= i)
            tm, gm = unrank(i - self.off[n][c], n - c)
            return Position(tm, gm, c, n, SIDE[side])
        tm, gm = unrank(where["index"], where["k"])
        return Position(tm, gm, GOATS - where["k"], GOATS, SIDE[where["side"]])


def unrank(index, k):
    from rules import unrank_board
    return unrank_board(index, k)


def moves_of(pos):
    """[(action, text with square numbers, next position or None when the move wins at once)]."""
    out = []
    if pos.mover == "goat":
        placed = pos.placed + 1 if pos.placing else GOATS
        for a, g2 in goat_moves(pos.tm, pos.gm, pos.placing):
            p, d = divmod(a, 5)
            text = f"@{p}" if pos.placing else f"{p}-{int(N[p, d])}"
            out.append((a, text, Position(pos.tm, g2, pos.captured, placed, "tiger")))
        return out
    for a, t2, g2, took in tiger_moves(pos.tm, pos.gm):
        p, d = divmod(a, 5)
        c2 = pos.captured + took
        text = f"{p}x{int(L[p, d])}" if took else f"{p}-{int(N[p, d])}"
        out.append((a, text, None if c2 >= 6 else Position(t2, g2, c2, pos.placed, "goat")))
    return out


def candidate_predecessors(pos):
    """Positions one ply earlier that could lead to `pos` (checked against the move list by the caller)."""
    tm, gm, c, n = pos.tm, pos.gm, pos.captured, pos.placed
    occ = tm | gm
    out = []
    if pos.mover == "tiger":                       # the goats just moved
        for g in range(23):
            if not (gm >> g) & 1:
                continue
            if n >= 1:                             # undo a placement: the goat on g was placed last
                out.append(Position(tm, gm ^ (1 << g), c, n - 1, "goat"))
            if not pos.placing:                    # undo a slide into g
                for d in range(1, 5):
                    q = int(N[g, d])
                    if q >= 0 and not (occ >> q) & 1:
                        out.append(Position(tm, gm ^ (1 << g) ^ (1 << q), c, GOATS, "goat"))
        return out
    for t in range(23):                            # the tigers just moved
        if not (tm >> t) & 1:
            continue
        for d in range(1, 5):
            q = int(N[t, d])
            if q >= 0 and not (occ >> q) & 1:      # undo a slide q -> t
                out.append(Position(tm ^ (1 << t) ^ (1 << q), gm, c, n, "tiger"))
        if c >= 1:                                  # undo a capture s -> t over m
            for s in range(23):
                if (occ >> s) & 1:
                    continue
                for d in range(1, 5):
                    m = int(N[s, d])
                    if int(L[s, d]) == t and m >= 0 and not (occ >> m) & 1:
                        out.append(Position(tm ^ (1 << t) ^ (1 << s), gm | (1 << m), c - 1, n, "tiger"))
    return out


def perfect_line(pp, pos):
    """Perfect play from a forced-win position to the end: [(action, text, next)]."""
    out = []
    while pos is not None:
        rows = pp.move_values(pos)
        if not rows:
            break
        action = rows[0][0]
        mv = next(m for m in moves_of(pos) if m[0] == action)
        out.append(mv)
        pos = mv[2]
    return out


def game_text(first, moves, result):
    parts, turn = [], 1
    for i, (_a, text, _n) in enumerate(moves):
        side = first if i % 2 == 0 else ("tiger" if first == "goat" else "goat")
        if side == first:
            parts.append(f"{turn}.")
            turn += 1
        parts.append(text)
    return "\n".join([f"start: tigers=0,3,4 goats= captured=0 mover={first}", f"first: {first}",
                      "moves: " + " ".join(parts), f"result: {result}"])


def describe(pos):
    sq = lambda m: ",".join(str(p) for p in range(23) if (m >> p) & 1)  # noqa: E731
    return f"tigers={sq(pos.tm)} goats={sq(pos.gm)} captured={pos.captured} mover={pos.mover}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", nargs="+", default=["goat", "tiger"], choices=["goat", "tiger"])
    ap.add_argument("--tables", default=str(HERE / "tables"))
    args = ap.parse_args()
    tables = Path(args.tables)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    src = (HERE / "kernels.cu").read_text() + "\n" + (HERE / "reach.cu").read_text()
    module = cp.RawModule(code=src, options=("-std=c++17",))
    for name, table, dtype in (("NB", N, np.int32), ("LD", L, np.int32), ("BIN", BIN, np.int64)):
        cp.ndarray(table.size, dtype, module.get_global(name))[...] = cp.asarray(table.ravel().astype(dtype))
    pp = PerfectPlay(tables)
    out_path = tables / "reachable.json"
    report = json.loads(out_path.read_text()) if out_path.exists() else {}
    for first in args.first:
        t0 = time.time()
        rc = Reach(tables, first, module, log)
        holder = [rc.placement()]                  # movement() drops the last reference to free GPU memory
        cp.get_default_memory_pool().free_all_blocks()
        deepest = rc.movement(holder)
        cp.get_default_memory_pool().free_all_blocks()
        longest = {}
        for (winner, lim), where in rc.best.best.items():
            if where is None:
                continue
            target = rc.target_position(where)
            route = rc.route(target)
            line = perfect_line(pp, target)
            assert len(route) == where["reached_at_ply"] - plies_before(0, "G" if first == "goat" else "T", first)
            assert len(line) == where["plies_to_end"], (len(line), where)
            last = line[-1]
            total = len(route) + len(line)
            loser = "tigers" if winner == "goat" else "goats"
            how = "6th goat captured" if last[2] is None else f"the {loser} have no legal move"
            result = f"{'Goats' if winner == 'goat' else 'Tigers'} win ({how}) after {total} plies"
            key = f"{winner}s_win" + ("_within_200_plies" if lim else "")
            longest[key] = dict(
                plies_to_end=where["plies_to_end"], reached_at_ply=where["reached_at_ply"], total_plies=total,
                position=describe(target), game=game_text(first, route + line, result))
            log(f"  {first} first, {key}: {where['plies_to_end']} plies from ply {where['reached_at_ply']} "
                f"(game of {total} plies) | {describe(target)}")
        report[f"{first}_first"] = dict(
            seconds=round(time.time() - t0, 1), deepest_first_reach_ply=deepest,
            reachable_positions=sum(v["reachable"] for v in rc.counts.values()),
            counts=rc.counts, longest_forced_wins=longest)
        out_path.write_text(json.dumps(report, indent=2))
        log(f"{first} first done in {time.time() - t0:.0f}s: "
            f"{report[f'{first}_first']['reachable_positions']:,} reachable positions")
        del rc
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
