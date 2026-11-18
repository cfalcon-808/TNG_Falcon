"""Models against each other and against perfect play, with every move judged by the solved tables.

usage (needs torch + sb3-contrib for the models): python perfect_play/compare_models.py [--games 200]

Every goat-side player meets every tiger-side player. Games start from the opening with the goats to move
and use TnGEnv's limits (200 plies; the 4th repeat of a board with the goats to move). Models play their
deterministic masked move, as in training and evaluation. Before and after every move the exact value of
the position is read from the tables, so a move is either value-preserving or a blunder:
  W>D  threw away a forced win for a draw      W>L  turned a forced win into a forced loss
  D>L  turned a draw into a forced loss
Two further experiments:
- models against models from balanced random openings: the first 4 plies are drawn uniformly from the moves
  that keep the draw, so two deterministic models play many different games (those plies are not scored);
- the effect of the capture fix: goat models against the scripted tigers under the fixed rules and under the
  capture rule the models were trained with (a tiger jumping d1 over c1 to b1 removed the goat on b0, or none).
Results go to tables/model_comparison.json.
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

GOATS = {
    "Perfect": ("perfect", None),
    "Perfect (plain)": ("plain", None),
    "Random": ("random", None),
    "NormalGoat": ("model", "stable_models/models/Goats/mppo_NormalGoat030726.zip"),
    "SmartGoat": ("model", "stable_models/models/Goats/mppo_SmartGoat030726.zip"),
    "RobustGoat": ("model", "stable_models/models/Goats/mppo_RobustGoat030726.zip"),
    "PPO+nll 0.5 (10M)": ("model", "artifacts/legality_study/ppo_clip_nll05_s42/baseline/models/phase_01_final.zip"),
    "ReLU Maskable (10M)": ("model", "artifacts/legality_study/relu_maskable_s42/baseline/models/phase_01_final.zip"),
}
TIGERS = {
    "Perfect": ("perfect", None),
    "Perfect (plain)": ("plain", None),
    "Random": ("random", None),
    "Greedy": ("greedy", None),
    "Smart": ("smart", None),
    "Tiger vs NormalGoat": ("model", "stable_models/models/Tigers/best_tiger_vs_normal_goat_10M_20260307_220408.zip"),
    "Tiger vs SmartGoat": ("model", "stable_models/models/Tigers/best_tiger_vs_smart_goat_10M_20260307_225859.zip"),
    "RobustTiger": ("model", "stable_models/models/Tigers/mppo_RobustTiger030726.zip"),
}
DETERMINISTIC = {"model"}
RANK = {"win": 0, "draw": 1, "loss": 2}
DROP = {(0, 1): "W>D", (0, 2): "W>L", (1, 2): "D>L"}
_state = {}


def _setup():
    if _state:
        return _state
    import torch
    torch.set_num_threads(1)
    import env_tng
    from tngviz import game, models
    _state.update(game=game, models=models, an=game.Analyzer(), env=env_tng.TnGEnv(), policies={}, env_tng=env_tng)
    return _state


def _policy(path):
    st = _setup()
    if path not in st["policies"]:
        st["policies"][path] = st["models"].load_policy(REPO / path)
    return st["policies"][path]


def choose(kind, path, pos, moves, seen, rng):
    st = _setup()
    game, an = st["game"], st["an"]
    if kind == "random":
        return rng.choice(moves)
    if kind == "model":
        policy = _policy(path)
        mask = np.zeros(115, dtype=bool)
        by_action = {m.action: m for m in moves}
        mask[list(by_action)] = True
        import torch
        obs_tensor, _ = policy.obs_to_tensor(st["models"].observation(pos))
        with torch.no_grad():                         # masked argmax over the raw logits (plain PPO has no mask)
            logits = policy.get_distribution(obs_tensor).distribution.logits[0].numpy()
        return by_action[int(np.where(mask, logits, -np.inf).argmax())]
    if kind in ("greedy", "smart"):
        env = st["env"]
        for p in range(23):
            env.board[p] = 2 if (pos.tm >> p) & 1 else 1 if (pos.gm >> p) & 1 else 0
        env.eaten, env.goats_placed, env.phase = pos.captured, pos.placed, 0 if pos.placing else 1
        options = env._tiger_moves()
        pick = env._greedy_tiger_move(options) if kind == "greedy" else env._smart_tiger_move(options)
        return next(m for m in moves if m.frm == pick[0] and m.to == pick[1])
    analysis = an.analyse(pos, 0, False) if kind == "perfect" else game.Analysis(
        pos, *an.value(pos), False, an.lines(pos, 0, False, traps=False))
    if kind == "perfect":
        return an.choose(analysis, seen, rng)        # fastest win, slowest loss; draws: unseen, then most traps
    return rng.choice(analysis.best).move            # plain: uniform over the equally good moves


def legacy_capture(game, pos, mv):
    """The pre-fix TnGEnv capture d1 x b1: the goat on c1 stays; the goat on b0 is removed if there is one."""
    import dataclasses
    if not (mv.side == "tiger" and mv.frm == 4 and mv.to == 2 and mv.jumped == 3):
        return mv
    hub = pos.gm & 1
    c2 = pos.captured + hub
    tm = pos.tm ^ (1 << 4) ^ (1 << 2)
    nxt = None if c2 >= 6 else game.Position(tm, pos.gm & ~1, c2, pos.placed, "goat")
    return dataclasses.replace(mv, jumped=0 if hub else None, nxt=nxt)


def play_pairing(args):
    goat_name, tiger_name, games, seed0 = args[:4]
    opening_plies, legacy = (args[4], args[5]) if len(args) > 4 else (0, False)
    st = _setup()
    game, an = st["game"], st["an"]
    gk, tk = GOATS[goat_name], TIGERS[tiger_name]
    if gk[0] in DETERMINISTIC and tk[0] in DETERMINISTIC and not opening_plies:
        games = 1                                     # two deterministic players always play the same game
    out = dict(goat=goat_name, tiger=tiger_name, games=games, results=Counter(), plies=[], sides={})
    side_stats = {s: dict(decisions=0, by_value=Counter(), blunders=Counter(), had_win_games=0, converted_games=0,
                          first_blunder_plies=[]) for s in ("goat", "tiger")}
    for g in range(games):
        seed = seed0 + g
        rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)
        gm = game.Game(game.start_position("goat"), "goat")
        had_win = {"goat": False, "tiger": False}
        first_blunder = {"goat": None, "tiger": None}
        while True:
            o = gm.outcome(limits=True)
            if o is not None:
                break
            pos = gm.pos
            side = pos.mover
            kind, path = gk if side == "goat" else tk
            moves = game.legal_moves(pos)
            if gm.ply < opening_plies:                # balanced random opening: any move that keeps the draw
                keep = [m for m in moves if m.nxt is not None and an.value(m.nxt)[0] == "draw"]
                gm.push(rng.choice(keep or moves))
                continue
            mv = choose(kind, path, pos, moves, gm.seen(), rng)
            if legacy:                                # outcomes only: the tables describe the fixed rules
                gm.push(legacy_capture(game, pos, mv))
                continue
            before = an.value(pos)[0]
            after = "win" if mv.nxt is None else game.FLIP[an.value(mv.nxt)[0]]
            s = side_stats[side]
            s["decisions"] += 1
            s["by_value"][before] += 1
            had_win[side] |= before == "win"
            drop = DROP.get((RANK[before], RANK[after]))
            if drop:
                s["blunders"][drop] += 1
                if first_blunder[side] is None:
                    first_blunder[side] = gm.ply
            gm.push(mv)
        if o.winner:
            label = f"{o.winner}_win"
        else:
            label = "draw_turn_limit" if "turn limit" in o.reason else "draw_repeat"
        out["results"][label] += 1
        out["plies"].append(gm.ply)
        for side in ("goat", "tiger"):
            s = side_stats[side]
            s["had_win_games"] += had_win[side]
            s["converted_games"] += had_win[side] and o.winner == side
            if first_blunder[side] is not None:
                s["first_blunder_plies"].append(first_blunder[side])
    out["results"] = dict(out["results"])
    out["mean_plies"] = float(np.mean(out["plies"]))
    del out["plies"]
    for side, s in side_stats.items():
        fb = s.pop("first_blunder_plies")
        s["by_value"], s["blunders"] = dict(s["by_value"]), dict(s["blunders"])
        s["games_with_a_blunder"] = len(fb)
        s["mean_first_blunder_ply"] = float(np.mean(fb)) if fb else None
        out["sides"][side] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20_000)
    args = ap.parse_args()
    for name, (kind, path) in {**GOATS, **TIGERS}.items():
        if kind == "model" and not (REPO / path).is_file():
            sys.exit(f"missing model for {name}: {path}")
    jobs = [(g, t, args.games, args.seed) for g in GOATS for t in TIGERS]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(play_pairing, jobs))
    goat_models = [n for n, (k, _) in GOATS.items() if k == "model"]
    tiger_models = [n for n, (k, _) in TIGERS.items() if k == "model"]
    with ProcessPoolExecutor(args.workers) as pool:
        openings = list(pool.map(play_pairing, [(g, t, args.games, args.seed + 50_000, 4, False)
                                                for g in goat_models for t in tiger_models]))
        fix_jobs = [(g, t, 1000, args.seed + 100_000, 0, legacy) for g in goat_models for t in ("Greedy", "Smart")
                    for legacy in (False, True)]
        fix = list(pool.map(play_pairing, fix_jobs))
    for r, job in zip(fix, fix_jobs):
        r["legacy_capture"] = job[5]
    report = dict(games_per_pairing=args.games, seed=args.seed, first="goat", limits="TnGEnv: 200 plies, 4th repeat",
                  models_from_random_openings=openings, capture_fix=fix,
                  goat_players=list(GOATS), tiger_players=list(TIGERS),
                  model_files={n: p for n, (k, p) in {**GOATS, **TIGERS}.items() if k == "model"},
                  seconds=round(time.time() - t0, 1), pairings=results)
    out = Path(__file__).resolve().parent / "tables" / "model_comparison.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"{len(results)} pairings in {report['seconds']}s -> {out}")
    for r in results:
        res = r["results"]
        print(f"{r['goat']:22} vs {r['tiger']:20} n={r['games']:3} G {res.get('goat_win', 0):3} T {res.get('tiger_win', 0):3} "
              f"D {res.get('draw_turn_limit', 0) + res.get('draw_repeat', 0):3} plies {r['mean_plies']:6.1f}")


if __name__ == "__main__":
    main()
