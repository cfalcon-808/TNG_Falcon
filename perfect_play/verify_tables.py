"""Verify the solved tables independently of the GPU, then play with them.

usage: python perfect_play/verify_tables.py [--samples 3000] [--games 200] [--models a.zip,b.zip]
1. Sampled minimax check: random positions of every layer are decoded with the pure-Python indexing of
   rules.py, their successors generated with the Python move generator (checked against TnGEnv by
   prove_rules.py), and the stored value and plies to the end recomputed (placement plies need the .u8
   tables from solve_placement_dtm.py).
2. Perfect vs perfect from both openings, with draw and repetition detection.
3. Perfect goat vs the greedy and smart tigers, and the perfect tiger vs random goats, inside TnGEnv.
4. Perfect tiger vs saved goat models (optional, needs stable-baselines3).
Every game that does not end in a win is classified: theoretical draw, or a win someone missed.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oracle import PerfectPlay, Position, RepetitionTracker  # noqa: E402
from rules import GOATS, block_size, env_tng, placement_offsets, popcount, unrank_board  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=3000)
ap.add_argument("--games", type=int, default=200)
ap.add_argument("--models", default="")
args = ap.parse_args()
pp = PerfectPlay()
rng = random.Random(7)
report = {}

# ── 1. sampled minimax check ────────────────────────────────────────────────────────────────────────
def recompute(pos):
    rows = []
    for a, nxt in pp.successors(pos):
        if nxt is None:
            rows.append(("win", 1))
            continue
        res, d = pp.evaluate(nxt)
        rows.append(({"win": "loss", "loss": "win", "draw": "draw"}[res], None if d is None else d + 1))
    if not rows:
        return "loss", 0
    wins = [d for r, d in rows if r == "win"]
    if wins:
        return "win", (min(wins) if None not in wins else None)
    if any(r == "draw" for r, _ in rows):
        return "draw", None
    losses = [d for r, d in rows]
    return "loss", (max(losses) if None not in losses else None)


checked = bad = 0
for k in range(10, 16):
    for mover in ("goat", "tiger"):
        for _ in range(args.samples):
            tm, gm = unrank_board(rng.randrange(block_size(k)), k)
            pos = Position(tm, gm, GOATS - k, GOATS, mover)
            stored, expect = pp.evaluate(pos), recompute(pos)
            checked += 1
            bad += stored != expect
for n in range(GOATS):
    offs = placement_offsets(n)
    for mover in ("goat", "tiger"):
        for _ in range(args.samples):
            i = rng.randrange(offs[-1])
            c = max(j for j in range(len(offs) - 1) if offs[j] <= i)
            tm, gm = unrank_board(i - offs[c], n - c)
            pos = Position(tm, gm, c, n, mover)
            stored, expect = pp.evaluate(pos), recompute(pos)
            if pp.pl8 is None:                        # 2-bit tables only: compare the value alone
                stored, expect = stored[0], expect[0]
            checked += 1
            bad += stored != expect
report["sampled_minimax_check"] = dict(positions=checked, mismatches=bad)
print(f"1. sampled minimax check: {checked:,} positions, {bad} mismatches", flush=True)

# ── 2. perfect vs perfect, with repetition detection ────────────────────────────────────────────────
START_T = sum(1 << p for p in env_tng.TIGER_START_POSITIONS)


def self_play(first, avoid_repeats, max_plies=400):
    pos, tracker, seen = Position(START_T, 0, 0, 0, first), RepetitionTracker(), set()
    result = None
    while tracker.ply < max_plies:
        seen.add(pos.key())
        a = pp.best_action(pos, seen if avoid_repeats else None)
        nxt = dict(pp.successors(pos))[a]
        tracker.update(pos, a, nxt)
        if nxt is None:
            result = f"{pos.mover} wins"
            break
        if not pp.successors(nxt):
            result = f"{pos.mover} wins (opponent has no move)"
            break
        if tracker.max_occurrences >= 3:
            result = "draw by threefold repetition"
            break
        pos = nxt
    return result or f"draw (no result in {max_plies} plies)", tracker.summary(), pp.evaluate(Position(START_T, 0, 0, 0, first))[0]


report["self_play"] = {}
for first in ("goat", "tiger"):
    for avoid in (False, True):
        result, reps, value = self_play(first, avoid)
        report["self_play"][f"{first}_first_avoid_repeats_{avoid}"] = dict(table_value=value, result=result, **reps)
        print(f"2. perfect vs perfect, {first} first, avoid repeats={avoid}: table says {value}; {result}; {reps}", flush=True)

# ── 3. perfect player inside TnGEnv ─────────────────────────────────────────────────────────────────
def classify_end(info, last_pos):
    winner = info.get("winner")
    if winner in ("Goat", "Tiger"):
        return f"{winner} win"
    value = pp.evaluate(last_pos)[0]
    return f"{winner or info.get('reason')}: " + ("theoretical draw" if value == "draw" else f"missed {last_pos.mover} {value}")


def env_games(role, opponent, games, goat_model=None):
    """The perfect player is the learner `role`; returns outcome counts and repetition statistics."""
    kwargs = dict(learner_role=role)
    if role == "goat":
        kwargs["tiger_ai"] = opponent
    else:
        kwargs["goat_opponent_ai"] = "model" if goat_model else "random"
        kwargs["goat_model_predict_fn"] = goat_model
    env = env_tng.TnGEnv(**kwargs)
    outcomes, reps = Counter(), []
    for g in range(games):
        random.seed(g); np.random.seed(g)
        obs, _ = env.reset(seed=g)
        tracker, seen, done, pos = RepetitionTracker(), set(), False, None
        while not done:
            pos = Position.from_obs(obs, role)
            seen.add(pos.key())
            a = pp.best_action(pos, seen)
            nxt = dict(pp.successors(pos)).get(a)
            tracker.update(pos, a, nxt)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
            if not done and nxt is not None:
                tracker.update(nxt, None, Position.from_obs(obs, role))   # the opponent's reply
        outcomes[classify_end(info, pos)] += 1
        reps.append(tracker.summary())
    env.close()
    rep = {key: float(np.mean([r[key] for r in reps])) for key in ("plies", "repeated_plies", "longest_repeat_run", "longest_shuffle")}
    rep["games_with_a_repeat"] = sum(r["repeated_plies"] > 0 for r in reps)
    return dict(outcomes, **{"mean_" + k if not k.startswith("games") else k: v for k, v in rep.items()})


report["env_games"] = {}
for role, opp in (("goat", "greedy"), ("goat", "smart"), ("tiger", "random")):
    res = env_games(role, opp, args.games)
    report["env_games"][f"perfect_{role}_vs_{opp}"] = res
    print(f"3. perfect {role} vs {opp} ({args.games} games): {res}", flush=True)

# ── 4. perfect tiger against saved goat models ──────────────────────────────────────────────────────
if args.models:
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import PPO
    for path in args.models.split(","):
        data = json.loads(__import__("zipfile").ZipFile(path).read("data"))
        cls = MaskablePPO if "maskable" in data["policy_class"].get("__module__", "") else PPO
        model = cls.load(path, device="cpu")

        def goat_fn(obs, mask, model=model):
            import torch
            obs_t, _ = model.policy.obs_to_tensor(np.asarray(obs))
            with torch.no_grad():
                logp = model.policy.get_distribution(obs_t).distribution.logits[0].numpy()
            return int(np.argmax(np.where(mask, logp, -np.inf)))

        res = env_games("tiger", "model", args.games, goat_model=goat_fn)
        report["env_games"][f"perfect_tiger_vs_{Path(path).parent.parent.parent.name or Path(path).stem}"] = res
        print(f"4. perfect tiger vs {path} ({args.games} games): {res}", flush=True)

out = Path(__file__).resolve().parent / "tables" / "verification.json"
out.write_text(json.dumps(report, indent=2, default=str))
print("wrote", out)
sys.exit(1 if bad else 0)
