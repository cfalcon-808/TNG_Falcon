# Perfect play: Tigers & Goats solved

Exact game-theoretic values **and exact plies to the end** for **every** position of the game: all
19,017,918,920 of them, both phases, both sides to move. From the tables, a perfect move can be read for
either side after any move history. Rules are those of `env_tng.py` (with the one-capture fix described
under [The solved game and TnGEnv](#the-solved-game-and-tngenv)). The plan and its estimates are in
[`docs/PERFECT_PLAY_DP_PLAN.md`](../docs/PERFECT_PLAY_DP_PLAN.md).

## Result

**The game is a draw.** With perfect play neither side can force a win, whether the goats move first (as in
the goat-learner environment) or the tigers do (as in the tiger-learner environment).

| Positions | Mover wins | Mover loses | Draw |
| --- | ---: | ---: | ---: |
| goats to move, all | 1,699,870,871 | 873,639,885 | 6,935,448,704 |
| tigers to move, all | 1,889,233,606 | 558,324,869 | 7,061,400,985 |
| goats to move, movement phase | 66,849,681 | 224,239,532 | 790,053,157 |
| tigers to move, movement phase | 455,794,551 | 23,345,358 | 602,002,461 |

**Longest forced wins** (plies with best play on both sides: the winner as fast as possible, the loser as slow):
- **Over all positions: 215 plies.** Four positions (two mirror pairs) with the goats to move after 4 or 5
  goats placed and none captured. Example: tigers on 4, 12, 13, goats on 3, 5, 6, 22. The goats win in 215
  plies, and placing on square 2 is their only winning move.
- **In the movement phase: 210 plies.** Tigers to move, 15 goats on the board, no captures; the goats win.
  The tigers' best defence includes a capture.

Per-layer counts, maximum plies and ply histograms are in `tables/metadata.json` (`movement`,
`placement_dtm`).

## How it was solved

On an RTX 5070 Ti the solve took **35 s of GPU time** (`solve.py`) plus **27 s** for the placement ply counts
(`solve_placement_dtm.py`). The CUDA kernels are in `kernels.cu`, run through CuPy.

- **Movement phase** (2.16 billion positions; cycles are possible): retrograde analysis, one capture count at
  a time (5 captured first). Inside each layer, positions are resolved level by level in plies to the end,
  expanding resolved positions to their predecessors with "un-moves" and GPU atomics. Unresolved positions
  are **draws**.
- **Placement phase** (16.9 billion positions; no cycles, since every goat turn places a piece): backward
  induction from 14 goats placed down to 0, once for win/draw/loss (`solve.py`, 2-bit tables) and once with
  exact plies (`solve_placement_dtm.py`, 1 byte per position). The two agreed on every position, after which
  the 2-bit tables were deleted.

Each position's mover, captured count and goats placed come from the observation. The indexing is
`rank3(tigers) * C(20, k) + rank_k(goats among the 20 non-tiger squares)` (`rules.py`).

## Files (`tables/`, 21 GB, not in git)

| File | Contents |
| --- | --- |
| `mv_k{k}_{G\|T}.u16` | movement phase with k = 15 − captured goats on board, goats (`G`) or tigers (`T`) to move: `uint16` per position. 0 = draw, 2d = mover wins in d plies, 2d+1 = mover loses in d plies |
| `pl_n{n}_{G\|T}.u8` | placement phase with n goats placed: 1 byte per position. 0 = draw, otherwise d + 1 with d the plies to the end: d odd = mover wins, d even = mover loses (the winner always makes the last move). Blocks by captured count at `metadata.json` → `placement.{n}.block_offsets` |
| `pl_n{n}_{G\|T}.u2` | **deleted** (4.2 GB). The same positions at 2 bits each (0 draw, 1 mover wins, 2 mover loses), written by `solve.py`. The `.u8` files hold the same values; the oracle and `check_all.py` fall back to the 2-bit files only if the `.u8` files are missing |
| `metadata.json` | rules hash, encodings, counts per layer, maximum plies and ply histograms, opening value |
| `rules_proof.json`, `check_all.json`, `verification.json` | results of `prove_rules.py`, `check_all.py` and `verify_tables.py` |
| `solve_log.txt` | the solver's log |

## Using it

```python
from perfect_play.oracle import PerfectPlay, Position, RepetitionTracker
pp = PerfectPlay()
pos = Position.from_obs(obs, mover="goat")        # any TnGEnv observation
pp.evaluate(pos)       # ("win" | "loss" | "draw", plies or None) for the mover
pp.evaluate_with_limit(pos, plies_played)         # the same in TnGEnv's game, which stops after 200 plies
pp.move_values(pos)    # every legal move with its exact result, best first
pp.best_action(pos)    # a perfect move (TnGEnv action id)
pp.is_draw(pos)
env = TnGEnv(tiger_ai="model", tiger_model_predict_fn=pp.predict_fn("tiger"))   # a perfect tiger opponent
```

**How the player picks a move** (in both phases):
- From a winning position, it plays the fastest win, so the game always progresses.
- From a lost position, it delays the loss as long as possible.
- From a drawn position, it keeps the draw. Passing `seen` (the positions so far) makes it prefer moves to
  positions not yet visited.

**TnGEnv's turn limit and repeat rule.** The solved game has neither; the environment stops after
`max_turns` = 100 turns (200 plies) and, in the goat-learner path, at the 4th repeat of a board. With ply
counts everywhere, both are handled exactly:
- A forced result in d plies still happens in the environment iff `plies_played + d <= 200`. The winner's
  fastest line takes at most d plies against any defence, and the loser can always last d. Otherwise the
  game is cut off, a draw in effect. `evaluate_with_limit` applies this.
- The same moves are optimal with or without the limit, so `best_action` needs no change.
- The repeat rule never interrupts the perfect player's win. On its winning line the plies to go fall every
  ply, so no position repeats, and no earlier position had the same winning value.
- Placement positions are always reached after a fixed number of plies: 2n (goats to move) or 2n − 1
  (tigers to move) when the goats move first, 2n + 1 or 2n when the tigers do. Of the 4,250,840,109 placement
  positions with a forced result, **1,316,096 (0.03%)** need more plies than the limit leaves (counts per
  layer in `metadata.json` → `placement_dtm.layers.{n}.{G|T}.beyond_ply_limit`). In the movement phase,
  where the ply count depends on the history, check `plies_played + d` per position.

**Draws and repetition.** `pp.is_draw(pos)` flags theoretical draws. `RepetitionTracker` follows a game ply by
ply and reports:
- repeated positions (`repeated_plies`) and the first cycle with its period;
- the longest run of plies that only revisits earlier positions (`longest_repeat_run`);
- the longest back-and-forth shuffle, where each move undoes that side's previous move (`longest_shuffle`).

`verify_tables.py` also labels every game that ends without a win as a *theoretical draw* or a *missed win*.

## Verification

The tables are proven exact for the rules below by two facts:
- **Every stored entry satisfies the minimax recursion over its successors' stored entries.** A win in d
  needs a successor where the opponent loses in d − 1 and none faster. A loss in d needs every successor to
  be an opponent win, the slowest in d − 1. A draw needs no opponent-loss successor and at least one draw
  successor.
- **Those local checks determine the solution.** Ply counts strictly fall along every winning line, so no
  win or loss can be unfounded, and a draw-labelled position lets both sides keep the draw forever. So the
  entries are the true values and the true plies.

What establishes each part:

| Check | Scope | Result |
| --- | --- | --- |
| **Independent full check**, `check_all.py`. Written from `env_tng.py` and the file spec by a separate agent that never saw the solver code. Checks values and plies, and each `.u2` entry against its `.u8` entry while the `.u2` files existed | **every position**, 19,017,918,920 | **0 mismatches** (164 s, 16 threads; rerun after deleting the `.u2` files: 0 mismatches, 140 s) |
| **Rule equivalence**, `prove_rules.py` parts 1–2. Every piece's legal moves and capture effects in `TnGEnv` vs `rules.py`, on every local pattern, with every board square read recorded. Every read stays within distance 2 of the piece, so the patterns cover every board | 433,127 local patterns, all 23 squares | **0 mismatches** (after the fix below) |
| Constants, move graph, and every line of `env_tng.py` that writes the board, captures, goats placed or phase (found by parsing the file), part 3 | whole file | consistent |
| Indexing bijection, part 4 | every index, 1,846,054,980 | 0 failures |
| `check_all.py`'s indexing and move generator vs `rules.py`'s, part 5 | **every board**, 1,846,054,980 | 0 mismatches |
| Minimax recursion on the GPU, during the solve | every movement position | 0 failures |
| 2-bit vs ply-count placement tables | every placement position | 0 disagreements |
| Pure-Python recomputation of values and plies, `verify_tables.py` | 126,000 random positions, every layer | 0 mismatches |
| Perfect vs perfect from both openings | | draw both times: threefold repetition with a 4-ply cycle, or 400 plies with no third repeat when avoiding repeats |
| Perfect goat vs greedy tiger, 200 games | | 78 wins, 122 draws (52 turn limit, 70 repeat rule), **0 losses** |
| Perfect goat vs smart tiger, 200 games | | 200 draws (repeat rule), **0 losses**: the smart tiger never makes a losing mistake |
| Perfect tiger vs random goats, 200 games | | 193 wins, 7 draws, **0 losses** |
| Perfect tiger vs PPO + nll 0.5 (seed 42), ReLU Maskable PPO, shipped RobustGoat | 200 games each | all draws at the turn limit; the goats shuffle (148–161 of 199 plies repeat a position) |

The tables give *minimax*-perfect play. The player never loses and wins whenever the opponent errs into a
lost position, but in a drawn position it does not try to provoke mistakes. That is why the trained PPO goat,
which exploits the smart tiger, beats it 94.7% of the time while the perfect goat only draws. The best
response to a scripted tiger is method B of the plan.

## The solved game and TnGEnv

- **Moves:** exactly `TnGEnv._tiger_moves` and `TnGEnv._update_valid_moves` (proved above).
- **Capture effect:** the jumped goat is removed and the capture counted; the 6th capture wins at once.
- **No legal move:** the side to move loses. TnGEnv's tiger-learner path does this for both sides. Its
  goat-learner path detects only the tigers having no move; when the goats have none, the learner's mask is
  empty.
- **Not in the solved game:** the turn limit and the repeat rule, handled as described under *Using it*.

**The fix found by the proof.** In the committed `env_tng.py`, `TnGEnv._find_jumped_goat(4, 2)` returned
square 0. That broke the capture of a tiger on 4 over the goat on 3, landing on 2: the path 4 → 0 → 2 also
matches the "same direction twice" rule, and it was checked first. Both step functions then removed the goat
on the hub if there was one, and otherwise removed nothing and left the capture count unchanged; the goat on
3 stayed in every case. `_tiger_moves` already excluded that direction, so the move list was right and only
its effect was wrong. No other capture (56 checked) is affected.

The fix adds the same hub restriction to `_find_jumped_goat` (4 lines), so the environment now matches the
tables. `prove_rules.py` found 1,248 differing local patterns before the fix and none after. The independent
checker flagged the same capture from `env_tng.py` alone. Models trained or evaluated before the fix saw the
old behaviour.

The notebooks `TnG_Notebook.ipynb`, `TnG_MPPO_Notebook.ipynb` and `TnG_PPO_Notebook.ipynb` each contain their
own copy of the environment. Those copies still have the old `_find_jumped_goat`. `TnG_Notebook.ipynb` uses
the fixed file when `ENV_SOURCE = "repo"`.

## Comparison with the published Bagh-Chal solution

Lim Yew Jin and J. Nievergelt studied Bagh-Chal, the larger Nepali version: *Computing Tigers and Goats*
(ICGA Journal 27(3), 2004) and *Tigers and Goats is a draw* (Games of No Chance 3, MSRI Publications 56, 2009).

| | Bagh-Chal (Lim & Nievergelt) | This game (`env_tng.py`) |
| --- | --- | --- |
| Board | 5×5 grid, 25 points, with diagonals; 8 symmetries | 23 points (hub plus rows of 6, 6, 6 and 4), no diagonals, at most 4 neighbours; 1 mirror symmetry |
| Pieces | 4 tigers on the corners, 20 goats | 3 tigers on 0, 3, 4; 15 goats |
| Tigers win | 5 captures | 6 captures |
| Goats win / lose | tigers immobilized / goats immobilized | same |
| First move | goats | goats (goat-learner environment) or tigers (tiger-learner environment); both solved |
| Placement phase | 39 plies, ~33 billion positions | 29–30 plies, 16.9 billion positions |
| Sliding phase | 88,260,972 positions after symmetry | 2,162,284,740 positions (no symmetry reduction) |
| Sliding-phase method | retrograde analysis, win/draw/loss | retrograde analysis, win/draw/loss **plus exact plies to the result** (longest 210) |
| Placement-phase method | forward search (tree estimated at ~10⁴¹): a heuristic player against an exhaustive opponent, and a halfway database at ply 19 (216.5 million of ~1.6 billion positions computed as needed) | **every position tabulated** by backward induction, with exact plies (longest 215) |
| What is solved | the opening position (a *weak* solution): "Goat can at least draw" and "Tiger can at least draw" | **every position** (a *strong* solution), so a perfect move exists after any history |
| Compute | about 3 years on and off: an 8-PC cluster, a 5-month run on a PowerMac G5, 2 months of verification; re-proofs took 7 h and 48 h | **62 s** on one RTX 5070 Ti; full independent verification in 164 s on 16 CPU threads |
| Result | **draw** | **draw** |

Their sliding-phase statistics (their Table 2) and ours show the same shape: the herd's survival hinges on
the capture count. In Bagh-Chal, Goat-to-move losses jump from 20.4% at 3 captures to 72.2% at 4, one short
of the tigers' 5. Here the same jump comes one capture later, from 8.9% at 4 to 60.4% at 5, one short of 6.
With no captures, the goats win 69.1% of goat-to-move sliding positions here, against 91.4% in Bagh-Chal.
In the sliding phase, the 3 tigers have no forced win in any position with fewer than two captures.

## Rebuild

```bash
python perfect_play/prove_rules.py               # rules and indexing vs TnGEnv, every board (about 11 minutes)
python perfect_play/solve.py                     # needs CuPy + CUDA GPU; movement + 2-bit placement tables (8.5 GB)
python perfect_play/solve_placement_dtm.py --delete-u2   # placement ply counts (17 GB), checked against the
                                                         # 2-bit tables, which are then deleted
python perfect_play/check_all.py                 # independent check of every position (about 3 minutes)
python perfect_play/verify_tables.py --samples 3000 --games 200
```
