# Plan: solving Tigers & Goats exactly with dynamic programming

**Status:** implemented in [`perfect_play/`](../perfect_play/README.md) (results, verification and the one rule fix
found on the way are there) · **Date:** 2026-11-05 · **Rules:** `env_tng.py` v1.1 (commit `11ee06e`) · **Notation:** as in
[`TRAINING_MATHEMATICS.md`](TRAINING_MATHEMATICS.md)

## 1. Answer

**Yes, it is feasible on this machine.** The game has about **20.1 billion positions**, or about 10.05
billion after the board's mirror symmetry. That is small enough for an exact retrograde solution on one
16-core desktop.
- **Compute:** measured throughput predicts about **1–3 hours** for the core solve (§7.1).
- **Storage:** about 4.3 GB for the movement-phase tables and up to 37 GB for a complete placement table
  (half that with symmetry). The machine has 60 GB of RAM and 130 GB free on disk.
- **Engineering:** about **7–10 working days** for the solver, its verification and the integration with
  the notebook, plus 3–4 days for the optional best-response tables of §5.

The result is a table giving every position's exact outcome under perfect play (goat win, tiger win, or
draw) and the number of plies to that outcome. A perfect strategy for **both** sides follows directly: from
any position, play a move that keeps the best value.

Precedent: the larger Nepali version, Bagh-Chal (25 points, 4 tigers, 20 goats), was studied by Lim Yew Jin
and J. Nievergelt.
- *Computing Tigers and Goats* (ICGA Journal 27(3), 2004) solved its sliding phase by retrograde analysis.
- *Tigers and Goats is a draw* (Games of No Chance 3, 2009) proved the game a draw. It used forward search
  through the placement phase, because that phase's ~33 billion positions were then too many to tabulate.

The comparison with this solve is in [`perfect_play/README.md`](../perfect_play/README.md).

## 2. What "perfect" means here

| Target | Question it answers | Method | §  |
| --- | --- | --- | --- |
| **A. Game-theoretic solution** | Best play for both sides against a perfect opponent | minimax retrograde analysis | 4 |
| **B. Best response to a scripted tiger** | Highest goat win probability against the greedy or smart tiger, within the 100-turn limit | finite-horizon expectimax (an MDP) | 5 |

A is "perfect play" in the usual sense and covers both roles. B maximizes the win rate the environment
measures. Because the scripted tigers are random, their best response can be different from the minimax
strategy, and can win more often.

### 2.1 Canonical rules

The solver uses one rule set: the environment's rules, with two asymmetries between its learner paths
removed (from `TRAINING_MATHEMATICS.md` §2):

| Rule | Solver | Environment |
| --- | --- | --- |
| Board and moves | `move_map`, hub captures `HUB0_CAPTURE_JUMP`, no capture "up" from 2–5 | same |
| Goat win | tigers to move with no legal move | same (both paths) |
| Tiger win | 6 captures, or goats to move with no legal move | the no-goat-move rule exists only in the tiger-learner path; the goat-learner path never reaches it |
| Turn limit | not part of the position; handled at play time from the stored ply counts (§4.6) | `MAX_TURNS = 100` learner turns → `MaxTimeout` |
| Repetition | not part of the position; cycles are draws | goat-learner path only: `MAX_REPEATS` repeats → `RepeatTimeout` |

`main` has a newer commit, "terminate on illegal moves". Before the solver is built, the rules must be
re-checked against the `env_tng.py` it will be compared with.

## 3. State space

A position is $x = (S_T, S_G, c, n, \text{mover})$:
- $S_T$ is the set of 3 tiger squares, and $S_G$ the set of goat squares, disjoint from $S_T$.
- $c \in \{0, \dots, 5\}$ is the number of goats captured ($c = 6$ is a terminal tiger win).
- $n$ is the number of goats placed, $0 \leq n \leq 15$, with $|S_G| = n - c$.
- mover $\in \{\mathrm{G}, \mathrm{T}\}$ is the side to move. The phase is placement while $n < 15$.

This is the observation $s = (b, c, \phi)$ plus the side to move; $n$ follows from $|S_G| + c$.

| Part | Positions | Notes |
| --- | ---: | --- |
| movement, $c = 5$ (10 goats) | 654,405,752 | $1771 \cdot \binom{20}{10} \cdot 2$ |
| movement, $c = 4$ (11 goats) | 594,914,320 | |
| movement, $c = 3$ (12 goats) | 446,185,740 | |
| movement, $c = 2$ (13 goats) | 274,575,840 | |
| movement, $c = 1$ (14 goats) | 137,287,920 | |
| movement, $c = 0$ (15 goats) | 54,915,168 | |
| **movement total** | **2,162,284,740** | the cyclic part (§4.2) |
| placement, goats to move ($n = 0..14$) | 8,427,817,090 | acyclic (§4.3) |
| placement, tigers to move ($n = 1..15$) | 9,508,957,689 | acyclic |
| **total** | **20,099,059,519** | about 10.05 billion after mirror symmetry |

$\binom{23}{3} = 1771$ tiger placements. These counts are upper bounds: they include positions that
cannot be reached from the start, which is harmless for correctness.

**Symmetry.** The left-right mirror (columns a↔f, b↔e, c↔d; hub directions 1↔4, 2↔3; side directions 2↔4) maps
the move graph onto itself. This was checked against `move_map`. Each position and its mirror image have
the same value, so storing one of each pair roughly halves time and space.

## 4. Method A: minimax retrograde analysis

### 4.1 Values

For each position $x$ the solver computes a value $v(x)$ and a ply count $d(x)$:
- $v(x) \in \{\mathrm{G}, \mathrm{T}, \mathrm{D}\}$: forced goat win, forced tiger win, or draw (neither side can force a win).
- For a forced win, $d(x)$ is the number of plies to the win when the winner hurries and the loser delays
  (distance to mate, DTM). For a draw, $d(x) = \infty$.

They satisfy the minimax recursion over the successor set $\Gamma(x)$:

$$
v(x) = \begin{cases}
\text{loss for the mover} & \Gamma(x) = \varnothing \\
\text{win for the mover} & \exists\, y \in \Gamma(x):\ v(y) = \text{win for the mover} \\
\text{loss for the mover} & \forall\, y \in \Gamma(x):\ v(y) = \text{loss for the mover} \\
\mathrm{D} & \text{otherwise}
\end{cases}
\qquad
d(x) = 1 + \begin{cases} \min_{y \,\text{winning}} d(y) & \text{mover wins} \\ \max_{y} d(y) & \text{mover loses} \end{cases}
$$

A capture that reaches $c = 6$ is an immediate tiger win, with $d = 1$.

### 4.2 Movement phase: layer by layer, with retrograde BFS inside each layer

Captures only increase $c$, so the movement phase splits into layers $c = 5, 4, \dots, 0$. Each layer
depends only on itself and on the layer with one more capture. Solve them in that order. Within a layer the
move graph has cycles, so use the standard retrograde algorithm:

1. **Initialize.** For every position, count its successors inside the layer, $u(x)$. Resolve any position
   that has a capture into an already-solved layer, or no legal move.
2. **Propagate.** Process resolved positions in increasing $d$. For each in-layer predecessor $x'$ (found by
   *unmoving* a goat or a tiger):
   - if $x'$'s mover wins by moving here, resolve $x'$ as a win with $d + 1$;
   - otherwise decrement $u(x')$, and when it reaches 0, resolve $x'$ as a loss with $d + 1$.
3. **Finish.** Whatever is unresolved when the queue is empty is a draw.

Each move edge is visited a constant number of times. The work is therefore about the number of edges:
$2.16\times10^{9}$ positions times about 19.5 moves, so about $4.2\times10^{10}$ edge visits.

### 4.3 Placement phase: backward induction over goats placed

During placement every goat turn raises $n$ by one, so no position can repeat and the graph is acyclic.
Solve layer by layer for $n = 15, 14, \dots, 0$:
- tigers to move with $n$ placed take their values from goats-to-move positions with the same $n$;
- goats to move with $n$ placed take their values from tigers-to-move positions with $n + 1$;
- tigers to move with $n = 15$ take their values from the movement-phase tables.

Only two layers need to be in memory at a time; the largest, around $n = 14$, has about $1.35\times10^{9}$
positions per side. The value of the opening position comes out at $n = 0$.

### 4.4 Indexing

Each position maps to a dense index within its layer:

$$
\iota(x) = r_3(S_T)\cdot\binom{20}{k} + r_k\big(\pi_{S_T}(S_G)\big), \qquad k = |S_G|
$$

$r_k$ is the combinatorial-number-system rank of a $k$-subset, and $\pi_{S_T}$ renumbers the 20 squares not
held by tigers as $0, \dots, 19$. Ranking costs about 23 table lookups and needs no hash table.
Canonicalize under the mirror before ranking: keep the smaller index of a position and its mirror.

### 4.5 Storage

Store 2 bytes per position: 2 bits of value and 14 bits of $d$ (up to 16,383 plies). If the maximum $d$
turns out to exceed that, which is not expected, use a 1-byte value table plus a separate $d$ table.

| Table | Size | With symmetry |
| --- | ---: | ---: |
| movement, all layers | 4.3 GB | 2.2 GB |
| placement, all layers (needed to *play* perfectly from the start) | 35.9 GB | ~18 GB |
| placement, two layers in memory while solving | ~11 GB | ~5.5 GB |

The tables are kept as memory-mapped files under `artifacts/perfect_play/`, not in git. They compress well,
because most positions are decided wins or losses with small $d$.

### 4.6 Playing from the tables, including the environment's limits

- **Winning position:** play a winning move with the smallest $d$. This wins within $d(x)$ plies, so it also
  wins inside the turn limit whenever $d(x)$ is at most the plies left.
- **Losing position:** play the move with the largest $d$, delaying the loss as long as possible.
- **Drawn position:** play any move that keeps the draw. Under the goat-learner rules, the goats should
  prefer moves that avoid the repeat-stall limit (§2.1). That is a tie-break, not part of the solved value.

The environment's `MaxTimeout` and `RepeatTimeout` only end drawn games, or wins longer than the remaining
budget. The table identifies both cases exactly.

## 5. Method B (optional): best response to the scripted tigers

The greedy tiger and the smart tiger are random policies $\mu(\cdot \mid x)$. Against such a fixed tiger,
the goat's problem is a Markov decision process over goat-to-move positions. Including the turn limit,
it can be solved exactly by finite-horizon backward induction over the remaining learner turns $h$:

$$
W_h(x) = \max_{a \in \mathcal{L}(x)} \sum_{y} \mu(y \mid x, a)\, \Big[\mathbb{1}[\text{goat wins at } y] + \mathbb{1}[y \text{ not terminal}]\, W_{h-1}(y)\Big], \qquad W_0 \equiv 0
$$

$W_h(x)$ is the highest probability of a goat win within $h$ more turns. The horizon grows from 1 to 100
learner turns. $W_{100}$ at the opening position is the best win rate any goat can achieve against that
tiger in this environment. That is a ceiling for every model in the legality study.
- **Size:** goat-to-move positions only: $1.08\times10^{9}$ in the movement phase plus $8.4\times10^{9}$ in
  placement. Two `float32` arrays of the movement part take 8.6 GB.
- **Cost:** each horizon step is two sweeps. First, tigers-to-move positions average over the tiger's replies
  (about 6 each). Then goats-to-move positions take the maximum over goat moves (about 13 each). That is
  about $2\times10^{10}$ table reads, or about 4 minutes per step at the measured rate. The 85 movement-phase
  steps then take about 6 hours, plus computing the smart tiger's move distribution (cached once per position).
- **Prerequisite:** the smart tiger's move distribution has to be re-implemented exactly. That means its
  capture ranking, breadth-first anchor search and random roaming (`TRAINING_MATHEMATICS.md` §2.4), so
  that $\mu$ can be computed for any position.

## 6. Verification

1. **Differential rules test.** On $10^{7}$ random and $10^{6}$ game-reached positions, the solver's
   legal-move sets, captures, terminal flags and successors must equal `TnGEnv`'s exactly.
2. **Recursion check.** After solving, re-verify the recursion of §4.1 for every position in parallel. This
   checks value consistency, and that $d$ satisfies min over winning successors and max over losing successors.
3. **Small-board cross-check.** Run the same solver on a reduced game (for example 4 goats on a sub-board)
   and compare against a brute-force minimax search.
4. **Play test.** The table's goat and tiger play each other: they must reproduce the opening value. They
   must never lose a position the table rates as a win, against greedy, smart, random and every trained model.
5. **Determinism.** Two solves with different thread counts must produce byte-identical tables (checksums).

## 7. Estimates

### 7.1 Compute, from measured throughput

`tools/dp_bench.py` uses the exact move rules and the indexing of §4.4. On 16 threads it measured
**4.2 million positions per second, or 82 million successor reads per second**. Each position generated all
of its ~19.5 successors, indexed them, and read their values from a full-size 446 MB table, so every read is
a realistic cache miss. The machine was partly loaded during the run, so these figures are conservative.
Single-threaded the rate was 0.4M/s, about 10× slower than with 16 threads.

| Step | Work | Estimated time (16 threads) |
| --- | --- | ---: |
| Movement phase: initialization pass | $2.16\times10^{9}$ positions | ≈ 9 min |
| Movement phase: retrograde propagation | ≈ $4.2\times10^{10}$ edge visits (unmoves and counter updates) | ≈ 10–30 min |
| Placement phase: backward induction | $1.79\times10^{10}$ positions, one pass | ≈ 70 min |
| Recursion check (§6.2) | one more pass over everything | ≈ 80 min |
| **Method A total** | | **≈ 3 h** (about 1.5 h with mirror symmetry) |
| Method B, per tiger | 85 sweeps over $1.08\times10^{9}$ positions, plus placement | ≈ 7–10 h |

Each estimate is about ±2×. The retrograde inner loop does a little more work per edge than the benchmark
(writes plus counter updates), and the ranking code has not been optimized; both could swing the time.
**Pure Python would be about 1,000× slower (months), so the solver must be compiled.** That means Numba
(as in the benchmark) or C++/Rust.

### 7.2 Engineering schedule

| # | Task | Effort |
| --- | --- | ---: |
| 1 | Canonical rule spec (§2.1), re-checked against the current `main` | 0.5 day |
| 2 | Compiled engine: bitboards, move and unmove generation, captures, terminal tests | 1.5 days |
| 3 | Differential tests against `TnGEnv` (§6.1) | 1 day |
| 4 | Indexing, mirror canonicalization, memory-mapped storage | 1 day |
| 5 | Movement-phase retrograde solver (§4.2) and recursion check | 1.5 days |
| 6 | Placement-phase backward induction (§4.3) | 1 day |
| 7 | Small-board brute-force cross-check and play tests (§6.3–6.4) | 1 day |
| 8 | Integration: `PerfectGoat` and `PerfectTiger` opponents, oracle evaluation, notebook option | 1–1.5 days |
| | **Method A total** | **≈ 8–9 days** (7–10 with contingency) |
| 9 | Optional: exact greedy and smart tiger distributions, plus the finite-horizon MDP (§5) | 3–4 days, plus about a day of compute |

### 7.3 Risks

| Risk | Mitigation |
| --- | --- |
| The rules drift (e.g. the new "terminate on illegal moves" commit) | Freeze a rules version; the differential test (§6.1) gates every solve |
| $d$ exceeds 14 bits | Detect in the first layer and switch to a separate `uint16` table for $d$ |
| Placement tables are too large to keep | Keep movement tables (2–4 GB) plus placement values for reachable positions only, found by a forward sweep from the opening |
| Memory-bandwidth limits | Process positions in index order per tiger rank; use the symmetry; keep one layer in RAM at a time |

## 8. What the tables would give the project

- **A perfect opponent** for training and evaluation, the strongest possible "tiger" or "goat", replacing
  greedy and smart as the hardest test.
- **Exact grading of any model:** the fraction of its moves that keep the best value, and how many plies
  from optimal it is. This is a much finer measure than win rate.
- **Imitation data:** supervised targets $\arg\max$ over value-optimal moves for distilling perfect play into
  the network. It also gives an upper bound on what PPO with any loss can achieve.
- **The game's value:** whether the opening is a goat win, a tiger win or a draw under perfect play. With
  method B, it also gives the best achievable win rate against each scripted tiger in this environment.
