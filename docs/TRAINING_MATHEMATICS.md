# Training mathematics: environment, algorithms, models and tests

**Version 1.2 · 2026-10-28** · Code: `env_tng.py` (v1.1, commit `11ee06e`), `TnG_Notebook.ipynb` (control center, v1.2),
`TnG_MPPO_Notebook.ipynb`, `TnG_PPO_Notebook.ipynb`, `train.py`.

This document specifies, in one notation:
- the game as a decision process
- every learning objective used in the project
- how each saved model was trained
- every test and experiment run on them

Sections 1–2 describe the environment. Sections 3–5 cover the networks and losses, Section 6 the models,
and Sections 7–8 the tests and the legality study.

## Contents

1. [Notation](#1-notation)
2. [Environment and rewards](#2-environment-and-rewards)
3. [Policy and value network](#3-policy-and-value-network)
4. [Data collection, returns and advantages](#4-data-collection-returns-and-advantages)
5. [Objectives](#5-objectives)
6. [How each model was trained](#6-how-each-model-was-trained)
7. [Evaluation protocol and metrics](#7-evaluation-protocol-and-metrics)
8. [Verification tests](#8-verification-tests)
9. [The legality study](#9-the-legality-study)

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| $t$ | learner decision index within an episode; $T$ its final index |
| $s_t = (b_t, c_t, \phi_t) \in \mathbb{Z}^{25}$ | observation: board $b_t \in \{0,1,2\}^{23}$ (empty, goat, tiger), goats captured $c_t \in \{0,\dots,6\}$, phase $\phi_t \in \{0,1\}$ (placement, movement) |
| $a_t \in \mathcal{A} = \{0,\dots,114\}$ | action $a = 5p + d$: position $p \in \{0,\dots,22\}$, direction $d \in \{0,\dots,4\}$ ($d=0$ places a goat) |
| $\mathcal{L}(s) \subseteq \mathcal{A}$, $m(s) \in \{0,1\}^{115}$ | legal actions in $s$ and the mask $m_a(s) = \mathbb{1}[a \in \mathcal{L}(s)]$ |
| $\omega \in \{\text{goat}, \text{tiger}\}$, $\mu$ | learner role, and the opponent policy that replies inside each environment step |
| $r_t$, $\text{term}_t$, $\text{trunc}_t$ | reward, terminal flag, truncation flag after $a_t$ (and the opponent's reply) |
| $\mathsf{NAME}$ | an environment knob from `DEFAULT_KNOBS`, e.g. $\mathsf{REWARD\_STEP}$ |
| $\theta$, $\psi$ | policy parameters and value parameters (one network with shared input, separate trunks; §3) |
| $z_\theta(s) \in \mathbb{R}^{115}$ | policy logits; $\pi_\theta(a \mid s) = \operatorname{softmax}(z_\theta(s))_a$ is the **raw** policy |
| $\pi^{m}_\theta(a \mid s)$ | the **masked** policy (§3.2) |
| $V_\psi(s)$ | value estimate |
| $\gamma$, $\lambda$ | discount and GAE parameter |
| $\hat A_t$, $\hat R_t$ | advantage estimate and return target (§4) |
| $\rho_t(\theta)$ | probability ratio $\pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$ of the policy that is being optimized |
| $\epsilon$ | ratio clip range or SPO trust-region width ($\mathsf{CLIP\_RANGE}$ in the notebook) |
| $c_{\text{vf}}, c_{\text{ent}}, c_{\text{legal}}$ | value, entropy and legality loss weights |
| $P_{\mathcal{L}}(s)$ | probability the raw policy picks a legal action: $\sum_{a \in \mathcal{L}(s)} \pi_\theta(a \mid s)$ |
| $\mathcal{B}$ | a minibatch of transitions; $\mathbb{E}_{\mathcal{B}}$ its sample mean |
| $N_{\text{env}}, N_{\text{steps}}, B, K$ | parallel environments, rollout length per environment, minibatch size, epochs per rollout |

Notebook option names (all caps, e.g. `PPO_LEGALITY_COEF`) map onto these symbols. §6 lists the mapping
for every model.

## 2. Environment and rewards

Everything in this section is derived from `env_tng.py` at commit `11ee06e` (v1.1). The notebooks'
environment cell is the same code. Line numbers refer to that file.

### 2.1 State, observation and actions

| Symbol | Meaning | Code |
| --- | --- | --- |
| $b_t \in \{0,1,2\}^{23}$ | board before learner decision $t$: 0 empty, 1 goat, 2 tiger | `self.board` |
| $c_t \in \{0,\dots,6\}$ | goats captured so far | `self.eaten` |
| $\phi_t \in \{0,1\}$ | phase: 0 placement, 1 movement | `self.phase` |
| $g_t \in \{0,\dots,15\}$ | goats placed so far | `self.goats_placed` |
| $\tau_t$ | turn counter: learner decisions taken before step $t$ ($\tau_0 = 0$) | `self.turns` |
| $n^{\text{mv}}_t$ | goat *moves* (not placements) made so far in the movement phase, by whichever side controls the goats | `self.move_steps` |
| $s_t = (b_t, c_t, \phi_t) \in \mathbb{Z}^{25}$ | observation given to the policy | `get_state`, `env_tng.py:399` |

The observation omits $\tau_t$, $n^{\text{mv}}_t$ and the repeat history. All three enter the reward and the
episode endings below, so the process is only partially observed. ($g_t = \#\{i : b_{t,i}=1\} + c_t$ can be
recovered from $s_t$.)

**Actions.** $\mathcal{A} = \{0,\dots,114\}$, with $a = 5p + d$ for position $p \in \{0,\dots,22\}$ and direction
$d \in \{0,\dots,4\}$ (`encode_action`, `env_tng.py:505`). The board graph is the fixed neighbour map
$N(p, d) \in \{0,\dots,22\} \cup \{\varnothing\}$ (`self.move_map`, `env_tng.py:275`). The hub $p = 0$ connects to
$2, 3, 4, 5$ through $d = 1,\dots,4$. Tigers start on $\{0, 3, 4\}$ (`TIGER_START_POSITIONS`).

**Legal sets.** $\mathcal{L}_\omega(s)$ for learner role $\omega$; mask $m_a(s) = \mathbb{1}[a \in \mathcal{L}_\omega(s)]$
(`get_action_mask`, `env_tng.py:468`).

$$
\mathcal{L}_{\text{goat}}(s) = \begin{cases}
\{\,5p : b_p = 0\,\} & \phi = 0 \text{ (place on any empty cell, } d = 0) \\
\{\,5p + d : b_p = 1,\ N(p,d) \neq \varnothing,\ b_{N(p,d)} = 0\,\} & \phi = 1 \text{ (slide a goat to an empty neighbour)}
\end{cases}
$$

$$
\mathcal{L}_{\text{tiger}}(s) = \{\,5p + d : (p, q, \kappa, d) \in \mathcal{T}(b)\,\}
$$

$\mathcal{T}(b)$ is the tiger move list (`_tiger_moves`, `env_tng.py:1675`). For every tiger at $p$ and
direction $d$ with $j = N(p,d) \neq \varnothing$:

- **slide:** if $b_j = 0$, then $(p, j, \kappa{=}0, d) \in \mathcal{T}(b)$;
- **capture:** if $b_j = 1$, the landing is $q = Q(d)$ when $p = 0$ (hub table `HUB0_CAPTURE_JUMP`:
  $Q(1..4) = 8, 9, 10, 11$), otherwise $q = N(j, d)$ (same direction twice). The capture is legal when
  $q \neq \varnothing$ and $b_q = 0$, except that tigers on $p \in \{2,3,4,5\}$ cannot capture with $d = 1$
  (towards the hub; `env_tng.py:1708`). It gives $(p, q, \kappa{=}1, d)$ and removes the goat on $j$.

A tiger action names only the tiger and direction; the environment decides whether it is a slide or a
capture. $M(b) = |\mathcal{T}(b)|$ is **tiger mobility**.

### 2.2 One environment step

One step is a learner decision and then the opponent's reply:
$s_{t+1} \sim P(\cdot \mid s_t, a_t;\, \mu)$ with opponent policy $\mu$. Write $b^{+}_t$ for the board *after the learner
action and before the reply*. The reward depends on features measured at $b^{+}_t$, so the timing matters.

**Goat learner** (`_step_goat_transition`, `env_tng.py:899`):

1. $\tau \leftarrow \tau + 1$. Validate $a_t$ (invalid cases in §2.3).
2. Apply the placement ($g \leftarrow g+1$; when $g = 15$, $\phi \leftarrow 1$) or the slide ($n^{\text{mv}} \leftarrow n^{\text{mv}} + 1$). This gives $b^+_t$.
3. Compute the shaping features of §2.5 on $b^+_t$.
4. If $M(b^+_t) = 0$: **goat win** (no reply).
5. Tiger reply $\mu(b^+_t)$ (§2.4); a capture sets $c \leftarrow c + 1$.
6. Check endings in this order: capture threshold, turn limit, repeat stall (§2.3).

**Tiger learner** (`_step_tiger_transition`, `env_tng.py:1125`):

1. $\tau \leftarrow \tau + 1$. Validate $a_t \in \mathcal{L}_{\text{tiger}}(s_t)$.
2. Apply the slide or capture. This gives $b^+_t$. Compute $k_t$, $M(b^+_t)$ and the tiger centre score on $b^+_t$.
3. If $c \geq 6$: **tiger win** (no reply).
4. Goat reply: placement or slide (a goat slide increments $n^{\text{mv}}$). If the goats have no legal action: **tiger win**.
5. If $M(b_{t+1}) = 0$: **goat win**. Otherwise check the turn limit. The tiger learner has *no* repeat-stall rule.

### 2.3 Episode endings

| Outcome (`info["reason"]` → `winner`) | Condition | Flags $(\text{term}, \text{trunc})$ | Code |
| --- | --- | --- | --- |
| `goat_win_no_tiger_moves` → Goat | goat learner: $M(b^+_t) = 0$; tiger learner: $M(b_{t+1}) = 0$ after the goat reply | $(1, 0)$ | `:1043`, `:1299` |
| `tiger_win_capture_threshold` → Tiger | $c_{t+1} \geq 6$ (`GOATS_EATEN_FOR_TIGER_WIN`) | $(1, 0)$ | `:1095`, `:1221` |
| `tiger_win_no_goat_moves` → Tiger | tiger learner only: the goat opponent has no legal action | $(1, 0)$ | `:1278` |
| `max_timeout` → MaxTimeout | $\tau_{t+1} \geq \mathsf{MAX\_TURNS}$, after the checks above | $(0, 1)$ | `:1101`, `:1305` |
| `repeat_timeout` → RepeatTimeout | goat learner only: $h_t + 1 \geq \mathsf{MAX\_REPEATS}$ (see $h_t$ in §2.5) | $(0, 1)$ | `:1113` |
| `invalid_soft` → none | goat: malformed action, or an illegal *placement*; tiger: malformed action | $(0, 1)$ | `:924–966`, `:1150–1162` |
| `invalid_hard` → none | goat: an illegal *slide*; tiger: any $(p,d) \notin \mathcal{L}_{\text{tiger}}$ | $(0, 1)$ | `:978`, `:1173` |

In the default mode an invalid action leaves the board unchanged, sets `terminate`, and reports the episode
as **truncated, not terminated**. The next call returns `already_terminated` with reward 0.

Notes:
- Stable-Baselines3 bootstraps truncated episodes with $\gamma V(s_{t+1})$, and here
  $s_{t+1} = s_t$. Under unmasked training the invalid-action penalty is therefore largely cancelled by the
  bootstrap. The control-center notebook's `IllegalMoveEndsGame` wrapper re-labels these endings as
  terminated for plain PPO.
- The turn-limit check comes before the repeat check, so a stall on the last turn is reported as `max_timeout`.

**"Lose a turn" variant** (`truncate_on_invalid=False`, `step`, `env_tng.py:389–396`). After an invalid
action the board, opponent and repeat history are left as they were ($s_{t+1} = s_t$), but $\tau$ has already
been incremented. Then:

$$
(\text{term}_t, \text{trunc}_t) = \big(0,\ \mathbb{1}[\tau_{t+1} \geq \mathsf{MAX\_TURNS}]\big)
$$

On the final turn the reason is rewritten to `max_timeout` (the original is kept in `info["invalid_reason"]`),
so the reward of that attempt is the timeout reward of §2.5/§2.6, not the invalid-action penalty.

### 2.4 Opponent policies $\mu$

All scripted opponents draw from the global `random` / `np.random` generators, which is why evaluation
seeds those as well as the environment.

- **Greedy tiger** (`_greedy_tiger_move`, `env_tng.py:1655`). Let $\mathcal{C} \subseteq \mathcal{T}(b^+)$ be the
  captures. With probability $\beta = \mathsf{BASE\_TIGER\_CAPTURE\_BIAS}$ (default 1), and only if
  $\mathcal{C} \neq \varnothing$, the tiger plays a uniform draw from $\mathcal{C}$. Otherwise it plays a uniform draw
  from all of $\mathcal{T}(b^+)$, captures included.
- **Smart tiger** (`_smart_tiger_move`, `env_tng.py:1387`). The first rule that applies wins, using the
  fixed square values $v(\cdot)$ (`POS_VALS`; $v = 8$ on $\{0, 9, 10\}$, 7 on $\{3, 4, 8, 11, 15, 16\}$, …, 4 on the rim):
  1. **Capture:** shuffle $\mathcal{C}$, then take $\arg\max_{(p,q,1)} v(q)$. Ties are broken by the shuffle.
  2. **Anchors:** if no tiger is on 0, or no tiger is on 9 or 10, run a breadth-first search through
     empty cells from each tiger (except those already holding a satisfied anchor) to the missing anchor(s):
     $\{0\}$, and the empty cells among $\{9, 10\}$, or both if neither is empty. Take the first step of the
     shortest path (ties: lowest tiger index, then anchor 0 before $\{9,10\}$). Use it only if it is a legal slide.
  3. **Roam:** take a uniform draw over legal moves with $v(q) \geq 5$. Moves leaving square 0 are excluded,
     as are moves leaving the only tiger on 9/10.
  4. **Fallback:** take a uniform draw from $\mathcal{T}(b^+)$.
- **Random goat** (`env_tng.py:1245`): a uniform draw from $\mathcal{L}_{\text{goat}}$.
- **Model goat / model tiger:** a callback $f(s, m) \mapsto a$. The notebooks pass a frozen saved policy
  $\pi_{\bar\theta}$ and play $\arg\max_{a \in \mathcal{L}} \pi_{\bar\theta}(a \mid s)$.
  - If the goat model's answer is missing, illegal or raises an error, the goat falls back to the random goat (`env_tng.py:1272–1275`).
  - If the tiger model's answer is illegal or missing, `_model_tiger_move` returns `None` and the step raises
    `RuntimeError` (`env_tng.py:1052–1057`). There is no fallback.

### 2.5 Goat-learner reward $r^{\text{goat}}_t$

`_sparse_reward_goat`, `env_tng.py:616`. Features (all computed in `_step_goat_transition`):

| Symbol | Definition | Measured at | Code |
| --- | --- | --- | --- |
| $M^+_t$ | $M(b^+_t)$, tiger mobility after the goat action; $M^+_{-1} = M(b_0)$ | before the reply | `:999` |
| $\Delta M_t$ | $M^+_{t-1} - M^+_t$ (positive = goats removed tiger options) | consecutive pre-reply boards | `:1011` |
| $U(b)$ | number of cells with $b_i \in \{0,1\}$ that no tiger can reach by walking through **empty** cells | — | `:1755` |
| $\Delta U_t$ | $U(b^+_t) - U(b^+_{t-1})$ | pre-reply boards | `:1015` |
| $\Sigma(b)$ | $\sum_{i<j} \text{dist}(x_i, x_j)$ over tiger squares $x$; graph distance ignoring pieces | — | `:1830` |
| $\Delta\Sigma_t$ | $\Sigma(b^+_{t-1}) - \Sigma(b^+_t)$ (positive = tigers pushed together) | pre-reply boards | `:1019` |
| $\ell_t$ | near-lock: $\mathbb{1}[0 < M^+_t \leq 2]$ | $b^+_t$ | `:1005` |
| $Z^{\text{g}}_t$ | $\sum_{i \in K} \big(\mathsf{REWARD\_CENTER\_GOAT}\,\mathbb{1}[b^+_{t,i}=1] + \mathsf{REWARD\_CENTER\_TIGER}\,\mathbb{1}[b^+_{t,i}=2]\big)$, $K = \{0, 9, 10, 15, 16\}$ | $b^+_t$ | `:1022` |
| $k_t$ | goats captured by the reply: $c_{t+1} - c_t$ | after the reply | `:1080` |
| $h_t$ | earlier visits of the key $(b_{t+1}, c_{t+1}, \phi_{t+1})$ at post-reply checkpoints in this episode | after the reply | `:1107` |
| $\eta_t$ | shaping decay: $\mathsf{GOAT\_SHAPING\_DECAY\_BASE}^{\,n^{\text{mv}}_{t+1}}$ if $\phi_{t+1} = 1$, else 1 | after the step | `:679` |
| $\sigma_t$ | step cost: $\mathsf{REWARD\_STEP}$ if $\phi_t = 0$, else $\max\big(\mathsf{MOVE\_STEP\_BASE} + \mathsf{MOVE\_STEP\_SLOPE}\, n^{\text{mv}}_t,\ \mathsf{MOVE\_STEP\_MIN}\big)$ | before the action | `:949` |

**Invalid actions and the goat win replace everything else:**

$$
r^{\text{goat}}_t = \begin{cases}
\mathsf{REWARD\_INVALID\_SOFT} & \text{invalid\_soft (illegal placement)} \\
\mathsf{REWARD\_INVALID\_HARD} & \text{invalid\_hard (illegal slide)} \\
\mathsf{REWARD\_GOAT\_WIN} - \mathsf{GOAT\_WIN\_TURN\_DECAY}\cdot \tau_{t+1} & \text{goat win} \\
\sum_k r^{(k)}_t & \text{otherwise}
\end{cases}
$$

Let $\chi_t = \mathbb{1}[\text{reason} \in \{\text{capture threshold}, \text{max\_timeout}, \text{repeat\_timeout}\}]$ flag a losing ending.
The decomposition matches the `info["reward_*_component"]` fields:

| $k$ | Component $r^{(k)}_t$ | Formula |
| --- | --- | --- |
| step | `reward_step_component` | $\sigma_t$ |
| near-lock | `reward_near_lock_component` | $(1-\chi_t)\ \mathsf{NEAR\_LOCK\_BONUS}\ \eta_t\ \ell_t$ |
| block | `reward_block_component` | $(1-\chi_t)\, B_t$, with $B_t$ below |
| bubble | `reward_bubble_component` | $(1-\chi_t)\ \mathsf{REWARD\_BUBBLE\_SPACE}\ \tanh(\Delta U_t)\ \eta_t\ \mathbb{1}[\Delta U_t > 0]$ |
| cluster | `reward_cluster_component` | $(1-\chi_t)\ \mathsf{REWARD\_CLUSTER\_TIGERS}\ \tanh(\Delta\Sigma_t)\ \eta_t\ \mathbb{1}[\Delta\Sigma_t > 0]$ |
| centre | `reward_center_component` | $(1-\chi_t)\ Z^{\text{g}}_t\ \mathbb{1}[\phi_{t+1} = 0]$ |
| goat eaten | `reward_goat_eaten_component` | $\mathsf{REWARD\_GOAT\_EATEN}\ k_t$ |
| repeat | `reward_repeat_component` | $(1-\chi_t)\ \mathsf{REWARD\_REPEAT\_STATE}\ h_t^2$ |
| terminal | `reward_terminal_component` | $\mathsf{REWARD\_TIGER\_WIN}\cdot S_t$, with $S_t$ below |
| (logged) | `reward_decay_mult` | $\eta_t$ (a multiplier, not added) |

$$
B_t = \begin{cases} \lvert\mathsf{REWARD\_BLOCK\_TIGER}\rvert\ \tanh(\Delta M_t)\ \eta_t & \Delta M_t > 0 \\ \lvert\mathsf{REWARD\_BLOCK\_TIGER}\rvert\ \mathsf{MOBILITY\_BACKSLIDE\_SCALE}\ \Delta M_t & \Delta M_t < 0 \\ 0 & \Delta M_t = 0 \end{cases}
\qquad
S_t = \begin{cases} 1 & \text{capture threshold} \\ \mathsf{MAX\_TIMEOUT\_SCALE} & \text{max\_timeout} \\ \mathsf{REPEAT\_STALL\_SCALE} & \text{repeat\_timeout} \\ 0 & \text{otherwise} \end{cases}
$$

Notes:
- Positive shaping is squashed by $\tanh$ and decayed by $\eta_t$. The mobility *backslide* is linear in
  $\Delta M_t$ and not decayed. The step cost, centre term and capture penalty are never decayed.
- $U$ counts every goat cell as unreachable, because the search never enters a goat square. Every placement
  on a reachable cell therefore raises $U$ by at least 1, so $r^{(\text{bubble})}$ fires on most placements.
- The knob comment for $\mathsf{REWARD\_CLUSTER\_TIGERS}$ says "spreading tigers apart", but the code rewards
  a *decrease* in $\Sigma$, i.e. tigers pushed together.
- The centre score is measured on $b^+_t$ (before the reply), but gated by the post-reply phase $\phi_{t+1}$. The
  step that places the 15th goat therefore earns no centre term.
- A `late` factor, $\min\big(\max(\tau_{t+1} - \mathsf{LATE\_GAME\_START\_TURN}, 0) / (\mathsf{MAX\_TURNS} - \mathsf{LATE\_GAME\_START\_TURN}), 1\big)$,
  is computed and logged (`:1006`) but **not used** by the reward.
- The stall fires at $h_t = \mathsf{MAX\_REPEATS} - 1$; on that step the terminal term replaces the repeat penalty.
- In the lose-a-turn variant, an invalid attempt on the final turn gets
  $\mathsf{REWARD\_STEP} + \mathsf{REWARD\_TIGER\_WIN}\cdot\mathsf{MAX\_TIMEOUT\_SCALE}$. Its `info` has no `step_penalty` and no features, so the defaults apply.

### 2.6 Tiger-learner reward $r^{\text{tiger}}_t$

`_sparse_reward_tiger`, `env_tng.py:790`. Features (from `_step_tiger_transition`):

| Symbol | Definition | Code |
| --- | --- | --- |
| $k_t$ | goats captured by the tiger's own move ($0$ or $1$) | `:1198` |
| $\Delta M^{\text{T}}_t$ | $M(b^+_t) - M(b^+_{t-1})$: mobility after this tiger move minus mobility after the previous tiger move ($M(b_0)$ at reset). The goat reply in between is included | `:1200–1203` |
| $Z^{\text{t}}_t$ | $\lvert\mathsf{REWARD\_CENTER\_GOAT}\rvert\ \big(\#\{i \in K: b^+_{t,i} = 2\} - \#\{i \in K: b^+_{t,i} = 1\}\big)$ | `:1205–1210` |

$$
r^{\text{tiger}}_t = \begin{cases}
\mathsf{REWARD\_INVALID\_SOFT} & \text{invalid\_soft (malformed action)} \\
-\lvert\mathsf{REWARD\_INVALID\_HARD}\rvert & \text{invalid\_hard (illegal tiger move)} \\
\mathsf{REWARD\_STEP} + \mathsf{REWARD\_TIGER\_CAPTURE}\ k_t + \lvert\mathsf{REWARD\_BLOCK\_TIGER}\rvert\ \Delta M^{\text{T}}_t + \mathbb{1}[\phi_{t+1} = 0]\ Z^{\text{t}}_t + r^{\text{term}}_t & \text{otherwise}
\end{cases}
$$

$$
r^{\text{term}}_t = \begin{cases}
+\mathsf{REWARD\_TIGER\_WIN\_BONUS} & \text{capture threshold or no goat moves} \\
-\mathsf{REWARD\_TIGER\_LOSS\_PENALTY} & \text{goat win} \\
-\mathsf{REWARD\_TIGER\_LOSS\_PENALTY}\cdot\mathsf{MAX\_TIMEOUT\_SCALE} & \text{max\_timeout} \\
0 & \text{otherwise}
\end{cases}
$$

Notes:
- Unlike the goat reward:
  - the shaping terms are kept on terminal steps;
  - mobility is linear (no $\tanh$, no decay);
  - there is no movement-phase step ramp, no repeat penalty, and no `reward_*` component logging. Only the
    total is returned, and the component fields stay at their zero defaults.
- The tiger centre term reuses $\lvert\mathsf{REWARD\_CENTER\_GOAT}\rvert$ for both signs;
  $\mathsf{REWARD\_CENTER\_TIGER}$ is ignored.
- `invalid_hard` returned $+\lvert\mathsf{REWARD\_INVALID\_HARD}\rvert$ before v1.1, rewarding illegal tiger moves.
  It is negative from v1.1 on (`env_tng.py:822`).
- In the lose-a-turn variant, an invalid attempt on the final turn gets
  $\mathsf{REWARD\_STEP} - \mathsf{REWARD\_TIGER\_LOSS\_PENALTY}\cdot\mathsf{MAX\_TIMEOUT\_SCALE}$.

### 2.7 Reward and game knobs (`DEFAULT_KNOBS`, `env_tng.py:78`)

Any knob can be overridden per run or phase with `reward_overrides` / `REWARD_OVERRIDES`. Fixed game
constants: 23 squares, 15 goats to place, 6 captures to win, tigers start on $\{0, 3, 4\}$.

| Knob | Default | Used in |
| --- | --- | --- |
| $\mathsf{REWARD\_GOAT\_WIN}$ | 3.2 | goat win, before turn decay |
| $\mathsf{REWARD\_TIGER\_WIN}$ | −3.2 | goat terminal loss; scaled for timeouts |
| $\mathsf{REWARD\_TIGER\_CAPTURE}$ | 0.35 | tiger: per capture |
| $\mathsf{REWARD\_TIGER\_WIN\_BONUS}$ | 3.2 | tiger win |
| $\mathsf{REWARD\_TIGER\_LOSS\_PENALTY}$ | 3.2 | tiger loss and timeout (subtracted) |
| $\mathsf{REWARD\_STEP}$ | −0.001 | step cost: goat placement phase, every tiger step |
| $\mathsf{REWARD\_GOAT\_EATEN}$ | −0.35 | goat: per goat captured |
| $\mathsf{MAX\_TIMEOUT\_SCALE}$ | 1.5 | timeout multiplier (both roles) |
| $\mathsf{REPEAT\_STALL\_SCALE}$ | 1.25 | goat repeat-stall multiplier |
| $\mathsf{MAX\_TURNS}$ | 100 | turn limit (also the `max_turns` argument) |
| $\mathsf{MOVE\_STEP\_BASE}$ | −0.01 | goat movement-phase step cost, intercept |
| $\mathsf{MOVE\_STEP\_SLOPE}$ | −0.01 | goat movement-phase step cost, per prior goat move |
| $\mathsf{MOVE\_STEP\_MIN}$ | −0.6 | goat movement-phase step cost, floor |
| $\mathsf{GOAT\_WIN\_TURN\_DECAY}$ | 0.01 | goat win reward lost per turn |
| $\mathsf{REWARD\_BLOCK\_TIGER}$ | 0.08 | mobility shaping magnitude (both roles) |
| $\mathsf{REWARD\_BUBBLE\_SPACE}$ | 0.02 | goat: unreachable-cell shaping |
| $\mathsf{REWARD\_CLUSTER\_TIGERS}$ | 0.02 | goat: tiger-spread shaping |
| $\mathsf{REWARD\_CENTER\_GOAT}$ | 0.03 | centre squares (goat; magnitude for tiger) |
| $\mathsf{REWARD\_CENTER\_TIGER}$ | −0.03 | goat's centre score, per tiger on $K$ |
| $\mathsf{NEAR\_LOCK\_BONUS}$ | 0.05 | goat near-lock bonus |
| $\mathsf{LATE\_GAME\_START\_TURN}$ | 40 | `late` factor only (unused by the reward) |
| $\mathsf{MOBILITY\_BACKSLIDE\_SCALE}$ | 0.5 | goat: penalty multiplier when $\Delta M_t < 0$ |
| $\mathsf{GOAT\_SHAPING\_DECAY\_BASE}$ | 0.95 | base of $\eta_t$ |
| $\mathsf{REWARD\_INVALID\_SOFT}$ | −0.05 | invalid\_soft |
| $\mathsf{REWARD\_INVALID\_HARD}$ | −1.0 | invalid\_hard (tiger uses $-\lvert\cdot\rvert$) |
| $\mathsf{REWARD\_REPEAT\_STATE}$ | −0.5 | goat repeat penalty, times $h_t^2$ |
| $\mathsf{MAX\_REPEATS}$ | 4 | repeat-stall threshold |
| $\mathsf{BASE\_TIGER\_CAPTURE\_BIAS}$ | 1.0 | greedy tiger: probability of preferring captures |

## 3. Policy and value network

### 3.1 Architecture

Every model in the project, whether shipped, notebook-trained or from the study, uses Stable-Baselines3's
`MlpPolicy` with `net_arch = [256, 256, 256]`. Since SB3 1.8 this list builds **two separate trunks** of the
same shape, one for the policy and one for the value, with no shared hidden layers. The observation is cast
to `float32` without normalization: $x = s_t \in \mathbb{R}^{25}$.

$$
h^{\pi}_0 = x, \qquad h^{\pi}_\ell = \operatorname{act}\!\big(W^{\pi}_\ell h^{\pi}_{\ell-1} + \beta^{\pi}_\ell\big),\ \ell = 1,2,3, \qquad z_\theta(s) = W^{\pi}_4 h^{\pi}_3 + \beta^{\pi}_4 \in \mathbb{R}^{115}
$$

$$
h^{V}_0 = x, \qquad h^{V}_\ell = \operatorname{act}\!\big(W^{V}_\ell h^{V}_{\ell-1} + \beta^{V}_\ell\big),\ \ell = 1,2,3, \qquad V_\psi(s) = w^{V\top} h^{V}_3 + \beta^{V}_4
$$

Here $\theta = \{W^\pi_\ell, \beta^\pi_\ell\}$ and $\psi = \{W^V_\ell, \beta^V_\ell, w^V\}$. The activation $\operatorname{act}$ is
$\tanh$ (SB3 default: all shipped models and the study's main arm) or ReLU (`TnG_MPPO_Notebook`,
`TnG_PPO_Notebook`, control-center default since v1.2, and the study's ReLU arm).

| Component | Shape | Parameters |
| --- | --- | ---: |
| policy trunk | 25 → 256 → 256 → 256 | 138,240 |
| value trunk | 25 → 256 → 256 → 256 | 138,240 |
| action head | 256 → 115 | 29,555 |
| value head | 256 → 1 | 257 |
| **total** | | **306,292** |

**Initialization.** Orthogonal weights, zero biases. Gains: $\sqrt 2$ for both trunks, $0.01$ for the action
head (near-uniform initial policy), $1$ for the value head.
**Optimizer.** One Adam optimizer for $(\theta, \psi)$ with learning rate $\alpha$ and $\epsilon_{\text{Adam}} = 10^{-5}$.
Before each step the gradient is clipped to global norm $g_{\max}$ (`max_grad_norm`).

### 3.2 Raw and masked policies

The **raw policy** is the network output:

$$
\pi_\theta(a \mid s) = \frac{\exp z_{\theta,a}(s)}{\sum_{b \in \mathcal{A}} \exp z_{\theta,b}(s)}
$$

The **masked policy** (sb3-contrib `MaskableCategorical`) replaces each illegal logit with the constant $-10^{8}$:

$$
\tilde z_a = \begin{cases} z_{\theta,a}(s) & a \in \mathcal{L}(s) \\ -10^{8} & a \notin \mathcal{L}(s) \end{cases}
\qquad
\pi^{m}_\theta(a \mid s) = \operatorname{softmax}(\tilde z)_a = \frac{\pi_\theta(a \mid s)\,\mathbb{1}[a \in \mathcal{L}(s)]}{P_{\mathcal{L}}(s)}
$$

Two consequences matter for §9:

$$
\frac{\partial \tilde z_a}{\partial z_{\theta,a}} = 0 \quad \text{for } a \notin \mathcal{L}(s)
\qquad\text{and}\qquad
\mathcal{H}\big[\pi^m_\theta(\cdot \mid s)\big] = -\sum_{a \in \mathcal{L}(s)} \pi^m_\theta(a \mid s) \log \pi^m_\theta(a \mid s)
$$

No term of a masked loss ever puts a gradient on an illegal logit. Those logits drift only through the shared
trunk, so nothing teaches the network that illegal actions are bad.

**Action selection.**

| Use | Masked | Raw |
| --- | --- | --- |
| training rollouts | $a_t \sim \pi^m_\theta(\cdot \mid s_t)$ | $a_t \sim \pi_\theta(\cdot \mid s_t)$ |
| deterministic play | $\arg\max_{a \in \mathcal{L}(s)} z_{\theta,a}(s)$ | $\arg\max_{a \in \mathcal{A}} z_{\theta,a}(s)$ |
| saved-model opponents | $\arg\max_{a \in \mathcal{L}(s)} z_{\bar\theta,a}(s)$ for frozen $\bar\theta$ | — |

The control-center notebook computes masked choices on log-probabilities. Its choices were verified to be
identical to `MaskablePPO.predict` (§8).

## 4. Data collection, returns and advantages

**Rollouts.**
- $N_{\text{env}}$ environment copies run in lockstep: `DummyVecEnv` (sequential) in the notebooks, or
  subprocess environments in `train.py`.
- Each update collects $N_{\text{steps}}$ transitions per copy, so a rollout holds $D = N_{\text{env}} N_{\text{steps}}$
  transitions.
- Each transition stores $\big(s_t, a_t, r_t, d_t, V_{\psi_{\text{old}}}(s_t), \log \pi_{\text{old}}(a_t \mid s_t), m(s_t)\big)$,
  where $d_t$ marks the first step of an episode.
- Masked training stores $\log \pi^m_{\text{old}}$. The mask is stored by MaskablePPO, and by the control
  center's plain PPO for its legality loss (§5.4).

**Truncation bootstrap (SB3).** If an episode ends with $\text{trunc}_t = 1$ and $\text{term}_t = 0$, the reward is
augmented before GAE:

$$
r_t \leftarrow r_t + \gamma\, V_{\psi_{\text{old}}}\big(s^{\text{final}}_{t+1}\big)
$$

Which endings are truncations is set by §2.3: timeouts, repeat stalls and (by default) invalid actions.
The `"end_game"` illegal-move handling (§5.4) relabels invalid actions as terminations, so no bootstrap is added.

**Generalized advantage estimation.**

$$
\delta_t = r_t + \gamma\,(1 - e_{t+1})\,V_{\psi_{\text{old}}}(s_{t+1}) - V_{\psi_{\text{old}}}(s_t),
\qquad
\hat A_t = \sum_{l \geq 0} (\gamma\lambda)^l \Big(\prod_{j=1}^{l} (1 - e_{t+j})\Big)\,\delta_{t+l},
\qquad
\hat R_t = \hat A_t + V_{\psi_{\text{old}}}(s_t)
$$

Here $e_{t+1} = 1$ when the episode ended at step $t$. The sum stops at the rollout boundary, where the value of
the last observation bootstraps the tail.

**Optimization schedule.** Each rollout gets $N_{\text{ep}}$ epochs. Each epoch is a pass over a random
permutation of the $D$ transitions in minibatches $\mathcal{B}$ of size $B$; it requires $B \mid D$. Within a
minibatch the advantages are normalized:

$$
\tilde A_t = \frac{\hat A_t - \operatorname{mean}_{\mathcal{B}}(\hat A)}{\operatorname{std}_{\mathcal{B}}(\hat A) + 10^{-8}}
$$

## 5. Objectives

Every update minimizes, per minibatch:

$$
\mathcal{J}(\theta, \psi) = \mathcal{J}^{\pi}(\theta) + c_{\text{vf}}\,\mathcal{J}^{V}(\psi) + c_{\text{ent}}\,\mathcal{J}^{\mathcal{H}}(\theta) + c_{\text{legal}}\,\mathcal{J}^{\text{legal}}(\theta)
$$

with $\mathcal{J}^{\mathcal{H}} = -\mathbb{E}_{\mathcal{B}}\big[\mathcal{H}[\pi(\cdot \mid s_t)]\big]$. The algorithms differ in which policy
($\pi^m_\theta$ or $\pi_\theta$) enters the ratio and the entropy, in $\mathcal{J}^\pi$ and $\mathcal{J}^V$, and in whether $c_{\text{legal}} > 0$.

### 5.1 Maskable PPO (shipped models, `train.py`, `TnG_MPPO_Notebook`, control center with `"maskable_ppo"`)

$$
\rho_t(\theta) = \frac{\pi^m_\theta(a_t \mid s_t)}{\pi^m_{\theta_{\text{old}}}(a_t \mid s_t)},
\qquad
\mathcal{J}^{\pi}_{\text{clip}} = -\mathbb{E}_{\mathcal{B}}\Big[\min\big(\rho_t \tilde A_t,\ \operatorname{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\tilde A_t\big)\Big]
$$

$$
\mathcal{J}^{V}_{\text{mse}} = \mathbb{E}_{\mathcal{B}}\big[(\hat R_t - V_\psi(s_t))^2\big],
\qquad
\mathcal{H} \text{ over } \mathcal{L}(s_t) \text{ only (§3.2)},
\qquad
c_{\text{legal}} = 0
$$

### 5.2 MaskablePPO with masking off (`TnG_PPO_Notebook`, `TRAIN_WITH_MASK = False`)

`MaskablePPO.learn(use_masking=False)` stores no masks, so $\pi^m_\theta \equiv \pi_\theta$ everywhere. The objective is
§5.1 with the raw policy in the ratio and the entropy over all 115 actions. It is exactly standard PPO.
This notebook trains with the "lose a turn" environment (`truncate_on_invalid=False`, §2.3), and its shipped
configuration overrides the invalid-move knobs. Each illegal attempt therefore costs its knob reward and one
turn, and play continues.

### 5.3 Control-center plain PPO (`LossConfigurablePPO`, `ALGORITHM = "ppo"`)

The ratio uses the raw policy, $\rho_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{\text{old}}}(a_t \mid s_t)$, and the entropy
runs over all of $\mathcal{A}$. `PPO_POLICY_LOSS` selects $\mathcal{J}^\pi$:

| Option | $\mathcal{J}^\pi$ |
| --- | --- |
| `"clip"` | $\mathcal{J}^{\pi}_{\text{clip}}$ as in §5.1, with the raw ratio |
| `"kl_penalty"` | $-\mathbb{E}_{\mathcal{B}}[\rho_t \tilde A_t] + \beta\,\hat k$, where $\hat k = \mathbb{E}_{\mathcal{B}}\big[(\rho_t - 1) - \log \rho_t\big]$ |
| `"spo"` | $-\mathbb{E}_{\mathcal{B}}\Big[\rho_t \tilde A_t - \dfrac{\lvert\tilde A_t\rvert}{2\epsilon}(\rho_t - 1)^2\Big]$ |

$\hat k$ is Schulman's unbiased, non-negative estimator of $\mathrm{KL}(\pi_{\text{old}} \Vert \pi_\theta)$, and it is differentiable.
For `"kl_penalty"`, $\beta$ starts at $\beta_0$ (`PPO_KL_BETA`) and is adapted once per rollout from the mean $\bar k$
of all minibatch estimates:

$$
\beta \leftarrow \begin{cases} \min(2\beta,\ 10^{4}) & \bar k > 1.5\, d_{\text{targ}} \\ \max(\beta/2,\ 10^{-4}) & \bar k < d_{\text{targ}}/1.5 \\ \beta & \text{otherwise} \end{cases}
\qquad d_{\text{targ}} = \texttt{PPO\_KL\_TARGET}
$$

`PPO_VALUE_LOSS` selects $\mathcal{J}^V$: `"mse"` as in §5.1, or `"huber"` with threshold $\delta_H$:

$$
\mathcal{J}^V_{\text{huber}} = \mathbb{E}_{\mathcal{B}}\big[\ell_{\delta_H}(V_\psi(s_t) - \hat R_t)\big],
\qquad
\ell_{\delta_H}(x) = \begin{cases} \tfrac12 x^2 & \lvert x\rvert \leq \delta_H \\ \delta_H\big(\lvert x\rvert - \tfrac12\delta_H\big) & \text{otherwise} \end{cases}
$$

With `"clip"`, `"mse"` and $c_{\text{legal}} = 0$, the update is identical to SB3 `PPO`; this was verified bit-for-bit in §8.

### 5.4 Legality loss and illegal-move handling (control center)

The legal-action probability is $P_{\mathcal{L}}(s) = \sum_{a \in \mathcal{L}(s)} \pi_\theta(a \mid s)$. It is computed stably as
$\log P_{\mathcal{L}} = \operatorname{logsumexp}_{a \in \mathcal{L}(s)} \log \pi_\theta(a \mid s)$. `PPO_LEGALITY_LOSS` selects:

$$
\mathcal{J}^{\text{legal}}_{\text{nll}} = -\mathbb{E}_{\mathcal{B}}\big[\log P_{\mathcal{L}}(s_t)\big]
\qquad\qquad
\mathcal{J}^{\text{legal}}_{\text{mass}} = \mathbb{E}_{\mathcal{B}}\big[1 - P_{\mathcal{L}}(s_t)\big]
$$

`"nll"` is the binary cross-entropy of the event "the sampled action is legal" with target 1. It is
supervised: it uses the stored mask as the label, does not depend on $a_t$ or $\tilde A_t$, and applies to
every state in $\mathcal{B}$. Its gradient with respect to the logits is

$$
\frac{\partial \mathcal{J}^{\text{legal}}_{\text{nll}}}{\partial z_a} = \pi_\theta(a \mid s) - \mathbb{1}[a \in \mathcal{L}(s)]\,\frac{\pi_\theta(a \mid s)}{P_{\mathcal{L}}(s)}
\qquad\qquad
\frac{\partial \mathcal{J}^{\text{legal}}_{\text{mass}}}{\partial z_a} = \pi_\theta(a \mid s)\,\big(P_{\mathcal{L}}(s) - \mathbb{1}[a \in \mathcal{L}(s)]\big)
$$

For an illegal $a$, the `nll` gradient is $\pi_\theta(a \mid s)$, but the `mass` gradient is
$\pi_\theta(a \mid s)\,P_{\mathcal{L}}(s)$. That vanishes when the policy puts almost no probability on legal
moves, which is exactly the state of a fresh network. §9 shows the predicted failure of `"mass"`.

**Mask alignment.** SB3 calls the step callback after `env.step` and before `rollout_buffer.add`. The control
center records $m(s_t)$ into buffer slot $t$ at that moment, then reads $m(s_{t+1})$ for the next slot, after
any automatic reset. §8 verifies the alignment.

**Illegal-move handling** (`ILLEGAL_MOVE_HANDLING`; plain PPO training only):

| Option | Transition after $a_t \notin \mathcal{L}(s_t)$ | Reward | Flags |
| --- | --- | --- | --- |
| `"end_game"` | episode ends, board unchanged | $r_{\text{ill}}$ | $(\text{term}, \text{trunc}) = (1, 0)$: no bootstrap |
| `"lose_turn"` | $s_{t+1} = s_t$, $\tau_{t+1} = \tau_t + 1$ | $r_{\text{ill}}$ (timeout reward on the last turn, §2.3) | $(0, \mathbb{1}[\tau_{t+1} \geq \mathsf{MAX\_TURNS}])$ |

$r_{\text{ill}}$ is `PPO_ILLEGAL_MOVE_REWARD`. It is written into both $\mathsf{REWARD\_INVALID\_SOFT}$ and
$\mathsf{REWARD\_INVALID\_HARD}$ unless a run sets them explicitly. The default is $-3.2$, equal to
$\mathsf{REWARD\_TIGER\_WIN}$, so an illegal move is as bad as losing.

## 6. How each model was trained

### 6.1 Opponent mixing and curricula

**Mixing.** A run with mixing probability $q$ (`MIX_PROB`, `mix_prob`) draws the opponent once per episode, at reset:

$$
\mu \sim \begin{cases} q\,\delta_{\text{smart}} + (1-q)\,\delta_{\text{greedy}} & \omega = \text{goat} \\ q\,\delta_{\text{model goat}} + (1-q)\,\delta_{\text{random goat}} & \omega = \text{tiger} \end{cases}
$$

**Curricula.** A run with phases $j = 1, \dots, J$ continues one parameter vector:
- Phase $j$ trains for $T_j$ steps against $\mu_j$ with knobs $\mathsf{K}_j$.
- It starts from $(\theta, \psi)$ at the end of phase $j-1$, keeping the optimizer state and step counter.
- Phases do not inherit settings from each other. Each phase's knobs are the environment defaults, then the
  global overrides, then the run's overrides, then the phase's own.

**Seeds.** Python, NumPy, PyTorch and the environments are seeded with the run seed. Copy $i$ of the
$N_{\text{env}}$ environments uses seed $+\, i$.

### 6.2 Shipped models (`stable_models/models/`)

All six are MaskablePPO (§5.1) with a tanh MLP (§3), $\gamma = 0.99$, $\lambda = 0.95$, $\epsilon = 0.2$ (constant),
$\alpha = 3\times10^{-4}$ (constant), $c_{\text{ent}} = 0.02$, $c_{\text{vf}} = 0.5$, $g_{\max} = 0.5$ and $N_{\text{env}} = 16$. They
were trained in March 2026 with `train.py` (then `train_abc.py`) on the environment of that time. That
environment's reward code predates v1.1, so its knob values may differ. Their evaluations in §9 use the
v1.1 environment.

| Model | $\omega$ | Opponent schedule | Steps | $N_{\text{steps}}$ | $B$ | $N_{\text{ep}}$ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `mppo_NormalGoat030726` | goat | greedy tiger | 100,007,936 | 4096 | 256 | 10 |
| `mppo_SmartGoat030726` | goat | smart tiger | 100,007,936 | 4096 | 256 | 10 |
| `mppo_RobustGoat030726` | goat | curriculum: greedy → smart → mix $q = 0.5$, $\approx$20.05M steps each (boundaries at 20,054,016 and 40,108,032 in its logs) | 60,162,048 | 4096 | 256 | 10 |
| `best_tiger_vs_normal_goat_10M_…` | tiger | model goat (NormalGoat, per name) | 5,005,312 | 512 | 128 | 5 |
| `best_tiger_vs_smart_goat_10M_…` | tiger | model goat (SmartGoat, per name) | 5,005,312 | 512 | 128 | 5 |
| `mppo_RobustTiger030726` | tiger | model goat (RobustGoat), realized model-goat fraction 1.0 in its log | 20,054,016 | 4096 | 256 | 10 |

The two `best_tiger` files say 10M in their names but hold 5.0M steps. They are the best checkpoints of
10M-step runs. Their rollout sizes match `train.py`'s CPU preset (`DEVICE_MODE = "cpu"`); the others match
its GPU preset.

### 6.3 Models trained by the notebooks

| Notebook | Algorithm | Environment | Defaults |
| --- | --- | --- | --- |
| `TnG_MPPO_Notebook` | §5.1 | default (an illegal move cannot occur) | ReLU, $N_{\text{env}} = 8$, $N_{\text{steps}} = 128$, $B = 64$, $N_{\text{ep}} = 5$, $\alpha = 3\times10^{-4}$, $c_{\text{ent}} = 0.02$, 2,048 steps |
| `TnG_PPO_Notebook` | §5.2 when `TRAIN_WITH_MASK = False`, else §5.1 | lose-a-turn for both training and evaluation | ReLU and the sizes above; 600,000 steps; reward overrides $\mathsf{REWARD\_GOAT\_WIN} = 6$, $\mathsf{REWARD\_TIGER\_WIN} = -3$, $\mathsf{REWARD\_BLOCK\_TIGER} = 0.16$, $\mathsf{NEAR\_LOCK\_BONUS} = 0.08$, $\mathsf{REWARD\_INVALID\_SOFT} = -0.5$, $\mathsf{REWARD\_INVALID\_HARD} = -1$ |
| `TnG_Notebook` (control center) | §5.1, or §5.3–5.4 per run (`"algorithm"`) | default for Maskable PPO; `ILLEGAL_MOVE_HANDLING` for plain PPO | ReLU, the sizes above, $\lambda = 0.95$, $\epsilon = 0.2$, $c_{\text{vf}} = 0.5$, $g_{\max} = 0.5$; plain PPO: clip, MSE, `nll`, $c_{\text{legal}} = 0.5$, $r_{\text{ill}} = -3.2$, `"end_game"` |

Every notebook-trained model saves `meta/phase_NN_settings.json` next to it. That file records the
algorithm, loss settings, illegal-move handling, seed, activation, every knob in effect ($\mathsf{K}_j$) and the
effective PPO hyperparameters, so §6.4 can be reproduced from it.

### 6.4 Models trained for the legality study (`artifacts/legality_study/`)

Common to all 19 runs (control center, run through `run_training()`):

- **Task:** goat learner against a tiger mix with $q = 0.5$ (§6.1), default v1.1 knobs, 10,002,432 steps.
- **Rollouts:** $N_{\text{env}} = 8$, $N_{\text{steps}} = 512$, so $D = 4096$; $B = 256$, $N_{\text{ep}} = 5$.
- **Hyperparameters:** $\alpha = 3\times10^{-4}$, $\gamma = 0.99$, $\lambda = 0.95$, $\epsilon = 0.2$, $c_{\text{ent}} = 0.02$,
  $c_{\text{vf}} = 0.5$, $g_{\max} = 0.5$.
- **Checkpoints:** 10 per run, every 1M steps.
- **Plain PPO only:** $r_{\text{ill}} = -3.2$, $d_{\text{targ}} = 0.01$, $\beta_0 = 1$, $\delta_H = 1$.

| Run | Algorithm | $\mathcal{J}^\pi$ | $\mathcal{J}^V$ | $\mathcal{J}^{\text{legal}}$, $c_{\text{legal}}$ | Illegal move | act | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `maskable` | Maskable PPO | clip (masked) | MSE | — | impossible | tanh | 42, 7 |
| `ppo_clip_nolegal` | plain PPO | clip | MSE | — , 0 | ends game | tanh | 42, 7 |
| `ppo_clip_nll05` | plain PPO | clip | MSE | nll, 0.5 | ends game | tanh | 42, 7 |
| `ppo_clip_nll2` | plain PPO | clip | MSE | nll, 2 | ends game | tanh | 42 |
| `ppo_clip_mass05` | plain PPO | clip | MSE | mass, 0.5 | ends game | tanh | 42 |
| `ppo_klpen_nll05` | plain PPO | KL penalty | MSE | nll, 0.5 | ends game | tanh | 42 |
| `ppo_spo_nll05` | plain PPO | SPO | MSE | nll, 0.5 | ends game | tanh | 42 |
| `ppo_clip_nll05_huber` | plain PPO | clip | Huber | nll, 0.5 | ends game | tanh | 42 |
| `relu_maskable` | Maskable PPO | clip (masked) | MSE | — | impossible | ReLU | 42 |
| `relu_ppo_clip_nolegal` | plain PPO | clip | MSE | — , 0 | ends game | ReLU | 42 |
| `relu_ppo_clip_nll05` | plain PPO | clip | MSE | nll, 0.5 | ends game | ReLU | 42 |
| `relu_ppo_clip_nll2` | plain PPO | clip | MSE | nll, 2 | ends game | ReLU | 42 |
| `relu_ppo_klpen_nll05` | plain PPO | KL penalty | MSE | nll, 0.5 | ends game | ReLU | 42 |
| `relu_ppo_spo_nll05` | plain PPO | SPO | MSE | nll, 0.5 | ends game | ReLU | 42 |
| `relu_ppo_clip_nolegal_loseturn` | plain PPO | clip | MSE | — , 0 | loses a turn, $r_{\text{ill}} = -1$ | ReLU | 42 |
| `relu_ppo_clip_nll2_loseturn` | plain PPO | clip | MSE | nll, 2 | loses a turn, $r_{\text{ill}} = -1$ | ReLU | 42 |

The tanh runs were trained with the control center before main was merged in (v1.1), the ReLU runs with
v1.2. With tanh, both versions produce bit-identical weights (§8, test 7), so activation is the only difference.

## 7. Evaluation protocol and metrics

**Games.**
- Game $i \in \{0, \dots, G-1\}$ resets every random generator and the environment with seed $s_0 + i$
  ($s_0 = 10{,}000$ in the study, `EVAL_SEED` in the notebooks), so every model faces the same games.
- The learner plays deterministically (§3.2).
- The opponent is a fixed tiger, greedy or smart, with no mixing.
- The study uses $G = 1000$ per cell for final models and $G = 200$ per checkpoint.

**Play modes.**

| Mode | Learner action | Illegal move |
| --- | --- | --- |
| masked | $\arg\max_{a \in \mathcal{L}(s)} z_a$ | impossible |
| raw | $\arg\max_{a \in \mathcal{A}} z_a$ | ends the game (IllegalMove) |
| raw, lose a turn | $\arg\max_{a \in \mathcal{A}} z_a$ | costs a turn (§2.3) |

Raw play is deterministic and an illegal attempt leaves $s$ unchanged. In lose-a-turn mode, a model that
picks an illegal action therefore repeats it until the turn limit.

**Metrics.**

| Metric | Definition |
| --- | --- |
| win rate $W$ | $\#\{\text{Goat wins}\}/G$; standard error $\sqrt{W(1-W)/G} \leq 1.6$ pp at $G = 1000$, $\leq 3.5$ pp at $G = 200$ |
| illegal endings | fraction of raw games ended by an illegal move |
| illegal probability by phase | $\bar P^{\text{ill}}_{\varphi} = \operatorname{mean}\{1 - P_{\mathcal{L}}(s) : s \text{ visited in masked play},\ \phi(s) = \varphi\}$, for $\varphi \in \{0, 1\}$ |
| `train/illegal_prob` (TensorBoard) | $\mathbb{E}_{\mathcal{B}}[1 - P_{\mathcal{L}}(s_t)]$ averaged over a rollout's minibatches: on-policy, training states |
| window win rate (TensorBoard) | goat wins over the last 1,000 training episodes. These use sampled actions and the $q = 0.5$ mix, so they are not comparable to $W$ |

## 8. Verification tests

All tests run the notebook's own code (its cells executed in a headless IPython shell or through `nbclient`).

| # | Test | Result |
| --- | --- | --- |
| 1 | Masked deterministic choice equals `MaskablePPO.predict(…, action_masks)` (2 shipped goats × 2 tigers × 10 games) | 1,981 / 1,981 decisions equal |
| 2 | §5.3 with clip, MSE and $c_{\text{legal}} = 0$ matches SB3 `PPO` (same seed, 256 steps, 3 epochs) | $\max\lvert\Delta\theta\rvert = 0$ |
| 3 | Stored masks match their observations: for placement states, $\mathcal{L}(s) = \{5p : b_p = 0\}$ | 128 / 128 |
| 4 | Every loss variant trains, saves and reloads as plain `PPO` with its `loss_settings` | pass (KL penalty adapted $\beta$ to 0.25) |
| 5 | The legality loss lowers `train/illegal_prob` (20k steps, $c_{\text{legal}} = 0$ vs 0.5) | 0.673 → 0.369 |
| 6 | A tiger learner's illegal move is penalized (v1.1 fix) | $r = -1.00$, reason `invalid_hard` |
| 7 | Control center v1.1 vs v1.2 with tanh, both algorithms, 16,384 steps | $\max\lvert\Delta\theta\rvert = 0$ |
| 8 | End-to-end notebook runs | 13 configurations pass. Three deliberate misconfigurations stop with a clear error: an unknown `ALGORITHM`, a run-level setting placed inside a phase, and resuming with several seeds |

Test 8 covered:
- every feature switch and preset
- both algorithms and all three policy losses
- curriculum phases with model-goat mixing
- resuming
- several seeds per run, with masked and plain PPO in one `TRAINING_RUNS`
- both illegal-move modes
- `ENV_SOURCE = "repo"`
- a fresh clone in a Colab-like empty folder

## 9. The legality study

**Question.** Can a policy trained *without* the mask learn the rules well enough to play legally on its
own? Does that cost or gain playing strength compared with Maskable PPO, and which objective (§5) gets
there? Setup: §6.4. Protocol: §7. Charts: `artifacts/legality_study/charts/`.

### 9.1 Final models, tanh (1,000 games per cell)

$W$ is the goat win rate against each tiger. $\bar P^{\text{ill}}_0, \bar P^{\text{ill}}_1$ are the illegal probabilities in the
placement and movement phases (§7).

| Model | $W$ masked, greedy | $W$ masked, smart | $W$ raw, greedy | $W$ raw, smart | $\bar P^{\text{ill}}_0$ | $\bar P^{\text{ill}}_1$ | Raw games ended illegally |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Maskable PPO, seed 42 | 0.692 | 0.599 | 0 | 0 | 0.778 | 0.876 | 100% |
| Maskable PPO, seed 7 | 0.718 | 0.619 | 0 | 0 | 0.790 | 0.805 | 100% |
| PPO, no legality loss, seed 42 | 0.530 | 0 | 0.525 | 0 | 0.549 | 0.881 | 73.8% |
| PPO, no legality loss, seed 7 | 0.637 | 0 | 0.576 | 0 | 0.448 | 0.870 | 71.2% |
| **PPO + nll 0.5, seed 42** | **0.754** | **0.947** | **0.752** | **0.947** | 0.0010 | 0.029 | 3.6% |
| PPO + nll 0.5, seed 7 | 0.547 | 0.023 | 0.536 | 0.023 | 0.0034 | 0.062 | 23.4% |
| PPO + nll 2, seed 42 | 0.714 | 0.795 | 0.714 | 0.795 | **0.0002** | **0.0015** | **0.1%** |
| PPO + mass 0.5 | 0.006 | 0 | 0.006 | 0 | 0.0007 | 0.504 | 99.6% |
| KL penalty + nll 0.5 | 0.566 | 0.445 | 0.564 | 0.445 | 0.0030 | 0.069 | 7.6% |
| SPO + nll 0.5 | 0.005 | 0 | 0.005 | 0 | 0.0023 | 0.179 | 97.4% |
| clip + nll 0.5 + Huber | 0.628 | 0.336 | 0.627 | 0.336 | 0.0015 | 0.018 | 1.5% |
| *shipped NormalGoat (100M, greedy)* | 0.796 | 0.016 | 0 | 0 | 0.977 | 0.973 | 100% |
| *shipped SmartGoat (100M, smart)* | 0.067 | 0.944 | 0 | 0 | 0.844 | 0.992 | 100% |
| *shipped RobustGoat (60M, curriculum)* | 0.798 | 0.857 | 0 | 0 | 0.984 | 0.990 | 100% |

The raw lose-a-turn win rates equal the raw win rates to three decimals for every model, so the table omits them.

### 9.2 Training dynamics (TensorBoard, last ~40k steps of each run)

| Run | `train/illegal_prob` | Illegal endings (window) | Win rate (window) | Win rate first ≥ 0.10 at |
| --- | ---: | ---: | ---: | ---: |
| Maskable PPO (42 / 7) | — | 0 / 0 | 0.567 / 0.594 | 0.49M / 0.57M |
| no legality loss (42 / 7) | 0.119 / 0.114 | 0.756 / 0.750 | 0.244 / 0.250 | 1.39M / 1.72M |
| nll 0.5 (42 / 7) | 0.004 / 0.020 | 0.084 / 0.440 | 0.798 / 0.152 | 5.00M / 9.26M |
| nll 2 | 0.000 | 0.005 | 0.680 | 3.77M |
| mass 0.5 | 0.049 | 0.999 | 0.001 | never |
| KL penalty + nll 0.5 | 0.022 | 0.489 | 0.163 | 9.34M |
| SPO + nll 0.5 | 0.042 | 0.877 | 0.000 | never |
| clip + nll 0.5 + Huber | 0.006 | 0.119 | 0.324 | 5.98M |

### 9.3 Findings

1. **Masking teaches the network nothing about legality.** After 10M steps, Maskable PPO still puts 78–88% of
   its raw probability on illegal actions. The shipped models put 84–99% there after 60–100M steps. Without
   the mask they lose every game. This is the zero-gradient property of §3.2: no loss term ever touches an
   illegal logit.
2. **The reward alone teaches legality only partly.** Plain PPO with $c_{\text{legal}} = 0$ learns to avoid most
   illegal moves in deterministic play (53–58% raw wins against greedy). But it keeps 11–12% illegal
   probability on its training states, and 71–74% of its raw games still end in an illegal move. It never
   beats the smart tiger.
3. **The `nll` legality loss gives fully legal play and, for seed 42, the strongest goat in the study.** With
   $c_{\text{legal}} = 0.5$ (seed 42), masked and raw win rates agree to 0.002. Of its 3.6% illegal raw endings, at most 2
   in 2,000 games are games it wins with the mask. Against the smart tiger it wins 94.7%, against 59.9% for Maskable PPO at
   the same budget and 85.7% for the 60M-step RobustGoat. With $c_{\text{legal}} = 2$ the network is almost
   perfectly legal ($\bar P^{\text{ill}}_1 = 0.0015$) and still strong (71.4% / 79.5%).
4. **Training under "end_game" has a long plateau, then a sharp take-off.** An illegal move ends the episode,
   so a winning game needs every move to be legal. With per-move illegal probability $p$, the chance of
   playing $n$ moves without one is $(1-p)^n$. At $p \approx 0.05$ and $n \approx 25$ moves, that is about 0.28,
   and the agent almost never reaches the movement phase. The `nll` term pushes $p$ down until play survives
   long enough for winning rewards to appear. The take-off then comes at 3.8M ($c_{\text{legal}} = 2$), 5.0M or
   9.3M (two seeds of 0.5). The onset time is very seed-sensitive, which is why seed 7 of `nll` 0.5 ends far
   behind seed 42.
5. **`mass` fails, as §5.4 predicts.** Its gradient on an illegal logit is $\pi(a \mid s)\,P_{\mathcal{L}}(s)$, which
   is tiny while $P_{\mathcal{L}}$ is tiny. It cleans up the placement phase ($\bar P^{\text{ill}}_0 = 0.0007$) but leaves half
   the movement-phase probability illegal, and 99.6% of games end illegally.
6. **The policy loss matters.** With the same legality term, SPO never took off and KL penalty took off only
   at 9.3M. Huber took off at 6M and finished weaker than plain MSE. These are single seeds, and finding 4
   shows how much the onset time varies between seeds. So these rank orders are suggestive, not established.

**Limitations.**
- Most variants have one seed.
- Evaluation uses the training opponents (greedy and smart).
- The shipped models were trained on an earlier environment version (§6.2).
- Win rate is measured with deterministic play only.

### 9.4 ReLU vs tanh

Same settings and seed as §9.1, with $\operatorname{act} = $ ReLU (§3.1). 1,000 games per cell. Chart:
`artifacts/legality_study/charts/4_relu_vs_tanh.png`.

| Model | act | $W$ masked, greedy | $W$ masked, smart | $W$ raw, greedy | $W$ raw, smart | $\bar P^{\text{ill}}_0$ | $\bar P^{\text{ill}}_1$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Maskable PPO | tanh | 0.692 | 0.599 | 0 | 0 | 0.778 | 0.876 |
| Maskable PPO | ReLU | **0.854** | **0.856** | 0 | 0 | 0.924 | 0.887 |
| PPO, no legality loss | tanh | 0.530 | 0 | 0.525 | 0 | 0.549 | 0.881 |
| PPO, no legality loss | ReLU | 0.008 | 0 | 0.002 | 0 | 0.010 | 0.843 |
| PPO + nll 0.5 | tanh | 0.754 | 0.947 | 0.752 | 0.947 | 0.0010 | 0.029 |
| PPO + nll 0.5 | ReLU | 0.662 | 0.743 | 0.660 | 0.743 | 0.0010 | 0.0084 |
| PPO + nll 2 | tanh | 0.714 | 0.795 | 0.714 | 0.795 | 0.0002 | 0.0015 |
| PPO + nll 2 | ReLU | 0.694 | 0.851 | 0.692 | 0.851 | 0.0001 | 0.0010 |
| KL penalty + nll 0.5 | tanh | 0.566 | 0.445 | 0.564 | 0.445 | 0.0030 | 0.069 |
| KL penalty + nll 0.5 | ReLU | 0.592 | 0 | 0.588 | 0 | 0.0035 | 0.066 |
| SPO + nll 0.5 | tanh | 0.005 | 0 | 0.005 | 0 | 0.0023 | 0.179 |
| SPO + nll 0.5 | ReLU | 0.009 | 0 | 0.009 | 0 | 0.0044 | 0.169 |
| PPO, no legality loss, lose a turn ($r_{\text{ill}} = -1$) | ReLU | 0.186 | 0.906 | 0.185 | 0.906 | 0.019 | 0.070 |
| PPO + nll 2, lose a turn ($r_{\text{ill}} = -1$) | ReLU | 0.011 | 0 | 0.011 | 0 | 0.0003 | 0.0026 |

Findings (one seed per cell, so differences within about the seed spread of §9.1 are not significant):

1. ReLU helps Maskable PPO the most: it becomes the strongest player under the mask (0.854 / 0.856), but it
   still needs the mask ($\bar P^{\text{ill}} \approx 0.9$).
2. With the `nll` loss, the ReLU and tanh results fall inside the seed-to-seed spread of tanh `nll` 0.5
   (0.547–0.754 vs greedy, 0.023–0.947 vs smart). `nll` 2 is the most consistent across activations
   (0.69–0.71 vs greedy, 0.80–0.85 vs smart), with almost no illegal probability.
3. Without a legality loss, ReLU with "end_game" handling never learned to win. With "lose a turn", the same
   learner reached 0.906 against the smart tiger but only 0.186 against greedy. `nll` 2 with "lose a turn"
   failed to win at all, although it plays legally. Under "lose a turn" an illegal attempt does not end the
   game, so the pressure to be legal comes only from the $-1$ reward and the legality loss.

### 9.5 Variance of the invalid-action probability for the best model

Model: PPO + nll 0.5, tanh, seed 42. It played the 2,000 games of §7 (1,000 per tiger) on its own: raw,
argmax, an illegal move ends the game. At every decision we record $p_t = 1 - P_{\mathcal{L}}(s_t)$. Both sides
are summed directly from float64 log-probabilities, so values down to about $10^{-9}$ are exact. Data and
plots are in `artifacts/legality_study/analysis/ppo_clip_nll05_s42/`.

| Decisions | $n$ | mean $\bar p$ | variance $\operatorname{Var}(p)$ | std | median | 95th pct | 99th pct | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 46,810 | $4.23\times10^{-3}$ | $1.14\times10^{-3}$ | 0.034 | $4.0\times10^{-5}$ | $8.3\times10^{-3}$ | 0.094 | 0.995 |
| placement phase | 29,873 | $9.9\times10^{-4}$ | $1.27\times10^{-4}$ | 0.011 | $2.9\times10^{-5}$ | $2.5\times10^{-3}$ | 0.015 | 0.966 |
| movement phase | 16,937 | $9.95\times10^{-3}$ | $2.87\times10^{-3}$ | 0.054 | $7.6\times10^{-5}$ | 0.036 | 0.261 | 0.995 |
| vs greedy | 18,328 | $9.56\times10^{-3}$ | $2.79\times10^{-3}$ | 0.053 | $5.9\times10^{-5}$ | 0.031 | 0.254 | 0.995 |
| vs greedy, movement | 3,455 | 0.044 | 0.0123 | 0.111 | $4.4\times10^{-3}$ | 0.254 | 0.613 | 0.995 |
| vs smart | 28,482 | $8.05\times10^{-4}$ | $4.30\times10^{-5}$ | 0.0066 | $3.8\times10^{-5}$ | $2.2\times10^{-3}$ | 0.012 | 0.641 |

**Decomposition.** By the law of total variance over games $j$:

$$
\operatorname{Var}(p) = \underbrace{\mathbb{E}_j\big[\operatorname{Var}(p \mid j)\big]}_{9.94\times10^{-4}\ (87\%)} + \underbrace{\operatorname{Var}_j\big(\mathbb{E}[p \mid j]\big)}_{1.44\times10^{-4}\ (13\%)}
$$

Most of the spread is between moves *within* a game rather than between games.

**What drives it.** The distribution is heavy-tailed: half the decisions have $p_t < 4\times10^{-5}$, but rare
movement-phase positions against the greedy tiger reach $p_t > 0.25$. The illegal probability rises through the
game against greedy, from about $10^{-5}$ in early placement to a median near $2\times10^{-2}$ late in the movement
phase. Against smart it stays near $10^{-4}$. In deterministic play only 73 of 46,810 choices were illegal.

**If the policy sampled instead of taking the argmax**, the probability that a whole game contains no illegal
move along the same positions is $\prod_t (1 - p_t)$. Its mean is 0.93 (0.88 vs greedy, 0.98 vs smart).
