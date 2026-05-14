# Reward Reference
### `If using vscode use ctrl+shift+V to view this as compiled mardown`  
Current reward structure, tuning knobs, and implementation notes for the live `env_tng_abc.py` environment.

## Scope

This document describes the reward logic currently implemented in:

- `env_tng_abc.py`
- `train_abc.py` for reward-component logging in TensorBoard
- `VAE/LVS_VALUE_SHAPING.py` for optional learned value-delta reward shaping

It is based on the current `DEFAULT_KNOBS` and the live goat/tiger reward paths in code.

## High-Level Summary

There are two reward paths in the environment:

- goat learner reward
- tiger learner reward

Both share the same environment state and many of the same tuning knobs, but they do not use those knobs in the same way.

For the final VAE deliverable, goat training can also wrap the goat reward with an LVS-VAE value-delta term:

```text
reward_total = reward_sparse + LVS_VAE_SHAPING_COEF * (V_LVS(s_next) - V_LVS(s))
```

At a high level:

- goat reward is more heavily shaped around containment, tempo pressure, and anti-repeat behavior
- tiger reward is more linear and simpler, with strong emphasis on captures, mobility, and terminal wins

## Important Implementation Notes

These details are current behavior in code and are easy to miss when tuning:

- Goat positive shaping decays only in move phase, using `GOAT_SHAPING_DECAY_BASE ** move_steps`.
- Goat center-control shaping is only applied in placing phase.
- Goat repeat-state penalties exist; tiger repeat-state penalties do not.
- `LATE_GAME_START_TURN` is currently computed into `info["late"]`, but it is not directly used in the current reward formulas.
- Tiger center shaping currently uses `abs(REWARD_CENTER_GOAT)` as its magnitude, not `REWARD_CENTER_TIGER`.
- Tiger `invalid_hard` reward currently returns `abs(REWARD_INVALID_HARD)`, which is positive with the current default of `-1.0`.
- LVS-VAE shaping is applied in `train_abc.py` by replacing the environment reward function with a wrapper. The base environment rules remain unchanged.

## Default Knobs

### Terminal and Material

| Knob | Default | Meaning |
| --- | ---: | --- |
| `REWARD_GOAT_WIN` | `3.2` | Goat terminal reward when goats immobilize all tigers |
| `REWARD_TIGER_WIN` | `-3.2` | Goat terminal penalty when tigers win |
| `REWARD_TIGER_CAPTURE` | `0.35` | Tiger reward per goat captured |
| `REWARD_TIGER_WIN_BONUS` | `3.2` | Tiger terminal reward for winning |
| `REWARD_TIGER_LOSS_PENALTY` | `3.2` | Tiger terminal penalty for losing |
| `REWARD_GOAT_EATEN` | `-0.35` | Goat penalty per goat captured by tigers |

### Tempo and Timeouts

| Knob | Default | Meaning |
| --- | ---: | --- |
| `REWARD_STEP` | `-0.001` | Base per-step penalty |
| `MAX_TIMEOUT_SCALE` | `1.5` | Timeout loss multiplier |
| `REPEAT_STALL_SCALE` | `1.25` | Repeat-timeout loss multiplier |
| `MAX_TURNS` | `100` | Hard episode turn limit |
| `MOVE_STEP_BASE` | `-0.01` | Goat move-phase step penalty base |
| `MOVE_STEP_SLOPE` | `-0.01` | Goat move-phase step penalty ramp |
| `MOVE_STEP_MIN` | `-0.6` | Goat move-phase step penalty floor |
| `GOAT_WIN_TURN_DECAY` | `0.01` | Goat win reward decreases slightly with turn count |
| `LATE_GAME_START_TURN` | `40` | Start turn for the `late` info value |

### Goat Tactical and Positional Shaping

| Knob | Default | Meaning |
| --- | ---: | --- |
| `REWARD_BLOCK_TIGER` | `0.08` | Mobility-shaping magnitude |
| `REWARD_BUBBLE_SPACE` | `0.02` | Reward for increasing safe unreachable space |
| `REWARD_CLUSTER_TIGERS` | `0.02` | Reward for making tigers less spread out |
| `REWARD_CENTER_GOAT` | `0.03` | Goat center-control term |
| `REWARD_CENTER_TIGER` | `-0.03` | Goat-facing penalty when tigers hold key centers |
| `NEAR_LOCK_BONUS` | `0.05` | Bonus when tigers are close to immobilized |
| `MOBILITY_BACKSLIDE_SCALE` | `0.5` | Penalty scale when goat move increases tiger mobility |
| `GOAT_SHAPING_DECAY_BASE` | `0.95` | Exponential decay base for positive goat shaping in move phase |

### Anti-Repeat, Invalid Actions, and Opponent Behavior

| Knob | Default | Meaning |
| --- | ---: | --- |
| `REWARD_INVALID_SOFT` | `-0.05` | Soft invalid-action penalty |
| `REWARD_INVALID_HARD` | `-1.0` | Hard invalid-action value |
| `REWARD_REPEAT_STATE` | `-0.5` | Goat repeat-state penalty coefficient |
| `MAX_REPEATS` | `4` | Repeat count before stall timeout triggers |
| `BASE_TIGER_CAPTURE_BIAS` | `1.0` | Tiger opponent behavior knob, not direct reward |

## Goat Learner Reward

The goat reward path is implemented in `_sparse_reward_goat(...)`.

### Goat Reward Formula Structure

For a normal non-terminal step, the goat reward is the sum of:

```text
step
+ near_lock
+ block
+ bubble
+ cluster
+ center
+ goat_eaten
+ repeat
+ terminal
```

On some immediate terminal or invalid cases, the code returns early and skips the rest of the shaping terms.

### Goat Reward Families

| Family | When it applies | Current formula |
| --- | --- | --- |
| Invalid soft | illegal but non-fatal action | `REWARD_INVALID_SOFT` |
| Invalid hard | fatal invalid action | `REWARD_INVALID_HARD` |
| Goat win | no tiger moves remain | `REWARD_GOAT_WIN - GOAT_WIN_TURN_DECAY * turns` |
| Base step | every goat step | placing: `REWARD_STEP`; moving: `max(MOVE_STEP_BASE + MOVE_STEP_SLOPE * move_steps, MOVE_STEP_MIN)` |
| Near lock | goat leaves tiger with very low mobility | `NEAR_LOCK_BONUS * decay_mult` |
| Positive block | goat reduces tiger move count | `abs(REWARD_BLOCK_TIGER) * tanh(delta_moves) * decay_mult` |
| Mobility backslide | goat increases tiger move count | `abs(REWARD_BLOCK_TIGER) * MOBILITY_BACKSLIDE_SCALE * delta_moves` where `delta_moves < 0` |
| Bubble | goat increases unreachable safe cells | `REWARD_BUBBLE_SPACE * tanh(delta_bubble) * decay_mult` when `delta_bubble > 0` |
| Cluster | goat reduces tiger spread | `REWARD_CLUSTER_TIGERS * tanh(delta_spread) * decay_mult` when `delta_spread > 0` |
| Center control | placing phase only | `center_score` from key-center occupancy |
| Goat eaten | tiger captured goats this turn | `REWARD_GOAT_EATEN * goats_eaten_this_turn` |
| Tiger win | tiger reaches capture threshold | `REWARD_TIGER_WIN` |
| Max timeout | turn limit reached | `REWARD_TIGER_WIN * MAX_TIMEOUT_SCALE` |
| Repeat timeout | repeat-state stall timeout | `REWARD_TIGER_WIN * REPEAT_STALL_SCALE` |
| Repeat penalty | repeated non-terminal state | `REWARD_REPEAT_STATE * (repeat_prev_count ** 2)` |

### Goat Reward Behavior Notes

- Positive goat shaping decays only in move phase.
- Goat center shaping is intentionally opening-only.
- On terminal loss reasons, positive shaping terms are skipped.
- Repeat penalties ramp quadratically using the previous repeat count.
- Goat material penalty is per captured goat, so multiple captures in a turn stack.

### Goat Shaping Inputs From `info`

The goat reward path consumes these step-summary fields:

- `reason`
- `step_penalty`
- `near_lock`
- `delta_moves`
- `delta_bubble`
- `delta_spread`
- `center_score`
- `goats_eaten_this_turn`
- `repeat_prev_count`
- `phase`
- `move_steps`

## Tiger Learner Reward

The tiger reward path is implemented in `_sparse_reward_tiger(...)`.

### Tiger Reward Formula Structure

For standard steps, tiger reward starts from the step penalty and then adds shaping:

```text
step
+ capture
+ mobility
+ center
+ terminal bonus_or_penalty
```

### Tiger Reward Families

| Family | When it applies | Current formula |
| --- | --- | --- |
| Invalid soft | illegal but non-fatal action | `REWARD_INVALID_SOFT` |
| Invalid hard | fatal invalid action | `abs(REWARD_INVALID_HARD)` |
| Base step | every tiger step | `REWARD_STEP` |
| Capture reward | tiger captures goats | `REWARD_TIGER_CAPTURE * goats_eaten_this_turn` |
| Mobility shaping | tiger increases or decreases own mobility | `abs(REWARD_BLOCK_TIGER) * delta_moves` |
| Center control | placing phase only | `center_score` |
| Tiger win | capture-threshold win or no-goat-moves win | `+REWARD_TIGER_WIN_BONUS` |
| Goat win | goats immobilize tigers | `-REWARD_TIGER_LOSS_PENALTY` |
| Max timeout | turn limit reached | `-REWARD_TIGER_LOSS_PENALTY * MAX_TIMEOUT_SCALE` |

### Tiger Reward Behavior Notes

- Tiger mobility shaping is linear, not saturated.
- Tiger reward currently has no dedicated repeat-state penalty path.
- Tiger reward currently has no dedicated repeat-timeout terminal branch.
- Tiger center shaping only applies in placing phase.
- Tiger center shaping currently uses `abs(REWARD_CENTER_GOAT)` in the environment helper that builds `center_score`.

### Tiger Shaping Inputs From `info`

The tiger reward path consumes these step-summary fields:

- `reason`
- `step_penalty`
- `goats_eaten_this_turn`
- `delta_moves`
- `center_score`
- `phase`

## Center Control Terms

Center shaping is based on the environment's `KEY_CENTERS`.

Current behavior differs by learner:

- goat reward uses both `REWARD_CENTER_GOAT` and `REWARD_CENTER_TIGER`
- tiger reward uses a separate `center_score` helper that currently adds or subtracts `abs(REWARD_CENTER_GOAT)`

That means:

- changing `REWARD_CENTER_TIGER` affects goat learner reward
- changing `REWARD_CENTER_TIGER` does not currently affect tiger learner center shaping

## Repeat-State Logic

Repeat tracking currently matters only on the goat-learner path.

## LVS-VAE Reward Shaping

The learned reward path is optional and goat-facing. It is implemented outside the core environment in `VAE/LVS_VALUE_SHAPING.py`.

The wrapper first calls the normal environment reward:

```text
reward_sparse = sparse_reward(prev_obs, action, obs, terminated, truncated, info)
```

It then predicts the previous and next state values using a frozen full/endgame LVS-VAE ensemble:

```text
value_delta = V_LVS(obs) - V_LVS(prev_obs)
reward_vae_component = LVS_VAE_SHAPING_COEF * value_delta
reward_total = reward_sparse + reward_vae_component
```

Logged fields:

| Field | Meaning |
| --- | --- |
| `reward_sparse_component` | Base handcrafted/terminal reward |
| `reward_vae_component` | Learned value-delta reward |
| `reward_total` | Final reward returned to PPO |
| `lvs_value_before` | VAE value before the action |
| `lvs_value_after` | VAE value after the action |
| `lvs_value_delta` | Difference between after and before values |
| `lvs_progress_ratio` | `turn_counter / MAX_TURNS` |
| `lvs_gate_active` | Whether the endgame blend is active |

Current goat behavior:

- repeated non-terminal states incur `REWARD_REPEAT_STATE * repeat_prev_count^2`
- once the repeat count reaches `MAX_REPEATS`, the episode ends with `reason = "repeat_timeout"`
- repeat timeout applies the scaled terminal loss `REWARD_TIGER_WIN * REPEAT_STALL_SCALE`

Tiger learner behavior:

- no repeat-state shaping
- no repeat-timeout-specific reward branch

## Invalid Action Handling

Action masking should prevent almost all invalid actions during normal training and evaluation, but the reward code still has explicit fallback behavior.

Current behavior:

| Case | Goat learner | Tiger learner |
| --- | ---: | ---: |
| `invalid_soft` | `REWARD_INVALID_SOFT` | `REWARD_INVALID_SOFT` |
| `invalid_hard` | `REWARD_INVALID_HARD` | `abs(REWARD_INVALID_HARD)` |

With the current defaults, that means:

- goat `invalid_hard` returns `-1.0`
- tiger `invalid_hard` returns `+1.0`

## Reward Components Logged to TensorBoard

`train_abc.py` logs these goat-side reward components when present in `info`:

- `reward_step_component`
- `reward_near_lock_component`
- `reward_block_component`
- `reward_bubble_component`
- `reward_cluster_component`
- `reward_center_component`
- `reward_goat_eaten_component`
- `reward_repeat_component`
- `reward_terminal_component`
- `reward_decay_mult`
- `reward_total`

These are useful for debugging whether the agent is actually learning to win or just exploiting one shaping family.

## Reading PPO Metrics at a High Level

The PPO optimizer plots are not direct task-success metrics. For this project, read the metrics in this order:

1. outcome metrics such as win rates and timeout rates
2. episode reward and episode length
3. PPO optimizer metrics such as `approx_kl`, `clip_fraction`, `policy_gradient_loss`, `value_loss`, and `explained_variance`

Good practical rules:

- If win rate is improving and timeout rate is dropping, some optimizer instability can be acceptable.
- If reward rises but wins do not, the policy may be farming shaping.
- If `value_loss` is noisy and outcome metrics are bad, the reward structure may be hard to fit or easy to exploit.

## Practical Tuning Guidance

When tuning goat reward, the highest-impact families are usually:

- terminal outcomes
- move-phase tempo pressure
- mobility containment
- repeat penalties

When tuning tiger reward, the highest-impact families are usually:

- terminal outcomes
- capture reward
- mobility shaping

Safe first ablations:

- set `REWARD_BUBBLE_SPACE = 0.0`
- set `REWARD_CLUSTER_TIGERS = 0.0`
- reduce `REWARD_BLOCK_TIGER`
- reduce `REWARD_REPEAT_STATE` in magnitude

## Short Takeaways

- Goat reward is more structured and constrained than tiger reward.
- Tiger reward is simpler and more linear, especially around mobility.
- Repeat-state punishment is currently a goat-side mechanism.
- Center-control tuning is asymmetric between goat and tiger paths.
- `LATE_GAME_START_TURN` is currently informational, not directly reward-active.
