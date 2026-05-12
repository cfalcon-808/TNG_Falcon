# Future Plans

## Beginning LVS-VAE

Goal: add an early-training VAE signal that rewards good opening structure without punishing weak beginner moves too aggressively.

Train a separate beginning-only LVS-VAE on early episode states.

- Source data: existing full-game VAE datasets
- Filter: `progress_ratio <= 0.30`
- Optional cleanup: exclude `move_index == 0`
- Target: binary opening survival

```text
good_opening = goat_win or timeout -> 1.0
bad_opening  = tiger_win           -> 0.0
```

Use the beginning VAE only in the opening phase.

```python
if progress_ratio <= 0.30:
    reward += early_coef * max(begin_value_after - begin_value_before, 0.0)
```

After the opening phase, hand off to the existing full/endgame LVS-VAE.

```text
opening: survival VAE
middle:  full-game VAE
late:    full/endgame blended VAE
```

Rationale: timeouts often share early structure with goat-favorable games. For early training, teaching "do not collapse" is simpler than teaching decisive wins immediately. The existing full/endgame VAE can later push survival states toward winning states.

## VAE Top-K Action Reranking

Goal: test whether a frozen PPO goat model generalizes better when a VAE critic helps choose among its strongest legal actions.

At inference time:

1. Get PPO action probabilities.
2. Keep the top-k legal actions.
3. Simulate each candidate action one step.
4. Score each next state with the LVS-VAE.
5. Choose the action with the best blended score.

```python
score = ppo_logprob[action] + alpha * vae_value_after
```

Start conservative:

```text
k = 3 or 5
alpha = small
goat decisions only
```

Generalization test:

```text
same frozen PPO model
baseline action selection vs VAE-reranked action selection
evaluate vs greedy tiger, smart tiger, and mixed tiger
```

Success signal: similar greedy performance, better smart/mixed performance, and fewer fast tiger wins.

## State-Action Opening Head

Goal: learn an early-game action-quality signal that can eventually replace or reduce hand-shaped center, bubble, cluster, and blocking rewards.

Start with a state-action model:

```text
input = placing_phase_state + goat_action
target = good opening action score
```

Initial labels from existing trajectory outcomes:

```text
goat_win or timeout trajectory -> 1.0
tiger_win trajectory           -> 0.0
```

Use only during goat placing phase:

```python
if phase_state == 0:
    reward += action_coef * action_quality(prev_obs, action)
```

Why state-action first:

```text
state-action asks: "is this a good move from this board?"
state-to-state asks: "did the board get better after the move?"
```

For opening placement, the chosen action is important because blocking, center control, and bubble space depend on the move made from the current board.

Later upgrade:

```text
input = prev_state + action + next_state
target = transition quality or structural shaping delta
```

Add negative samples from alternate legal actions when possible, so the model learns move quality instead of only memorizing outcome-correlated trajectories.

## State-Action-State Transition Quality Head

Goal: learn whether a goat move was useful from the local transition itself, instead of judging every move only by the final episode outcome.

Motivation:

```text
good move + later policy mistakes -> possible loss
bad move  + opponent mistakes     -> possible win
```

The model should learn good tactical building blocks even when the sampled trajectory eventually loses.

Dataset rows should store transition-level examples:

```text
prev_state
goat_action
post_goat_state
post_tiger_response_state
final_outcome_label
tiger_moves_before
tiger_moves_after
delta_tiger_moves
near_lock_after
lock_after
delta_bubble
delta_tiger_spread
goats_eaten_this_turn
repeat_or_stall_flag
```

Train separate heads rather than one blended target only:

```text
state_value_head        -> final goat outcome / survival probability
transition_quality_head -> immediate tactical quality of the move
rank_quality_head       -> quality compared with other legal moves from same state
combined_quality_head   -> tactical quality with a small outcome prior
```

Candidate tactical label:

```text
tactical_quality =
    + tiger_mobility_reduced
    + near_lock_created
    + lock_created
    + bubble_space_increased
    + tiger_spread_reduced
    - goat_capture_allowed
    - tiger_mobility_increased
    - repeat_or_stall_risk
```

Candidate combined label:

```text
combined_quality =
    0.65 * tactical_quality
  + 0.25 * next_state_value
  + 0.10 * final_outcome_label
```

Normalize labels to `[0, 1]` before training. Keep the unblended heads so analysis can show whether the model is learning local move quality or merely rediscovering outcome correlation.

Best upgrade: enumerate all legal goat actions from the same `prev_state`, simulate each transition, and train a ranking target.

```text
rank_quality = percentile_rank(tactical_score among legal moves from prev_state)
```

This asks the more useful question:

```text
Was this move good compared with the other legal moves available right now?
```

Reward-shaping use:

```python
reward += value_coef * (state_value_after - state_value_before)
reward += transition_coef * transition_quality(prev_state, action, next_state)
reward += rank_coef * rank_quality(prev_state, action, next_state)
```

Start conservative by training the head offline and evaluating whether high-ranked moves reduce fast tiger wins and increase near-lock/lock creation before adding it to PPO reward.

## Latent Feature Heads

Goal: use the VAE latent representation as a shared feature extractor for learned reward-shaping signals.

Architecture:

```text
state -> VAE encoder -> latent mu
                    -> reconstruction decoder
                    -> outcome value head
                    -> feature heads
```

Potential feature heads:

```text
center_control_head
bubble_space_head
tiger_mobility_head
near_lock_head
capture_risk_head
opening_survival_head
```

Use predicted feature deltas for shaping:

```python
reward += center_coef * (center_after - center_before)
reward += bubble_coef * (bubble_after - bubble_before)
reward -= risk_coef * (risk_after - risk_before)
```

Best first version:

```text
latent mu -> opening_survival_head
```

Later versions can train structure heads from recomputed board features or saved env reward components. Avoid treating individual latent dimensions as directly meaningful; train heads on top of the latent vector instead.
