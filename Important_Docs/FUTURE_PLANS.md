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
