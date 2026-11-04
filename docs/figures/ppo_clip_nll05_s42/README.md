# Plots: PPO + nll 0.5, seed 42 (best model that needs no mask)

**Model.** Plain PPO, no action masking.
- Losses: clipped surrogate, MSE value loss, and the `nll` legality loss at weight 0.5.
- Network and seed: tanh MLP 256×256×256, seed 42.
- Training: 10M steps as the goat, against a 50/50 greedy/smart tiger mix.
- Illegal moves: an illegal move ends the game with reward −3.2.

The full specification and the math are in
[`docs/TRAINING_MATHEMATICS.md`](../../TRAINING_MATHEMATICS.md) (§5.3–5.4, §6.4, §9). The model file and
its reproduction notes stay local in `artifacts/legality_study/` (ignored by git).

**Results** (1,000 games per cell, deterministic play): win rate 0.754 vs greedy and 0.947 vs smart with
the mask, 0.752 / 0.947 without it.

| File | What it shows |
| --- | --- |
| [`1_training_curves.png`](1_training_curves.png) | TensorBoard training curves (illegal-action probability, illegal endings, win rate) against Maskable PPO, PPO without the legality loss, and PPO + nll 2 |
| [`2_checkpoint_eval.png`](2_checkpoint_eval.png) | Win rate of every 1M-step checkpoint against both tigers, with the mask, raw, and raw with lose-a-turn |
| [`3_loss_variants.png`](3_loss_variants.png) | The same legality loss under the KL-penalty, SPO and Huber variants |
| [`4_relu_vs_tanh.png`](4_relu_vs_tanh.png) | Final models with ReLU vs tanh, for every method |
| [`5_valid_probability_by_move.png`](5_valid_probability_by_move.png) | Probability of a valid action at each move over 2,000 games: median, mean, 25–75% and 5–95% bands, 40 sample games; log-scale invalid probability below |
| [`6_invalid_probability_variance.png`](6_invalid_probability_variance.png) | Per-move variance across games, the distribution over all 46,810 decisions, and the per-game chance of an all-valid game if moves were sampled |
| `variance_summary.json`, `stats_by_mode_opponent_phase.csv` | The numbers behind plots 5–6 |

**Invalid-action probability** $p_t = 1 - P_{\mathcal{L}}(s_t)$, over 2,000 games playing without the mask:

| Moves included | Moves | Mean | Variance |
| --- | ---: | ---: | ---: |
| all | 46,810 | 0.0042 | 0.00114 |
| movement phase | 16,937 | 0.0100 | 0.00287 |
| vs greedy, movement phase | 3,455 | 0.044 | 0.0123 |
| vs smart | 28,482 | 0.0008 | 0.000043 |

- 87% of the variance is within games and 13% between games.
- The median $p_t$ is 4×10⁻⁵.
