# LVS-VAE Reward Shaping Guide

This guide describes the final value-head-only LVS-VAE reward shaping path.

## Purpose

The LVS-VAE supplies an auxiliary goat-favorability signal during PPO training. It does not change the Tigers and Goats rules. It wraps the existing goat reward and adds a small value-delta term:

```text
R_total = R_sparse + alpha * (V_LVS(s_next) - V_LVS(s))
```

The current implementation uses frozen full-game and endgame LVS-VAE checkpoints. The VAE is not trained online during PPO.

## State Input

The model receives the same 25-feature state shape used by the environment:

```text
23 board cells + goats_eaten_state + phase_state
```

Board cells use:

```text
0 = empty
1 = goat
2 = tiger
```

The value label is outcome-based:

```text
goat win  = 1.0
tie       = 0.5
tiger win = 0.0
```

## Active Files

| File | Purpose |
| --- | --- |
| `VAE/LVS_VAE.py` | VAE architecture, value head, and loss |
| `VAE/LVS_VALUE_SHAPING.py` | Frozen value predictor and reward wrapper |
| `train_abc.py` | Checkpoint paths, shaping coefficient, smoke check, PPO integration |

## Phase-Gated Ensemble

The reward wrapper predicts a full-game value and, after the configured progress threshold, blends in endgame value:

```text
progress = turn_counter / MAX_TURNS
```

Before threshold:

```text
V_LVS(s) = mean(full-game models)
```

After threshold:

```text
V_LVS(s) = (1 - endgame_weight) * mean(full-game models)
         + endgame_weight * mean(endgame models)
```

The checked-in training configuration uses full-game checkpoints from `full_40k_lvsvae_v1` and `full_20k_lvsvae_v1`, plus endgame checkpoints from `end_40k_lvsvae_v1`, `end_20k_lvsvae_v1`, and `end06_20k_lvsvae_v1`.

## Main Configs

These live near the top of `train_abc.py`:

```python
USE_LVS_VAE_SHAPING = True
RUN_LVS_VAE_SMOKE_CHECK = True
RUN_LVS_VAE_SMOKE_CHECK_ONLY = False
LVS_VAE_SHAPING_COEF = 0.5
LVS_VAE_PROGRESS_THRESHOLD = 0.7
LVS_VAE_ENDGAME_BLEND = 0.7
LVS_VAE_PROGRESS_MODE = "turn_max_ratio"
LVS_VAE_DEVICE = "cpu"
```

`LVS_VAE_DEVICE` is kept on CPU so each `SubprocVecEnv` worker can load the frozen models safely.

## Reward Logging

When shaping is enabled, the wrapper writes these fields into `info` and TensorBoard logging:

| Field | Meaning |
| --- | --- |
| `reward_sparse_component` | Reward before LVS-VAE shaping |
| `reward_vae_component` | `alpha * value_delta` |
| `reward_total` | Sparse reward plus VAE reward |
| `lvs_value_before` | Predicted value for previous state |
| `lvs_value_after` | Predicted value for next state |
| `lvs_value_delta` | `value_after - value_before` |
| `lvs_progress_ratio` | Current progress ratio |
| `lvs_gate_active` | `1.0` when endgame blending is active |

## Smoke Check

To test checkpoint loading and one shaped transition without training:

```python
RUN_LVS_VAE_SMOKE_CHECK = True
RUN_LVS_VAE_SMOKE_CHECK_ONLY = True
```

Then run:

```powershell
python train_abc.py
```

## Interpretation

The LVS-VAE value signal is a supplemental learned reward, not a replacement for handcrafted rewards. The current results show that it can be integrated and logged during PPO training. Stronger claims require repeated seeds, more ablations, and cross-opponent evaluation.
