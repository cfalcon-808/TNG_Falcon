# LVS-VAE Reward Shaping Guide
### `If using vscode use ctrl+shift+V to view this as compiled markdown`

Version context: `train_abc.py` (`mppo_train3.1`) + `env_tng_abc.py` (`env6.0`) + `VAE/LVS_VALUE_SHAPING.py`

This document explains the new LVS-VAE reward shaping functionality in the Tigers and Goats PPO training script.

## TLDR

The LVS-VAE adds an optional goat-favorability reward hint during PPO training. It wraps the existing goat reward in `train_abc.py`, predicts whether the new state looks better or worse for goats, and adds a small bonus or penalty using `0.02 * (value_after - value_before)`. It is opt-in through `use_lvs_vae_shaping=True`, leaves `env_tng_abc.py` rules unchanged, and uses a phase-gated ensemble that blends in endgame VAE models after `turn_counter / MAX_TURNS >= 0.7`.

## 1. Purpose

The LVS-VAE system adds an optional learned value signal to goat PPO training.

The VAE was trained offline to predict goat-favorability from a 25-column game state:

```text
[cell_00 ... cell_22, goats_eaten_state, phase_state]
```

The PPO reward shaping integration uses that predicted value as a potential-style delta:

```python
value_delta = value_after - value_before
reward_total = sparse_reward + LVS_VAE_SHAPING_COEF * value_delta
```

This means the goat learner gets a small bonus when a move shifts the state toward a more goat-favorable predicted value, and a small penalty when the value decreases.

## 2. Where It Lives

The VAE is integrated through `train_abc.py`, not by modifying the core environment rules in `env_tng_abc.py`.

The environment already supports reward injection through `reward_fn`:

```python
self.reward_fn = reward_fn or self.sparse_reward
```

During training, `train_abc.py` optionally replaces the environment reward function with a wrapper:

```python
base_env.reward_fn = make_lvs_vae_reward_fn(
    base_env.sparse_reward,
    build_lvs_vae_shaping_config(knobs),
)
```

This keeps the environment reusable:

- normal training still uses the original reward
- VAE shaping is opt-in
- tiger learner behavior is unchanged
- the same environment code can be used for clean comparison runs

## 3. Files Added or Changed

### `VAE/LVS_VALUE_SHAPING.py`

This file contains the frozen inference helper.

Important public pieces:

| Name | Purpose |
| --- | --- |
| `LVSValueShapingConfig` | Dataclass holding checkpoint paths and shaping settings. |
| `PhaseGatedLVSValuePredictor` | Lazy-loads VAE checkpoints and predicts phase-gated goat value. |
| `make_lvs_vae_value_predictor(...)` | Creates a reusable value predictor. |
| `smoke_check_lvs_vae_value_predictor(...)` | Loads the ensemble and checks predicted values are in `[0, 1]`. |
| `make_lvs_vae_reward_fn(...)` | Wraps the existing reward function with LVS-VAE value-delta shaping. |

### `train_abc.py`

This file owns training-time configuration and decides whether shaping is enabled.

Important additions:

- LVS-VAE user configs
- checkpoint path configs
- per-variation `use_lvs_vae_shaping`
- named shaped variation config
- smoke check path
- TensorBoard logging fields
- run metadata fields

## 4. The Phase-Gated Ensemble

The current shaping value is not one model. It is an ensemble.

Before the live progress threshold:

```python
value = mean(full_models)
```

After the live progress threshold:

```python
value = 0.3 * mean(full_models) + 0.7 * mean(end_models)
```

The current checkpoint groups are:

```python
LVS_VAE_FULL_CHECKPOINT_PATHS = [
    full_40k_lvsvae_v1_final_model.pt,
    full_20k_lvsvae_v1_final_model.pt,
]

LVS_VAE_END_CHECKPOINT_PATHS = [
    end_40k_lvsvae_v1_final_model.pt,
    end_20k_lvsvae_v1_final_model.pt,
    end06_20k_lvsvae_v1_final_model.pt,
]
```

The live gate uses max-turn progress:

```python
progress_ratio = turn_counter / MAX_TURNS
```

With the current default:

```python
LVS_VAE_PROGRESS_THRESHOLD = 0.7
```

So the endgame ensemble starts blending in when the current episode reaches 70% of the configured max turns.

## 5. User Configs

These configs live near the top of `train_abc.py`.

### Main enable flags

```python
USE_LVS_VAE_SHAPING = False
RUN_LVS_VAE_SMOKE_CHECK = False
RUN_LVS_VAE_SMOKE_CHECK_ONLY = False
```

What they mean:

| Config | Meaning |
| --- | --- |
| `USE_LVS_VAE_SHAPING` | Global default for shaping. Keep this `False` unless you want every default goat phase to inherit shaping. |
| `RUN_LVS_VAE_SMOKE_CHECK` | Runs the smoke check before normal training continues. |
| `RUN_LVS_VAE_SMOKE_CHECK_ONLY` | Runs the smoke check, prints results, then exits before PPO starts. |

Recommended use:

```python
RUN_LVS_VAE_SMOKE_CHECK_ONLY = True
```

Run once to verify checkpoint loading and reward wrapper behavior. Then set it back to `False`.

### Shaping strength and gate

```python
LVS_VAE_SHAPING_COEF = 0.02
LVS_VAE_PROGRESS_THRESHOLD = 0.7
LVS_VAE_ENDGAME_BLEND = 0.7
LVS_VAE_PROGRESS_MODE = "turn_max_ratio"
LVS_VAE_DEVICE = "cpu"
```

What they mean:

| Config | Meaning |
| --- | --- |
| `LVS_VAE_SHAPING_COEF` | Multiplier on `value_after - value_before`. Higher means the VAE has more influence on PPO reward. |
| `LVS_VAE_PROGRESS_THRESHOLD` | Progress ratio where the phase-gated blend starts using endgame models. |
| `LVS_VAE_ENDGAME_BLEND` | Weight on the endgame ensemble after threshold. `0.7` means 70% endgame, 30% full-game. |
| `LVS_VAE_PROGRESS_MODE` | Currently expected to be `"turn_max_ratio"`. |
| `LVS_VAE_DEVICE` | VAE inference device. Keep as `"cpu"` for `SubprocVecEnv` safety. |

Recommended first experiments:

- keep `LVS_VAE_SHAPING_COEF = 0.02`
- keep `LVS_VAE_PROGRESS_THRESHOLD = 0.7`
- keep `LVS_VAE_ENDGAME_BLEND = 0.7`
- keep `LVS_VAE_DEVICE = "cpu"`

## 6. How To Enable Shaping For Training

The active `VARIATIONS` block is still unchanged by default.

There is a separate shaped config:

```python
LVS_VAE_SHAPED_VARIATIONS = {
    "goat_vs_greedy_lvs_vae_shaping": {
        "timesteps": TIMESTEPS,
        "opponent_ai": OPP_TIGER_GREEDY,
        "mix_prob": None,
        "use_lvs_vae_shaping": True,
    },
}
```

To run only the shaped training variation, set:

```python
VARIATIONS = LVS_VAE_SHAPED_VARIATIONS
```

This is the safest way to use the feature because it makes shaping explicit in the run config.

You can also enable shaping inside any individual variation:

```python
VARIATIONS = {
    "my_shaped_goat_run": {
        "timesteps": 2_000_000,
        "opponent_ai": OPP_TIGER_GREEDY,
        "mix_prob": None,
        "use_lvs_vae_shaping": True,
    },
}
```

For comparison, make a matching unshaped run:

```python
VARIATIONS = {
    "baseline_no_vae": {
        "timesteps": 2_000_000,
        "opponent_ai": OPP_TIGER_GREEDY,
        "mix_prob": None,
        "use_lvs_vae_shaping": False,
    },
    "with_lvs_vae": {
        "timesteps": 2_000_000,
        "opponent_ai": OPP_TIGER_GREEDY,
        "mix_prob": None,
        "use_lvs_vae_shaping": True,
    },
}
```

## 7. Reward Data Flow

For each goat learner step:

1. The environment produces `prev_obs`, `obs`, and `info`.
2. The original goat reward is calculated by `base_env.sparse_reward(...)`.
3. The VAE ensemble predicts goat-favorability for `prev_obs`.
4. The VAE ensemble predicts goat-favorability for `obs`.
5. The wrapper calculates:

```python
value_delta = value_after - value_before
vae_component = LVS_VAE_SHAPING_COEF * value_delta
reward_total = sparse_reward + vae_component
```

6. The wrapper writes reward/value diagnostics into `info`.
7. PPO receives `reward_total`.

The VAE is not changing legal actions, terminal rules, board transitions, or scripted tiger behavior. It only changes the numeric reward returned to PPO when the shaped wrapper is enabled.

## 8. TensorBoard Fields

The callback tracks these reward/value fields when they appear in `info`:

| Field | Meaning |
| --- | --- |
| `reward_sparse_component` | Original reward before VAE shaping. |
| `reward_vae_component` | Added VAE shaping reward. |
| `reward_total` | Final reward returned to PPO. |
| `lvs_value_before` | VAE goat-favorability before the move. |
| `lvs_value_after` | VAE goat-favorability after the move. |
| `lvs_value_delta` | `lvs_value_after - lvs_value_before`. |
| `lvs_progress_ratio` | Live progress ratio, currently `turn_counter / MAX_TURNS`. |
| `lvs_gate_active` | `1.0` when the endgame blend is active, otherwise `0.0`. |

In TensorBoard, check:

- `4.reward_components (per 1000 steps)/11.vae_mean`
- `4.reward_components (per 1000 steps)/13.total_reward_mean`
- `4.reward_components (per 1000 steps)/16.lvs_value_delta_mean`
- `4.reward_components (per 1000 steps)/18.lvs_gate_active_frac`

If `reward_vae_component` is much larger than the other reward components, lower `LVS_VAE_SHAPING_COEF`.

## 9. Smoke Check

Before running PPO, you can verify the integration without training.

Set:

```python
RUN_LVS_VAE_SMOKE_CHECK_ONLY = True
```

Then run:

```powershell
python train_abc.py
```

The smoke check:

- loads all configured VAE checkpoints
- checks early and late predicted values are in `[0, 1]`
- creates one goat learner environment
- wraps the reward function
- takes one valid action
- verifies the reward info fields exist
- verifies `lvs_progress_ratio == turn_counter / MAX_TURNS`
- exits before PPO starts

Expected output includes:

```text
[LVS-VAE smoke check] passed
```

After the smoke check passes, set:

```python
RUN_LVS_VAE_SMOKE_CHECK_ONLY = False
```

## 10. Important Tuning Notes

### `LVS_VAE_SHAPING_COEF`

This is the most important tuning knob.

Current default:

```python
LVS_VAE_SHAPING_COEF = 0.02
```

Interpretation:

- `0.0` disables VAE reward effect even if the wrapper is active
- smaller values make the VAE a gentle hint
- larger values make PPO optimize the VAE signal more aggressively

Recommended first sweep:

```python
0.005, 0.01, 0.02, 0.05
```

### `LVS_VAE_PROGRESS_THRESHOLD`

Current default:

```python
LVS_VAE_PROGRESS_THRESHOLD = 0.7
```

This was selected because analysis showed the live max-turn gate worked well at 70%.

Lower values use the endgame ensemble earlier. Higher values delay it.

### `LVS_VAE_ENDGAME_BLEND`

Current default:

```python
LVS_VAE_ENDGAME_BLEND = 0.7
```

After the threshold:

- `0.0` means full-game models only
- `0.5` means half full-game, half endgame
- `0.7` means 30% full-game and 70% endgame
- `1.0` means endgame models only

### `MAX_TURNS`

The live progress gate depends on `MAX_TURNS`.

If a variation overrides:

```python
"MAX_TURNS": 150
```

then progress becomes:

```python
turn_counter / 150
```

This matters because the endgame gate starts at 70% of the configured max-turn budget.

## 11. Recommended First PPO Test

Use a small debug run first:

```python
DEBUG_MODE = True
NUM_CPU = 1
RUN_LVS_VAE_SMOKE_CHECK_ONLY = False
VARIATIONS = LVS_VAE_SHAPED_VARIATIONS
```

Then run:

```powershell
python train_abc.py
```

Watch TensorBoard and compare:

- goat win rate
- tiger win rate
- max timeout rate
- repeat timeout rate
- total reward
- VAE reward component
- VAE value delta
- gate active fraction

Then compare against the same debug run with:

```python
"use_lvs_vae_shaping": False
```

## 12. Common Issues

### Smoke check cannot find checkpoints

Check the checkpoint paths:

```python
LVS_VAE_FULL_CHECKPOINT_PATHS
LVS_VAE_END_CHECKPOINT_PATHS
```

Each path must point to an existing `.pt` checkpoint with a `model_state_dict`.

### VAE reward is too dominant

Lower:

```python
LVS_VAE_SHAPING_COEF
```

The VAE component should usually be a shaping hint, not the main objective.

### Shaping does not appear in TensorBoard

Confirm the active phase has:

```python
"use_lvs_vae_shaping": True
```

Also confirm `LEARNER_ROLE = GOAT_LEARNER`. The v1 integration intentionally does not shape tiger learner rewards.

### Endgame gate activates too early or late

Check:

```python
LVS_VAE_PROGRESS_THRESHOLD
MAX_TURNS
```

The live gate is based on turn count, not the dataset's offline `move_index / total_moves`.

## 13. Current Design Choice

The VAE is a training-time reward wrapper, not a permanent environment rule.

That is intentional. It lets you run clean ablations:

- same environment, no VAE
- same environment, VAE shaping enabled
- same PPO settings, different shaping coefficient
- same PPO settings, different phase gate or checkpoint ensemble

This makes it much easier to tell whether the VAE value signal actually improves goat policy learning.
