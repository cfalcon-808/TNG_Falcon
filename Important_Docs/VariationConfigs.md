# Variation Config Reference
### `If using vscode use ctrl+shift+V to view this as compiled mardown`  
Reference for the `VARIATIONS` structure used by `train_abc.py`.

## Overview

Each entry in `VARIATIONS` defines one named training run inside the current experiment.

A variation can be one of three shapes:

1. `None`
2. a single `dict`
3. a `list[dict]`

What each shape means:

- `None` means one default single-phase run using the global settings already defined in `train_abc.py`
- a single `dict` means one explicit training phase
- a `list[dict]` means a multi-phase curriculum that continues the same model across phases

## Accepted Shapes

### 1. Default single-phase variation

```python
VARIATIONS = {
    "defaultSettings": None,
}
```

This inherits defaults from the current top-level `train_abc.py` settings such as:

- `TIMESTEPS`
- `OPPONENT_AI`
- `MIX_PROB`
- `CHECKPOINTS_PER_RUN`
- `GOAT_MODEL_PATH`

### 2. Single-phase variation

```python
VARIATIONS = {
    "goat_vs_smart": {
        "timesteps": 20_000_000,
        "opponent_ai": OPP_TIGER_SMART,
        "mix_prob": None,
        "REWARD_BUBBLE_SPACE": 0.0,
    },
}
```

### 3. Multi-phase curriculum

```python
VARIATIONS = {
    "greedy_to_smart_mix": [
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.5},
    ],
}
```

In a multi-phase variation, the runner keeps training the same model across all listed phases.

## Supported Control Keys

These keys are treated as runner controls, not environment reward knobs.

| Key | Type | Meaning |
| --- | --- | --- |
| `timesteps` | `int` | Number of timesteps for this phase. If omitted, falls back to the global `TIMESTEPS`. |
| `opponent_ai` | `str` | Unified opponent selector for the current learner role. |
| `mix_prob` | `float \| None` | Per-reset mixing probability. `None` means fixed opponent. |
| `checkpoints_per_run` | `int` | Number of checkpoints to save during this phase. If omitted, falls back to `CHECKPOINTS_PER_RUN`. |
| `goat_model_path` | `str \| None` | Path to a goat model `.zip` file. Relevant for tiger-learner runs that use goat model opponents. |

Any other key is treated as a direct override into `env_tng_abc.py` `DEFAULT_KNOBS`.

## Allowed `opponent_ai` Values

The valid `opponent_ai` values depend on the current `LEARNER_ROLE`.

### Goat learner

If `LEARNER_ROLE = GOAT_LEARNER`, valid values are:

- `OPP_TIGER_GREEDY`
- `OPP_TIGER_SMART`

These map to:

- `tiger_greedy`
- `tiger_smart`

### Tiger learner

If `LEARNER_ROLE = TIGER_LEARNER`, valid values are:

- `OPP_GOAT_RANDOM`
- `OPP_GOAT_MODEL`

These map to:

- `goat_random`
- `goat_model`

Important current behavior:

- tiger-learner runs always use the smart tiger side internally
- the `opponent_ai` choice in tiger-learner mode controls the goat opponent type, not the tiger type

## How `mix_prob` Works

`mix_prob` enables per-episode opponent mixing.

### Goat learner behavior

When `LEARNER_ROLE = GOAT_LEARNER`:

- `mix_prob = 0.3` means 30% smart tiger, 70% greedy tiger on reset
- `mix_prob = None` means fixed opponent

### Tiger learner behavior

When `LEARNER_ROLE = TIGER_LEARNER`:

- `mix_prob = 0.3` means 30% model goat, 70% random goat on reset
- `mix_prob = None` means fixed opponent

If you use goat-model opponents, `goat_model_path` should point to a valid saved goat model.

## Environment Knob Overrides

All non-control keys are passed directly into `env_tng_abc.py` as knob overrides.

Examples:

- `REWARD_GOAT_WIN`
- `REWARD_TIGER_WIN`
- `REWARD_GOAT_EATEN`
- `REWARD_BLOCK_TIGER`
- `REWARD_BUBBLE_SPACE`
- `REWARD_CLUSTER_TIGERS`
- `REWARD_CENTER_GOAT`
- `REWARD_CENTER_TIGER`
- `REWARD_TIGER_CAPTURE`
- `REWARD_TIGER_WIN_BONUS`
- `REWARD_TIGER_LOSS_PENALTY`
- `MAX_TURNS`
- `REWARD_REPEAT_STATE`
- `BASE_TIGER_CAPTURE_BIAS`

Put these keys directly in the variation or phase dict.

## Current Template

This is a complete current template aligned with the live knobs in `env_tng_abc.py`.

```python
VARIATIONS = {
    "blank_template": {
        "timesteps": 50_000_000,
        "opponent_ai": None,
        "mix_prob": None,
        "checkpoints_per_run": 10,
        "goat_model_path": None,

        # Goat learner terminal rewards
        "REWARD_GOAT_WIN": 3.2,
        "REWARD_TIGER_WIN": -3.2,

        # Tiger learner terminal / capture rewards
        "REWARD_TIGER_CAPTURE": 0.35,
        "REWARD_TIGER_WIN_BONUS": 3.2,
        "REWARD_TIGER_LOSS_PENALTY": 3.2,

        # Shared step and material shaping
        "REWARD_STEP": -0.001,
        "REWARD_GOAT_EATEN": -0.35,

        # Timeout controls
        "MAX_TIMEOUT_SCALE": 1.5,
        "REPEAT_STALL_SCALE": 1.25,
        "MAX_TURNS": 100,

        # Goat move-phase tempo shaping
        "MOVE_STEP_BASE": -0.01,
        "MOVE_STEP_SLOPE": -0.01,
        "MOVE_STEP_MIN": -0.6,
        "GOAT_WIN_TURN_DECAY": 0.01,

        # Goat tactical and positional shaping
        "REWARD_BLOCK_TIGER": 0.08,
        "REWARD_BUBBLE_SPACE": 0.02,
        "REWARD_CLUSTER_TIGERS": 0.02,
        "REWARD_CENTER_GOAT": 0.03,
        "REWARD_CENTER_TIGER": -0.03,
        "NEAR_LOCK_BONUS": 0.05,
        "LATE_GAME_START_TURN": 40,
        "MOBILITY_BACKSLIDE_SCALE": 0.5,
        "GOAT_SHAPING_DECAY_BASE": 0.95,

        # Invalid action and repeat controls
        "REWARD_INVALID_SOFT": -0.05,
        "REWARD_INVALID_HARD": -1.0,
        "REWARD_REPEAT_STATE": -0.5,
        "MAX_REPEATS": 4,

        # Scripted tiger behavior
        "BASE_TIGER_CAPTURE_BIAS": 1.0,
    },
}
```

## Practical Examples

### Minimal ablation

```python
VARIATIONS = {
    "no_bubble": {
        "REWARD_BUBBLE_SPACE": 0.0,
    },
}
```

This inherits global training defaults and only overrides one knob.

### Fixed smart-tiger goat run

```python
VARIATIONS = {
    "goat_vs_ST": {
        "timesteps": 50_000_000,
        "opponent_ai": OPP_TIGER_SMART,
    },
}
```

### Goat curriculum

```python
VARIATIONS = {
    "goat_GT_to_ST_to_mix": [
        {"timesteps": 50_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.5},
    ],
}
```

### Tiger learner against mixed goat opponents

```python
VARIATIONS = {
    "tiger_vs_mixed_goats": {
        "timesteps": 20_000_000,
        "opponent_ai": OPP_GOAT_MODEL,
        "mix_prob": 0.4,
        "goat_model_path": r"stable_models\models\Goats\mppo_RobustGoat030726.zip",
    },
}
```

With current runner behavior, that means:

- 40% model goat
- 60% random goat

## What Is Not Supported

These older keys are explicitly rejected by `train_abc.py`:

- `tiger_ai`
- nested `reward_weights`

Use this instead:

- `opponent_ai` instead of `tiger_ai`
- direct flat knob keys instead of nested `reward_weights`

Example of the current correct style:

```python
VARIATIONS = {
    "direct_override": {
        "opponent_ai": OPP_TIGER_SMART,
        "REWARD_BLOCK_TIGER": 0.12,
        "REWARD_CENTER_GOAT": 0.05,
    },
}
```

## Fallback Rules

If a key is omitted from a phase dict:

- `timesteps` falls back to global `TIMESTEPS`
- `opponent_ai` falls back to global `OPPONENT_AI`
- `mix_prob` falls back to global `MIX_PROB`
- `checkpoints_per_run` falls back to global `CHECKPOINTS_PER_RUN`
- `goat_model_path` falls back to global `GOAT_MODEL_PATH`

If a variation is `None`, all of those defaults apply automatically.

## Short Takeaways

- Keep variation dicts flat.
- Use `opponent_ai`, not `tiger_ai`.
- Put reward knobs directly in the phase dict.
- Use `list[dict]` when you want a curriculum that continues the same model.
- `mix_prob` changes meaning depending on the learner role.
