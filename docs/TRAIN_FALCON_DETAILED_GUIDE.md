# Train Detailed Guide
### `If using vscode use ctrl+shift+V to view this as compiled mardown`  

Version context: `train_abc.py` (`mppo_train3.0`) + `env_tng_abc.py` (`env6.0`)  
Last updated: 2026-04-01

This keeps the same section structure as the older Falcon guide, but it documents the current repo entry point: `train_abc.py`.

## 1. Purpose of `train_abc.py`

`train_abc.py` is the main MaskablePPO training runner for this repo.

It runs an experiment suite made of one or more named variations. Each variation can be:

- a baseline/default run
- a single-phase run with config overrides
- a multi-phase curriculum that continues the same model across phases

Core pipeline per phase:

1. Build vectorized envs with `SubprocVecEnv`
2. Wrap envs with `Monitor`, `TigerMixWrapper`, or `GoatMixWrapper` when mixing is enabled
3. Apply `ActionMasker` so MaskablePPO only samples legal actions
4. Create or resume a `MaskablePPO` model
5. Train for the phase timestep budget
6. Save checkpoints and phase-final model snapshots
7. Log TensorBoard stats and move to the next phase if configured

## 2. What You Must Configure at the Top of the File

Edit these sections in `train_abc.py` before running.

### A. Identity and matchup config

- `EXPERIMENT_NAME`
- `LEARNER_ROLE`: `GOAT_LEARNER` or `TIGER_LEARNER`
- `OPPONENT_AI`
- `MIX_PROB`

Valid `OPPONENT_AI` values depend on learner role:

| Learner role | Valid opponent choices |
| --- | --- |
| `GOAT_LEARNER` | `OPP_TIGER_GREEDY`, `OPP_TIGER_SMART` |
| `TIGER_LEARNER` | `OPP_GOAT_RANDOM`, `OPP_GOAT_MODEL` |

`MIX_PROB` behavior:

- `None` = fixed opponent
- float in `[0, 1]` = per-reset probabilistic opponent mixing

### B. Version tags and output labels

- `ALGO_TAG`
- `ENV_VER`
- `MODEL_VER`

### C. Run scale and hardware

- `DEVICE_MODE`: `"cpu"` or `"gpu"`
- `DEBUG_MODE`
- `TIMESTEPS`
- `NUM_CPU`
- `SEED`

### D. PPO hyperparameters

- `GAMMA`
- `LEARNING_RATE`
- `ENT_COEF`
- `N_STEPS`
- `BATCH_SIZE`
- `N_EPOCHS`

The current file selects `N_STEPS`, `BATCH_SIZE`, and `N_EPOCHS` from the `DEVICE_MODE` block.

### E. Save and resume controls

- `CHECKPOINTS_PER_RUN`
- `RESUME_MODEL_PATH`
- `GOAT_MODEL_PATH`

`GOAT_MODEL_PATH` is only relevant when tiger-learner runs use goat-model opponents.

### F. Variation definitions

- `VARIATIONS`

This is the main experiment-design surface for the script.

## 3. How Matchups Are Resolved

`train_abc.py` uses one unified `OPPONENT_AI` setting, then resolves it by learner role through `resolve_opponent_or_exit(...)`.

### Goat learner

- `OPP_TIGER_GREEDY` -> tiger AI = greedy
- `OPP_TIGER_SMART` -> tiger AI = smart

### Tiger learner

- `OPP_GOAT_RANDOM` -> goat opponent AI = random
- `OPP_GOAT_MODEL` -> goat opponent AI = model

Important current behavior:

- goat learner mode varies the tiger opponent
- tiger learner mode always keeps the tiger learner active and varies the goat opponent type
- invalid learner/opponent combinations cause the script to exit early

## 4. What Happens When You Run

Command:

```powershell
python train_abc.py
```

### Startup behavior

1. Prints experiment summary and one core tag per variation
2. Iterates through all entries in `VARIATIONS`
3. For each variation:
   - `normalize_variation_config(...)` converts the config into a phase list
   - `resolve_variation_core(...)` computes the matchup/core tag
   - `run_single_variation(...)` executes the training flow one phase at a time

### Per-phase behavior

1. Resolve phase timesteps, opponent mode, `mix_prob`, checkpoint frequency, and knob overrides
2. Build `NUM_CPU` worker envs via `make_env(...)`
3. Wrap envs with:
   - `TigerMixWrapper` for mixed goat-learner runs
   - `GoatMixWrapper` for mixed tiger-learner runs
   - `Monitor` when not mixing
   - `ActionMasker` in all cases
4. Create or reuse the model
5. Train for the phase timesteps
6. Save a phase-final model
7. Track the best phase model by win-rate score
8. Close the vectorized env

### End of variation

At the end of a variation, the runner:

- aggregates goat/tiger/max-timeout/repeat-timeout rates
- saves a `best_*` model copy if a best phase was found
- reports the final saved model path

## 5. Output Locations

The current runner uses an experiment-first layout:

```text
artifacts/<experiment_name>/
```

Per-experiment folders:

- `meta/`
- `tb/`
- `checkpoints/`
- `models/`
- `eval/`
- `logs/`

### What goes where

| Folder | Contents |
| --- | --- |
| `meta/` | config snapshot JSON, git commit hash, notes file |
| `tb/` | TensorBoard runs, one subfolder per phase |
| `checkpoints/` | periodic checkpoints per phase |
| `models/` | saved final and best model exports |
| `eval/` | experiment-local evaluation outputs if you add them |
| `logs/` | auxiliary logs |

### Typical output names

Checkpoints:

```text
cp_<ALGO>_<CORE>_<variation_slug>_<run_stamp>_*.zip
```

Phase-final models:

```text
final_<variation_slug>_<run_stamp>_phase_XX.zip
```

Best model copies:

```text
best_<variation_slug>_<run_stamp>.zip
```

Usually best == final

## 6. TensorBoard Metrics to Watch

The runner logs three main groups of custom metrics.

### `1.episode_stats (total episodes)/*`

- episode counts
- cumulative goat win rate
- cumulative tiger win rate
- cumulative max-timeout rate
- cumulative repeat-timeout rate

### `2.window_stats/*`

- rolling goat/tiger/timeout/stall rates
- rolling realized opponent fractions in mixed runs
- rolling reward component summaries

### `3.matchup_info/*`

- learner role id
- opponent source id
- opponent family id
- mixed flag
- `mix_prob`

For actual training interpretation, read outcome metrics first and optimizer-health metrics second.

## 7. Detailed Variation Guide

This is the most important section for experiment design.

A variation is one key-value entry in `VARIATIONS`.

- the key is the variation name
- the value controls how it runs

### 7.1 Variation formats

#### Format A: `None` baseline

```python
VARIATIONS = {
    "baseline": None,
}
```

Meaning:

- uses global `TIMESTEPS`
- uses global `OPPONENT_AI`
- uses global `MIX_PROB`
- uses global `CHECKPOINTS_PER_RUN`
- uses global `GOAT_MODEL_PATH`
- uses environment defaults unless you changed them globally

#### Format B: single-phase dict

```python
VARIATIONS = {
    "no_bubble": {
        "timesteps": 30_000_000,
        "opponent_ai": OPP_TIGER_SMART,
        "mix_prob": None,
        "checkpoints_per_run": 12,
        "REWARD_BUBBLE_SPACE": 0.0,
    },
}
```

Meaning:

- one phase only
- control keys are parsed by the runner
- all other keys are treated as direct knob overrides

#### Format C: multi-phase curriculum

```python
VARIATIONS = {
    "goat_GT_to_ST_to_mix": [
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.5},
    ],
}
```

Meaning:

- trains continuously across phases on the same model
- each phase can change opponent, mix rate, timesteps, and knob overrides

### 7.2 Recognized per-phase control keys

These keys are consumed as phase controls, not environment knobs:

- `timesteps`
- `opponent_ai`
- `mix_prob`
- `checkpoints_per_run`
- `goat_model_path`

Anything else is treated as a direct knob override into `env_tng_abc.py`.

Important current behavior:

- legacy `tiger_ai` is rejected
- nested `reward_weights` is rejected

Use flat direct keys instead.

### 7.3 Reward override keys you can use

These map to `env_tng_abc.py` `DEFAULT_KNOBS`:

- `REWARD_GOAT_WIN`
- `REWARD_TIGER_WIN`
- `REWARD_TIGER_CAPTURE`
- `REWARD_TIGER_WIN_BONUS`
- `REWARD_TIGER_LOSS_PENALTY`
- `REWARD_STEP`
- `REWARD_GOAT_EATEN`
- `MAX_TIMEOUT_SCALE`
- `REPEAT_STALL_SCALE`
- `MAX_TURNS`
- `MOVE_STEP_BASE`
- `MOVE_STEP_SLOPE`
- `MOVE_STEP_MIN`
- `GOAT_WIN_TURN_DECAY`
- `REWARD_BLOCK_TIGER`
- `REWARD_BUBBLE_SPACE`
- `REWARD_CLUSTER_TIGERS`
- `REWARD_CENTER_GOAT`
- `REWARD_CENTER_TIGER`
- `NEAR_LOCK_BONUS`
- `LATE_GAME_START_TURN`
- `MOBILITY_BACKSLIDE_SCALE`
- `GOAT_SHAPING_DECAY_BASE`
- `REWARD_INVALID_SOFT`
- `REWARD_INVALID_HARD`
- `REWARD_REPEAT_STATE`
- `MAX_REPEATS`
- `BASE_TIGER_CAPTURE_BIAS`

Unknown keys are warned by the environment constructor and ignored.

### 7.4 Practical variation patterns

#### Pattern 1: baseline plus single ablations

```python
VARIATIONS = {
    "baseline": None,
    "no_bubble": {"REWARD_BUBBLE_SPACE": 0.0},
    "no_block": {"REWARD_BLOCK_TIGER": 0.0},
    "no_center": {
        "REWARD_CENTER_GOAT": 0.0,
        "REWARD_CENTER_TIGER": 0.0,
    },
}
```

#### Pattern 2: magnitude sweep

```python
VARIATIONS = {
    "bubble_025": {"REWARD_BUBBLE_SPACE": 0.25},
    "bubble_050": {"REWARD_BUBBLE_SPACE": 0.50},
    "bubble_100": {"REWARD_BUBBLE_SPACE": 1.00},
    "bubble_200": {"REWARD_BUBBLE_SPACE": 2.00},
}
```

#### Pattern 3: robustness curriculum for goat learner

```python
VARIATIONS = {
    "robust_goat_curriculum": [
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
        {"timesteps": 20_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.5},
    ],
}
```

#### Pattern 4: tiger learner with mixed goat opponents

```python
VARIATIONS = {
    "tiger_vs_mixed_goats": [
        {
            "timesteps": 30_000_000,
            "opponent_ai": OPP_GOAT_MODEL,
            "mix_prob": 0.4,
            "goat_model_path": r"stable_models\models\Goats\mppo_RobustGoat030726.zip",
        }
    ],
}
```

With current runner behavior, that means:

- 40% goat-model opponent
- 60% random-goat opponent

### 7.5 How to add a new variation safely

Checklist:

1. Pick a unique variation name
2. Decide single-phase vs multi-phase
3. Set timesteps per phase
4. Set `opponent_ai` and optional `mix_prob`
5. Add only valid knob keys
6. Set `checkpoints_per_run` high enough for recovery on long runs
7. Run a short debug pass before the long job

Recommended debug pass:

- `DEBUG_MODE = True`
- smaller `TIMESTEPS`
- lower `NUM_CPU` if startup or memory is unstable
- start with one variation before launching a large sweep

### 7.6 Common mistakes

- using goat-opponent enums while `LEARNER_ROLE` is goat, or tiger-opponent enums while `LEARNER_ROLE` is tiger
- forgetting `goat_model_path` when using `OPP_GOAT_MODEL`
- assuming `mix_prob` changes the model architecture; it only changes opponent sampling
- using legacy `tiger_ai` instead of `opponent_ai`
- using nested `reward_weights` instead of direct flat knob keys
- changing too many reward knobs at once before validating a baseline

## 8. Resume Training Notes

`RESUME_MODEL_PATH` supports:

- an exact `.zip` file path
- a directory, in which case the runner loads the newest `.zip` in that directory

Current continuation behavior:

- phase 1 loads the resume model if present
- resumed training is treated as continuation
- `reset_num_timesteps=False` is used for continuation phases

This is useful for:

- extending a prior run
- continuing a curriculum branch
- branching off from a previous checkpoint

## 9. Minimal Working Examples

### Example A: single baseline run

```python
LEARNER_ROLE = GOAT_LEARNER
OPPONENT_AI = OPP_TIGER_GREEDY
VARIATIONS = {"baseline": None}
```

### Example B: two-variation ablation run

```python
VARIATIONS = {
    "baseline": None,
    "ablate_repeat_penalty": {"REWARD_REPEAT_STATE": 0.0},
}
```

### Example C: short curriculum run

```python
VARIATIONS = {
    "GT_ST_MIX": [
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
    ],
}
```

## 10. Run Commands

Train:

```powershell
python train_abc.py
```

Monitor TensorBoard:

```powershell
tensorboard --logdir artifacts --port 6006
```

Evaluate a resulting model in single mode:

```powershell
python eval_abc.py battle path\to\model.zip
```

Important:

- `eval_abc.py` currently defaults to `EVAL_MODE = "sweep"` as committed
- for single-model evaluation, switch that file to `EVAL_MODE = "single"` first

## End of Guide
