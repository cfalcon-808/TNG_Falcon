# Tigers & Goats - Falcon Branch
### `If using vscode use ctrl+shift+V to view this as compiled mardown`  

Reinforcement learning environment, training runner, evaluation tools, and GUI utilities for the Tigers & Goats (Bagh-Chal) board game.

Read [`Important_Docs/QUICKSTART.md`](Important_Docs/QUICKSTART.md) for the full setup workflow. Python dependencies live in [`docs/requirements.txt`](docs/requirements.txt).

## Overview

This repository is built around a single unified Gymnasium environment and a small set of workflow scripts:

- `env_tng_abc.py` defines the game environment, reward knobs, legal-action masking, and opponent behavior.
- `train_abc.py` runs experiment suites made of one or more training variations, including multi-phase curricula.
- `eval_abc.py` evaluates trained models in either single-run or sweep mode.
- `gui_abc.py` provides a Tkinter GUI for live play, model smoke tests, and replay inspection.
- `model_summary_abc.py` prints `torchsummary` reports for fresh or saved MaskablePPO models.

The current codebase uses the `*_abc.py` files as the active entry points. Older `*_falcon.py` names still appear in some comments and docs, but they are not the current scripts at the repo root.

## Core Concepts

| Term | Meaning |
| --- | --- |
| Experiment | A named suite of related training runs, grouped under one artifact directory. |
| Variation | A single training configuration inside an experiment. A variation can be one phase or multiple phases. |
| Phase | One contiguous training segment within a variation. Multi-phase variations continue the same model across phases. |
| Core tag | Matchup shorthand such as `GvNT`, `GvST`, `GvMixT`, or `TvRG`. |
| Action masking | Legal actions are exposed through `get_action_mask()` so MaskablePPO never samples invalid moves. |

## Active Files

| File | Purpose |
| --- | --- |
| `env_tng_abc.py` | Unified Tigers & Goats environment with `Discrete(115)` action space, sparse/shaped rewards, repetition tracking, and greedy/smart/model opponent support. |
| `train_abc.py` | Experiment runner using `sb3_contrib.MaskablePPO`, `SubprocVecEnv`, checkpointing, TensorBoard logging, resume support, and per-phase curricula. |
| `eval_abc.py` | Single-model debug/evaluation runner plus full goat-model x tiger-model sweep evaluation. |
| `gui_abc.py` | Tkinter board UI for manual play, model-assisted play, and replay browsing. |
| `model_summary_abc.py` | Utility for printing model or module summaries with `torchsummary`. |

## Project Layout

```text
.
|-- env_tng_abc.py
|-- train_abc.py
|-- eval_abc.py
|-- gui_abc.py
|-- model_summary_abc.py
|-- Important_Docs/
|   |-- QUICKSTART.md
|   |-- REWARD_REFERENCE.md
|   |-- ROBUST_GOAT_TRAINING_STRATEGY.txt
|   `-- VariationConfigs.md
|-- docs/
|   |-- requirements.txt
|   |-- API_Index.txt
|   |-- HyperparameterGuide.txt
|   |-- RewardStructureOverview.txt
|   `-- TRAIN_FALCON_DETAILED_GUIDE.txt
|-- artifacts/
|   `-- <experiment_name>/
|       |-- meta/
|       |-- tb/
|       |-- checkpoints/
|       |-- models/
|       |-- eval/
|       `-- logs/
`-- stable_models/
    `-- models/
```

## Environment

`env_tng_abc.py` is the shared environment used by training, evaluation, and the GUI.

Current environment capabilities:

- Full game flow: goat placement phase and movement phase.
- Native `Discrete(115)` action space.
- Legal-action masking compatible with MaskablePPO.
- Goat learner and tiger learner modes.
- Greedy tiger, smart tiger, and model-driven tiger support.
- Random goat and model-driven goat opponent support.
- Reward shaping via configurable knob overrides.
- Repeat-state detection and timeout handling.

The default observation shape is 25 values:

- 23 board positions
- goats eaten
- phase indicator

## Quick Start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r docs/requirements.txt
```

If you want GPU training, install the correct PyTorch build for your CUDA version separately.

### 2. Run training

Edit `train_abc.py` first, then run:

```powershell
python train_abc.py
```

Monitor TensorBoard with:

```powershell
tensorboard --logdir artifacts --port 6006
```

### 3. Evaluate a model

Single-run evaluation is configured inside `eval_abc.py`. The script also accepts the older convenience CLI style for greedy vs smart tiger mode:

```powershell
python eval_abc.py
python eval_abc.py normal
python eval_abc.py battle
python eval_abc.py battle path\to\model.zip
```

### 4. Launch the GUI

```powershell
python gui_abc.py
python gui_abc.py --env normal
python gui_abc.py --env battle
python gui_abc.py --env battle --model path\to\model.zip
```

### 5. Print a model summary

```powershell
python model_summary_abc.py --model-path path\to\model.zip
python model_summary_abc.py --module policy.mlp_extractor.policy_net
```

## Training Workflow

`train_abc.py` treats one run as an experiment suite containing one or more variations.

Each variation can be:

- `None` for default environment settings.
- A single flat `dict` for one training phase.
- A `list[dict]` for a multi-phase curriculum that keeps training the same model across phases.

Typical controls in a variation:

- `timesteps`
- `opponent_ai`
- `mix_prob`
- reward knob overrides such as `REWARD_BUBBLE_SPACE`, `REWARD_BLOCK_TIGER`, or `MAX_TURNS`

Example:

```python
VARIATIONS = {
    "defaultSettings": None,
    "no_bubble": {
        "timesteps": 20_000_000,
        "opponent_ai": OPP_TIGER_SMART,
        "REWARD_BUBBLE_SPACE": 0.0,
    },
    "greedy_to_smart_mix": [
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
    ],
}
```

## Current Checked-In Defaults

As currently committed, `train_abc.py` defaults to:

- `EXPERIMENT_NAME = "baselineGoatTrainingV3"`
- goat learner mode
- smart tiger opponent
- `DEVICE_MODE = "gpu"`
- `NUM_CPU = 16`
- `TIMESTEPS = 20_000_000` when `DEBUG_MODE = False`
- checkpoint resume enabled through `RESUME_MODEL_PATH`

As currently committed, `eval_abc.py` defaults to:

- `EVAL_MODE = "sweep"`
- `DEBUG_MODE = True`
- `100` evaluation games per configured sweep pairing
- stable model comparisons from `stable_models/models/Goats` and `stable_models/models/Tigers`

Those values are configuration defaults, not fixed project requirements. Change them in the script before running if they do not match your target experiment.

## Artifacts

Training artifacts are written under:

```text
artifacts/<experiment_name>/
```

The active training runner creates:

- `meta/` for saved config snapshots, commit hashes, and notes
- `tb/` for TensorBoard logs
- `checkpoints/` for periodic phase checkpoints
- `models/` for final and best model exports
- `eval/` for experiment-local evaluation outputs
- `logs/` for auxiliary logs

Evaluation sweep outputs are written to:

```text
artifacts/eval_sweeps/
```

Single-run debug logs are written to:

```text
artifacts/logging/eval_debug/
```

## Matchup Tags

| Tag | Meaning |
| --- | --- |
| `GvNT` | Goat learner vs normal/greedy tiger |
| `GvST` | Goat learner vs smart tiger |
| `GvMixT` | Goat learner vs mixed tiger sampling |
| `TvRG` | Tiger learner vs random goat |

Additional `Tv*` tags may appear depending on opponent configuration.

## Notes

- The environment and tooling are built around `sb3_contrib.MaskablePPO`.
- `docs/requirements.txt` lists the core Python packages, but PyTorch GPU wheels should be installed separately when needed.
- Some internal file headers still mention older Falcon-era filenames. The repo-root script names in this README are the current ones to use.

## Reference Docs

- [`Important_Docs/QUICKSTART.md`](Important_Docs/QUICKSTART.md)
- [`Important_Docs/REWARD_REFERENCE.md`](Important_Docs/REWARD_REFERENCE.md)
- [`Important_Docs/ROBUST_GOAT_TRAINING_STRATEGY.txt`](Important_Docs/ROBUST_GOAT_TRAINING_STRATEGY.txt)
- [`Important_Docs/VariationConfigs.md`](Important_Docs/VariationConfigs.md)
- [`docs/API_Index.txt`](docs/API_Index.txt)
- [`docs/HyperparameterGuide.txt`](docs/HyperparameterGuide.txt)
- [`docs/RewardStructureOverview.txt`](docs/RewardStructureOverview.txt)
- [`docs/TRAIN_FALCON_DETAILED_GUIDE.txt`](docs/TRAIN_FALCON_DETAILED_GUIDE.txt)
