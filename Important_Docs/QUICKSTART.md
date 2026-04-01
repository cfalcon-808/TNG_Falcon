# Quickstart
### `If using vscode use ctrl+shift+V to view this as compiled mardown`  
Practical setup and first-run workflow for the current `*_abc.py` training and evaluation scripts.

## What This Covers

This guide walks through:

- environment setup
- running a first training experiment
- monitoring with TensorBoard
- evaluating trained models
- launching the GUI
- checking model architecture summaries

## Prerequisites

- Python 3.10 or newer
- Windows PowerShell examples are used below
- NVIDIA GPU is optional, but preferred for longer training runs
- If you want GPU support, install the PyTorch build that matches your CUDA version

Project dependencies:

```powershell
pip install -r docs/requirements.txt
```

PyTorch GPU install instructions:

- https://pytorch.org/get-started/locally/

## 1. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r docs/requirements.txt
```

If you are training on GPU, install the correct PyTorch wheel after that.

## 2. Know the Main Scripts

| File | Purpose |
| --- | --- |
| `env_tng_abc.py` | Unified Tigers & Goats Gymnasium environment with legal-action masking and reward knob overrides. |
| `train_abc.py` | Main experiment runner for MaskablePPO training. |
| `eval_abc.py` | Evaluation runner for either one model (`single`) or a full model sweep (`sweep`). |
| `gui_abc.py` | Tkinter GUI for live play, model smoke tests, and replay browsing. |
| `model_summary_abc.py` | Prints `torchsummary` reports for fresh or saved models. |

## 3. Run a First Training Smoke Test

Open `train_abc.py` and review these settings first:

```python
EXPERIMENT_NAME = "my_first_experiment"
LEARNER_ROLE = GOAT_LEARNER
OPPONENT_AI = OPP_TIGER_GREEDY
DEBUG_MODE = True
```

Then set a minimal variation:

```python
VARIATIONS = {
    "defaultSettings": None,
}
```

Run training:

```powershell
python train_abc.py
```

### Useful variation patterns

Single-phase override:

```python
VARIATIONS = {
    "no_bubble": {
        "timesteps": 2_000_000,
        "opponent_ai": OPP_TIGER_SMART,
        "REWARD_BUBBLE_SPACE": 0.0,
    }
}
```

Multi-phase curriculum:

```python
VARIATIONS = {
    "greedy_to_smart_mix": [
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_GREEDY},
        {"timesteps": 10_000_000, "opponent_ai": OPP_TIGER_SMART},
        {"timesteps": 30_000_000, "opponent_ai": OPP_TIGER_SMART, "mix_prob": 0.3},
    ],
}
```

## 4. Monitor Training

Run TensorBoard from the repo root:

```powershell
tensorboard --logdir {"Desired Experiment Directory"} --port 6006
```

The active training runner writes experiment outputs under:

```text
artifacts/<experiment_name>/
```

That folder contains:

- `meta/`
- `tb/`
- `checkpoints/`
- `models/`
- `eval/`
- `logs/`

## 5. Evaluate a Trained Model

`eval_abc.py` has two modes:

- `single` for one learner model against one configured opponent setup
- `sweep` for full goat-model x tiger-model comparisons

### Important current default

As currently committed, `eval_abc.py` defaults to:

```python
EVAL_MODE = "sweep"
```

That means CLI arguments like `battle path\to\model.zip` are only used after you switch the file to `single` mode.

### Single-model evaluation

In `eval_abc.py`, set:

```python
EVAL_MODE = "single"
```

Then either configure `SINGLE_CONFIG["model_path"]` directly, or pass a model path on the command line:

```powershell
python eval_abc.py
python eval_abc.py normal
python eval_abc.py battle
python eval_abc.py battle path\to\model.zip
```

Mode meaning:

- `normal` = greedy tiger
- `battle` = smart tiger

Single-mode logs are written to:

```text
artifacts/logging/eval_debug/
```

### Sweep evaluation

If you want the full configured sweep instead, leave:

```python
EVAL_MODE = "sweep"
```

Then update `SWEEP_CONFIG` as needed and run:

```powershell
python eval_abc.py
```

Configure the sweep in `eval_abc.py` by editing these two dictionaries:

```python
SWEEP_CONFIG = {
    "tiger_models": {
        "tiger_label_1": r"path\to\tiger_model_1.zip",
        "tiger_label_2": r"path\to\tiger_model_2.zip",
    },
    "goat_models": {
        "goat_label_1": r"path\to\goat_model_1.zip",
        "goat_label_2": r"path\to\goat_model_2.zip",
    },
}
```

Important:

- Each value should be a path to a specific `.zip` model file.
- The current sweep code does not auto-scan a directory for models.
- The dictionary key is just the display name used in the printed summary, JSON, and CSV outputs.

Example using model folders in this repo:

```python
SWEEP_CONFIG = {
    "n_eval_games": 100,
    "deterministic": True,
    "save_results": True,
    "results_dir": "artifacts/eval_sweeps",
    "tiger_models": {
        "robust_tiger": r"stable_models\models\Tigers\mppo_RobustTiger030726.zip",
        "best_vs_normal_goat": r"stable_models\models\Tigers\best_tiger_vs_normal_goat_10M_20260307_220408.zip",
    },
    "goat_models": {
        "normal_goat": r"stable_models\models\Goats\mppo_NormalGoat030726.zip",
        "robust_goat": r"stable_models\models\Goats\mppo_RobustGoat030726.zip",
    },
}
```

If your models are grouped in folders, open the folder, find the exact `.zip` files you want, and add one entry per model. For example:

- `artifacts\baselineGoatTrainingV3\models\best_robust_goat_training_20260321_070755.zip`
- `stable_models\models\Tigers\mppo_RobustTiger030726.zip`

The sweep will run every goat entry against every tiger entry.

Sweep outputs are saved under:

```text
artifacts/eval_sweeps/
```

## 6. Launch the GUI

At a high level, the GUI is a visual front end for the Tigers & Goats environment. It lets you open the board, play against the built-in tiger opponent, load a trained goat model for quick smoke tests, and inspect recorded games without using the evaluation script output directly.

The main modes are:

- live human play, where you make goat moves yourself on the board
- model-assisted play, where a loaded goat model can make moves for the goat side
- replay browsing, where you step through a recorded game timeline and inspect what happened move by move

You can also switch the tiger opponent style:

- `normal` uses the greedy tiger
- `battle` uses the smart tiger

Vanilla launch with default settings:

```powershell
python gui_abc.py
```

This opens the GUI with its default mode, which currently maps to `--env normal`.

Greedy tiger:

```powershell
python gui_abc.py --env normal
```

Smart tiger:

```powershell
python gui_abc.py --env battle
```

Smart tiger with a goat model loaded:

```powershell
python gui_abc.py --env battle --model path\to\model.zip
```

## 7. Print a Model Summary

Saved model:

```powershell
python model_summary_abc.py --model-path path\to\model.zip
```

Specific module:

```powershell
python model_summary_abc.py --module policy.mlp_extractor.policy_net
```

Fresh model with custom net architecture:

```powershell
python model_summary_abc.py --net-arch 256,256,256 --device auto
```

## 8. Common Training Knobs

Common settings in `train_abc.py`:

- `EXPERIMENT_NAME`
- `LEARNER_ROLE`
- `OPPONENT_AI`
- `MIX_PROB`
- `DEBUG_MODE`
- `TIMESTEPS`
- `NUM_CPU`
- `DEVICE_MODE`
- `CHECKPOINTS_PER_RUN`
- `RESUME_MODEL_PATH`
- `VARIATIONS`

Typical reward knobs inside a variation:

- `REWARD_GOAT_WIN`
- `REWARD_TIGER_WIN`
- `REWARD_GOAT_EATEN`
- `REWARD_BLOCK_TIGER`
- `REWARD_BUBBLE_SPACE`
- `REWARD_CENTER_GOAT`
- `MAX_TURNS`

## 9. If Something Breaks

- Confirm the virtual environment is active.
- Verify that PyTorch matches your installed CUDA version.
- Reduce `NUM_CPU` if worker startup is failing.
- Turn `DEBUG_MODE = True` for shorter test runs.
- In `eval_abc.py`, make sure `EVAL_MODE` matches what you are trying to do.
- Use an explicit model path when validating a single saved checkpoint.
- If TensorBoard looks cluttered, check whether you are pointing it at the full `artifacts` tree or a specific experiment folder.

## Summary

- `env_tng_abc.py` is the shared environment for training, evaluation, and GUI use.
- `train_abc.py` runs one named experiment suite containing one or more variations.
- `eval_abc.py` defaults to sweep mode unless you change it to `single`.
- `gui_abc.py` is the fastest way to do a visual smoke test.
- `model_summary_abc.py` is useful for confirming architecture details on saved models.
