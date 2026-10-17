# Quickstart

Run commands from the repository root. The source uses Python 3.10+ syntax.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The GUI requires Tkinter, included with standard Windows Python installations.

## 2. Play

```powershell
python gui.py
```

Choose a player mode or open the replay viewer. Pretrained models are in `stable_models/models/Goats/` and `stable_models/models/Tigers/`.

## 3. Evaluate

Set `EVAL_MODE = "single"` in `eval.py`, then run:

```powershell
python eval.py normal stable_models/models/Goats/mppo_RobustGoat030726.zip
```

Use `battle` instead of `normal` for smart tigers. Results go under `artifacts/logging/eval_debug/`. Set `EVAL_MODE = "sweep"` and review `SWEEP_CONFIG` to compare model matchups.

## 4. Train

For a short CPU run, change these settings in `train.py` first:

```python
EXPERIMENT_NAME = "first_training"
DEVICE_MODE = "cpu"
TIMESTEPS = 2_048
NUM_CPU = 1
CHECKPOINTS_PER_RUN = 2
```

The script retains full-scale GPU/multiple-worker defaults, so apply these small settings before running on a CPU machine. The default `VARIATIONS = {"baseline": None}` uses the global settings and the environment's handcrafted rewards.

```powershell
python train.py
tensorboard --logdir artifacts --port 6006
```

Outputs go under `artifacts/<experiment_name>/`. A short run verifies the workflow; training a strong policy takes longer.

## 5. Experiment

- Change reward knobs in a variation to compare reward settings.
- Use a list of phase dictionaries for a curriculum.
- Set `RESUME_MODEL_PATH` to a saved policy to continue training.
- For tiger training, select `TIGER_LEARNER` and `OPP_GOAT_RANDOM` or `OPP_GOAT_MODEL`. For model opponents, set `GOAT_MODEL_PATH` to an existing file under `stable_models/models/Goats/`.

See [variations](VARIATION_CONFIGS.md), the [training guide](TRAIN_SCRIPT_DETAILED_GUIDE.md), and [rewards](REWARD_REFERENCE.md).

Inspect a policy:

```powershell
python model_summary.py --model-path stable_models/models/Goats/mppo_RobustGoat030726.zip
```
