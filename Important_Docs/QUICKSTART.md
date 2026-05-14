# Quickstart

Practical setup and first-run workflow for the final LVS-VAE reward-shaping deliverable.

## 1. Install

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU training, install the PyTorch build that matches your CUDA version after the base install.

## 2. Main Files

| File | Purpose |
| --- | --- |
| `VAE/LVS_VAE.py` | Value-head LVS-VAE model and loss |
| `VAE/LVS_VALUE_SHAPING.py` | Frozen full/endgame value predictor and reward wrapper |
| `VAE/notebooks/DATASET_GENERATOR.ipynb` | Generates gameplay state datasets |
| `VAE/notebooks/VAE_DATASET_ANALYSIS.ipynb` | Reviews dataset balance and state metadata |
| `VAE/notebooks/VAE_HEAD_ANALYSIS.ipynb` | Reviews VAE training curves and value-head behavior |
| `env_tng_abc.py` | Tigers and Goats Gymnasium environment |
| `train_abc.py` | Maskable PPO experiment runner |
| `eval_abc.py` | Model evaluation runner |
| `gui_abc.py` | Tkinter board GUI |

## 3. Smoke Test VAE Reward Shaping

`train_abc.py` can load the frozen VAE ensemble and test one shaped environment transition before PPO training.

For a smoke-check-only run, set:

```python
RUN_LVS_VAE_SMOKE_CHECK = True
RUN_LVS_VAE_SMOKE_CHECK_ONLY = True
```

Then run:

```powershell
python train_abc.py
```

Set `RUN_LVS_VAE_SMOKE_CHECK_ONLY = False` when you want training to continue after the check.

## 4. Run PPO Training

Review these values near the top of `train_abc.py`:

```python
EXPERIMENT_NAME = "VAE_ABLATION_ANALYSIS"
USE_LVS_VAE_SHAPING = True
LVS_VAE_SHAPING_COEF = 0.5
LVS_VAE_PROGRESS_THRESHOLD = 0.7
LVS_VAE_ENDGAME_BLEND = 0.7
```

Run training:

```powershell
python train_abc.py
```

Outputs are written under:

```text
artifacts/<experiment_name>/
```

## 5. Monitor Training

```powershell
tensorboard --logdir artifacts --port 6006
```

Useful VAE-related TensorBoard fields include:

- `reward_vae_component`
- `reward_total`
- `lvs_value_before`
- `lvs_value_after`
- `lvs_value_delta`
- `lvs_progress_ratio`
- `lvs_gate_active`

## 6. Work With VAE Data

Open the notebooks:

```powershell
jupyter notebook VAE/notebooks/DATASET_GENERATOR.ipynb
jupyter notebook VAE/notebooks/VAE_DATASET_ANALYSIS.ipynb
jupyter notebook VAE/notebooks/VAE_HEAD_ANALYSIS.ipynb
```

The checked-in datasets are under `VAE_DATA/`. Each dataset folder contains:

- `states_labels.npz`
- `dataset.csv`
- `metadata.csv`
- `dataset_summary.json`

## 7. Evaluate or Inspect PPO Models

```powershell
python eval_abc.py
python gui_abc.py
python model_summary_abc.py --model-path stable_models/models/Goats/mppo_RobustGoat030726.zip
```

`stable_models/` contains the stable PPO models kept for reproducible environment and dataset workflows.
