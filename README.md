# Tigers and Goats LVS-VAE Reward Shaping

Final project deliverable for integrating latent value signals into a Tigers and Goats reinforcement learning pipeline.

This branch is organized around the VAE portion of the project first. It keeps the PPO environment and stable models needed to reproduce the workflow, but removes the experimental placing-survival / PS VAE path.

## Project Focus

The project tests whether a frozen Latent Value Shaping VAE can provide an auxiliary reward signal during Maskable PPO training. The VAE learns from board-state datasets and predicts a continuous goat-favorable value. PPO training then receives a small value-delta reward:

```text
R_total = R_sparse + alpha * (V_LVS(s_next) - V_LVS(s))
```

The current implementation should be read as a proof of concept. It shows that the learned signal can be integrated and logged without breaking PPO training, but it does not replace the handcrafted reward system.

## LVS-VAE Architecture

The active model is `VAE/LVS_VAE.py`.

| Component | Current setting |
| --- | --- |
| Input | 25 features: 23 board cells, goats eaten, phase |
| Encoder | MLP hidden sizes `(64, 64)` |
| Latent dimension | 8 |
| Decoder | MLP hidden sizes `(64, 64)` |
| Value head | Latent mean to scalar goat-favorable value |
| Loss | reconstruction MSE + beta KL + value MSE |

The value head predicts labels where goat win is `1.0`, tie is `0.5`, and tiger win is `0.0`.

## Reward Shaping Workflow

The active reward helper is `VAE/LVS_VALUE_SHAPING.py`.

Training uses a frozen phase-gated ensemble:

```text
early game: V_LVS = mean(full-game models)
late game : V_LVS = 0.3 * mean(full-game models) + 0.7 * mean(endgame models)
```

`train_abc.py` wires this into the environment by wrapping the normal goat reward with `make_lvs_vae_reward_fn(...)`. The core game rules remain in `env_tng_abc.py`.

## Important Files

| Path | Purpose |
| --- | --- |
| `VAE/LVS_VAE.py` | LVS-VAE model, value head, and loss function |
| `VAE/LVS_VALUE_SHAPING.py` | Frozen value predictor and PPO reward wrapper |
| `VAE/specialized_training_scripts/TRAIN_VAE.py` | Shared VAE trainer |
| `VAE/specialized_training_scripts/TRAIN_FULL_LVS_VAE.py` | Full-game value model entrypoint |
| `VAE/specialized_training_scripts/TRAIN_END_LVS_VAE.py` | Endgame value model entrypoint |
| `VAE/specialized_training_scripts/TRAIN_LVS_VAE_HEADS.py` | Frozen-body value-head trainer |
| `VAE/notebooks/DATASET_GENERATOR.ipynb` | Gameplay dataset generator |
| `VAE/notebooks/VAE_DATASET_ANALYSIS.ipynb` | Dataset balance and quality analysis |
| `VAE/notebooks/VAE_HEAD_ANALYSIS.ipynb` | Value-head and training-curve analysis |
| `env_tng_abc.py` | Gymnasium Tigers and Goats environment |
| `train_abc.py` | Maskable PPO experiment runner |
| `eval_abc.py` | PPO model evaluation |
| `gui_abc.py` | Tkinter play/replay GUI |

## Data and Artifacts

| Path | Contents |
| --- | --- |
| `VAE_DATA/` | Generated full-game and endgame VAE datasets |
| `artifacts/lvs_vae/` | Trained LVS-VAE checkpoints, configs, metrics, and PNG curves |
| `artifacts/VAE_ABLATION_ANALYSIS/` | PPO comparison runs used for VAE reward-shaping analysis |
| `stable_models/` | Stable PPO goat and tiger models used by the environment and dataset workflow |
| `AI_DISCLOSURE/` | Raw and readable AI chat records for disclosure |

## Quickstart

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

If you need CUDA-enabled PyTorch, install the matching PyTorch wheel from the official PyTorch instructions after creating the environment.

Run the VAE reward-shaping smoke check:

```powershell
python train_abc.py
```

For a smoke-check-only run, set this in `train_abc.py`:

```python
RUN_LVS_VAE_SMOKE_CHECK_ONLY = True
```

Analyze datasets or heads:

```powershell
jupyter notebook VAE/notebooks/VAE_DATASET_ANALYSIS.ipynb
jupyter notebook VAE/notebooks/VAE_HEAD_ANALYSIS.ipynb
```

Train PPO:

```powershell
python train_abc.py
```

Monitor PPO runs:

```powershell
tensorboard --logdir artifacts --port 6006
```

Evaluate or inspect models:

```powershell
python eval_abc.py
python gui_abc.py
python model_summary_abc.py --model-path stable_models/models/Goats/mppo_RobustGoat030726.zip
```

## Documentation

Start with these docs:

| Path | Purpose |
| --- | --- |
| `Important_Docs/QUICKSTART.md` | Practical setup and first-run workflow |
| `Important_Docs/LVS_VAE_REWARD_SHAPING_GUIDE.md` | Value-head reward shaping details |
| `Important_Docs/REWARD_REFERENCE.md` | Handcrafted reward terms and VAE reward logging |
| `Important_Docs/FULL40_RECON_THEN_HEADS_WORKFLOW.md` | Reconstruction-then-value-head training workflow |

## Notes

This branch intentionally excludes the placing-survival / PS VAE experiment. The final active learned reward signal is the goat-favorable value-head delta from the full/endgame LVS-VAE ensemble.
