# Full 40k Reconstruction Then Heads Workflow

Goal: train the VAE latent body on full 40k reconstruction first, then freeze it and train prediction heads.

## 1. Reconstruction Baseline

Run:

```powershell
python VAE/specialized_training_scripts/TRAIN_RECONSTRUCTION_LVS_VAE.py
```

Uses:

- Dataset: `VAE_DATA/full_40k_40_40_20/states_labels.npz`
- Train mode: `reconstruction`
- Losses: reconstruction + KL only
- Output: `artifacts/lvs_vae/full_40k_reconstruction_lvsvae_v1/full_40k_reconstruction_lvsvae_v1_final_model.pt`

## 2. Frozen VAE Heads

Run:

```powershell
python VAE/specialized_training_scripts/TRAIN_LVS_VAE_HEADS.py
```

Uses:

- Initial checkpoint: full 40k reconstruction final model
- Train mode: `prediction_heads`
- Frozen: encoder, latent layers, decoder, reconstruction head
- Trainable: outcome value head + placing-survival head

## Optional: Survival Head Only

Run:

```powershell
python VAE/specialized_training_scripts/TRAIN_PLACING_SURVIVAL_LVS_VAE.py
```

This freezes the full 40k reconstruction VAE and trains only the placing-survival head.

## One Command

Train reconstruction, then both heads:

```powershell
powershell -ExecutionPolicy Bypass -File VAE/specialized_training_scripts/RUN_FULL40_RECON_THEN_HEADS.ps1
```

Train reconstruction, then survival head only:

```powershell
powershell -ExecutionPolicy Bypass -File VAE/specialized_training_scripts/RUN_FULL40_RECON_THEN_HEADS.ps1 -SurvivalOnly
```
