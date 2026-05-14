# Full 40k Reconstruction Then Value-Head Workflow

Goal: train the LVS-VAE latent body on full 40k reconstruction first, then freeze the body and train the outcome value head.

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

## 2. Frozen Value Head

Run:

```powershell
python VAE/specialized_training_scripts/TRAIN_LVS_VAE_HEADS.py
```

Uses:

- Initial checkpoint: full 40k reconstruction final model
- Train mode: `value_head`
- Frozen: encoder, latent layers, decoder, reconstruction head
- Trainable: outcome value head

## One Command

Train reconstruction, then the value head:

```powershell
powershell -ExecutionPolicy Bypass -File VAE/specialized_training_scripts/RUN_FULL40_RECON_THEN_HEADS.ps1
```
