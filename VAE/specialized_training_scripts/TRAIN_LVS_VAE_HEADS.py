# ============================================================
#  Project    : Tigers & Goats
#  Module     : Frozen-Body LVS-VAE Value Head Trainer Entrypoint
#  File       : TRAIN_LVS_VAE_HEADS.py
#
#  Purpose / Goal:
#    Load a reconstruction-only LVS-VAE body, freeze the latent space,
#    and train only the outcome value head.
# ============================================================
from __future__ import annotations

import TRAIN_VAE as trainer


trainer.RUN_NAME = "full_40k_frozen_lvsvae_value_head_v1"
trainer.DATA_PATH = (
    trainer.PROJECT_ROOT
    / "VAE_DATA"
    / "full_40k_40_40_20"
    / "states_labels.npz"
)
trainer.TRAIN_MODE = "value_head"
trainer.INIT_CHECKPOINT_PATH = (
    trainer.PROJECT_ROOT
    / "artifacts"
    / "lvs_vae"
    / "full_40k_reconstruction_lvsvae_v1"
    / "full_40k_reconstruction_lvsvae_v1_final_model.pt"
)
trainer.BEST_MODEL_METRIC = "total"
trainer.VALUE_WEIGHT = 1.0


if __name__ == "__main__":
    trainer.main()
