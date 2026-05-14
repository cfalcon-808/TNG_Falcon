# ============================================================
#  Project    : Tigers & Goats
#  Module     : Reconstruction-Only LVS-VAE Trainer Entrypoint
#  File       : TRAIN_RECONSTRUCTION_LVS_VAE.py
#
#  Purpose / Goal:
#    Train only the LVS-VAE encoder/decoder latent body with reconstruction
#    and KL losses. Prediction heads are frozen and receive no loss.
# ============================================================
from __future__ import annotations

import TRAIN_VAE as trainer


trainer.RUN_NAME = "full_40k_reconstruction_lvsvae_v1"
trainer.DATA_PATH = (
    trainer.PROJECT_ROOT
    / "VAE_DATA"
    / "full_40k_40_40_20"
    / "states_labels.npz"
)
trainer.TRAIN_MODE = "reconstruction"
trainer.INIT_CHECKPOINT_PATH = None
trainer.BEST_MODEL_METRIC = "reconstruction"
trainer.VALUE_WEIGHT = 0.0


if __name__ == "__main__":
    trainer.main()
