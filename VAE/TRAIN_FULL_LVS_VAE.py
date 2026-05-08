# ============================================================
#  Project    : Tigers & Goats
#  Module     : Full-Game LVS-VAE Trainer Entrypoint
#  File       : TRAIN_FULL_LVS_VAE.py
#
#  Purpose / Goal:
#    Train the baseline full-game LVS-VAE using the full generated dataset.
# ============================================================
from __future__ import annotations

import TRAIN_VAE as trainer


trainer.RUN_NAME = "full_lvsvae_v1"
trainer.DATA_PATH = trainer.DEFAULT_DATA_PATH


if __name__ == "__main__":
    trainer.main()
