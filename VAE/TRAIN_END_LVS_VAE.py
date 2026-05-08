# ============================================================
#  Project    : Tigers & Goats
#  Module     : Endgame LVS-VAE Trainer Entrypoint
#  File       : TRAIN_END_LVS_VAE.py
#
#  Purpose / Goal:
#    Train the endgame LVS-VAE using the filtered endgame dataset.
# ============================================================
from __future__ import annotations

import TRAIN_VAE as trainer


trainer.RUN_NAME = "end_lvsvae_v1"
trainer.DATA_PATH = trainer.DEFAULT_ENDGAME_DATA_PATH


if __name__ == "__main__":
    trainer.main()
