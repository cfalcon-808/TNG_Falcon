# ============================================================
#  Project    : Tigers & Goats
#  Module     : LVS-VAE-PS Batch Runner
#  File       : RUN_JOINT_LVS_VAE_PS.py
#
#  Purpose / Goal:
#    Train placing-survival-only LVS-VAE-PS checkpoints across the standard
#    40/40/20 datasets, saving under artifacts/lvs_vae_ps.
# ============================================================
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VAE_DIR = PROJECT_ROOT / "VAE"
if str(VAE_DIR) not in sys.path:
    sys.path.insert(0, str(VAE_DIR))

import TRAIN_PLACING_SURVIVAL_LVS_VAE as trainer


RUNS = (
    {
        "run_name": "end_20k_joint_ps_v1",
        "data_dir": "end_20k_40_40_20",
    },
    {
        "run_name": "end_40k_joint_ps_v1",
        "data_dir": "end_40k_40_40_20",
    },
    {
        "run_name": "full_20k_joint_ps_v1",
        "data_dir": "full_20k_40_40_20",
    },
    {
        "run_name": "full_40k_joint_ps_v1",
        "data_dir": "full_40k_40_40_20",
    },
)


def configure_trainer(run_name: str, data_dir: str) -> None:
    """Apply one batch-run config to the shared PS trainer module."""
    trainer.RUN_NAME = run_name
    trainer.DATA_PATH = trainer.PROJECT_ROOT / "VAE_DATA" / data_dir / "states_labels.npz"
    trainer.OUTPUT_ROOT = trainer.PROJECT_ROOT / "artifacts" / "lvs_vae_ps"
    trainer.FILTER_PLACING_PHASE = True
    trainer.PLACING_SURVIVAL_WEIGHT = 1.0


def run_batch(skip_existing: bool) -> None:
    """Run each LVS-VAE-PS training job in sequence."""
    for run in RUNS:
        run_name = run["run_name"]
        data_dir = run["data_dir"]
        final_checkpoint = (
            PROJECT_ROOT
            / "artifacts"
            / "lvs_vae_ps"
            / run_name
            / f"{run_name}_final_model.pt"
        )

        if skip_existing and final_checkpoint.exists():
            print(f"Skipping existing run: {run_name}", flush=True)
            continue

        print("=" * 60, flush=True)
        print(f"Training LVS-VAE-PS run: {run_name}", flush=True)
        print(f"Dataset: VAE_DATA/{data_dir}/states_labels.npz", flush=True)
        print(f"Output: artifacts/lvs_vae_ps/{run_name}", flush=True)
        print("=" * 60, flush=True)

        configure_trainer(run_name=run_name, data_dir=data_dir)
        trainer.main()

    print("All LVS-VAE-PS runs complete.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose final checkpoint already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(skip_existing=args.skip_existing)
