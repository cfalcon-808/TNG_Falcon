<!-- Copilot instructions for TNG_Falcon repository -->
# Copilot Instructions — Tigers & Goats (Falcon branch)

This file gives concise, repository-specific guidance for automated coding agents.

- Entry points: prioritize changes under `train_falcon.py`, `eval_falcon.py`, and `env_tng_falcon.py`.
- Quick setup: see `docs/QUICKSTART.txt` — use a virtualenv and `pip install -r docs/requirements.txt`.

**Architecture (big picture)**
- RL pipeline: `env_tng_falcon.py` → `train_falcon.py` (EXPERIMENT → VARIATIONS) → `eval_falcon.py` / `tng_GUI_falcon.py`.
- `env_tng_falcon.py` contains both greedy and smart tiger logic (switch via `TIGER_AI_MODE`) and exposes `get_action_mask()` for MaskablePPO.

**Key patterns & conventions**
- Edit experiment metadata (`EXPERIMENT_NAME`, `TIGER_AI_MODE`, `VARIATIONS`) in `train_falcon.py`.
- `VARIATIONS` supports dict (single-phase) or list (multi-phase curricula). See `docs/README.txt` for examples.
- Artifacts: `artifacts/models/train/{ALGO}/{CORE}/{EXPERIMENT}/` and filenames like `mppo_{CORE}_{variation}_p{phase}.zip`.
- CORE tags: `GvNT`, `GvST` — preserve these in experiment names.

**Common workflows (explicit commands)**
- Setup (PowerShell):

  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r docs/requirements.txt
  ```

- Run training (use `DEBUG_MODE = True` for smoke tests):

  ```powershell
  python train_falcon.py
  tensorboard --logdir artifacts/logging/train --port 6006
  ```

- Evaluate a model (explicit path recommended):

  ```powershell
  python eval_falcon.py battle artifacts/models/train/.../mppo_GvST_defaultSettings_p0.zip
  ```

**Agent-specific guidance**
- Uses MaskablePPO and action masks. Use `get_action_mask()` and `eval_falcon.py`'s mask-inspection utilities to debug illegal actions.
- Hyperparameter baselines live in `docs/HyperparameterGuide.txt`; tune `NUM_CPU`, `N_STEPS`, and `ENT_COEF` first.

**Where to look for more context**
- `docs/QUICKSTART.txt`, `docs/README.txt`, and `train_falcon.py` for VARIATIONS/phase examples.
- `artifacts/` for save/log conventions.

If you'd like, I can add example `VARIATIONS` snippets, a smoke-run helper script, or merge an older `copilot-instructions.md` from the branch you pulled. Reply which you prefer.
