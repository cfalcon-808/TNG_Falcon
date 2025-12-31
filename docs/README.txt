TIGERS & GOATS — FALCON BRANCH
Reinforcement Learning Environment, Training Pipeline,
Experiment Sweeps, and Evaluation Toolkit
===================================================================
***Read QUICKSTART.txt for environment setup and dependencies***
===================================================================

Overview
--------
This repository implements a reinforcement-learning framework for the
Tigers & Goats (Bagh-Chal) board game. It supports training and evaluating
agents using MaskablePPO with action masking, reward shaping, and
multi-process environments.

The system is organized around the concept of:
- an EXPERIMENT: a suite of related training runs
- VARIATIONS: individual runs within an experiment

All training uses a single unified environment file that can switch
between greedy and smart tiger opponents without swapping code.


Core Components
---------------
- env_tng_falcon.py
    Unified Gymnasium environment for Tigers & Goats.
    Supports:
      • Greedy tiger AI
      • Smart tiger AI
      • Reward shaping via knobs + weights
      • Action masking

- experiment_sweep.py
    Training runner for MaskablePPO.
    Runs one EXPERIMENT consisting of multiple VARIATIONS
    (each variation = one PPO training run with different reward overrides).

- eval_falcon.py
    Post-training evaluation and debugging tool.
    Supports:
      • Deterministic rollouts
      • Batch evaluation
      • ASCII board rendering
      • Action mask inspection
      • Win/timeout statistics


Project Structure
-----------------
.
├── env_tng_falcon.py          # unified greedy / smart tiger environment
├── experiment_sweep.py       # PPO training + ablation sweeps
├── eval_falcon.py             # evaluation + debug
├── artifacts/
│   ├── models/
│   │   └── train/
│   │       └── {ALGO}/{CORE}/{EXPERIMENT_NAME}/
│   └── logging/
│       └── train/
│           └── {ALGO}/{CORE}/{EXPERIMENT_NAME}/
├── docs/                      # design notes and guides
├── QUICKSTART.txt
└── .venv/                     # local virtualenv (ignored by git)


Key Concepts
------------
EXPERIMENT
    A collection of related training runs (variations).
    Defined by:
      • EXPERIMENT_NAME
      • Tiger opponent type
      • Algorithm (e.g. MaskablePPO)

VARIATION
    A single PPO training run within an experiment.
    Each variation differs only by reward overrides
    (knobs and/or weights).

CORE TAG
    Encodes the learner/opponent matchup:
      • GvNT  → Goat vs Normal (greedy) Tiger
      • GvST  → Goat vs Smart Tiger
      • (future) TvNG / TvSG for tiger-learning agents

ALGO TAG
    Short algorithm identifier (e.g. "mppo").


Training Pipeline (experiment_sweep.py)
---------------------------------------
- MaskablePPO with MLP policy: [256, 256, 256]
- Parallel environments via SubprocVecEnv
- Action masking to enforce legal moves
- Shared hyperparameters across all variations
- Per-variation TensorBoard logging
- Periodic checkpoints + final model saves

Artifact naming:
- Checkpoints:
    cp_{ALGO}_{CORE}_{variation}_*.zip
- Final model:
    {ALGO}_{CORE}_{variation}.zip
  (auto-suffixed to avoid overwrites)


Running Training
----------------
1) Configure experiment metadata:
      EXPERIMENT_NAME
      TIGER_AI_MODE
      VARIATIONS

2) Run the experiment:
      python experiment_sweep.py

3) Monitor training:
      tensorboard --logdir artifacts/logging/train --port 6006


Example VARIATIONS block:
-------------------------
VARIATIONS = {
    "defaultSettings": None,
    "no_bubble": {"bubble": 0.0},
    "no_block": {"block_tiger": 0.0},
    "lower_step": {"REWARD_STEP": -0.001},
}


Evaluation & Debugging (eval_falcon.py)
---------------------------------------
Supports two main modes:

1) Debug mode
    - Step-by-step episode playback
    - ASCII board rendering
    - Action decoding + masks
    - Reward and termination inspection

2) Batch evaluation
    - Deterministic rollouts
    - Goat/tiger win rates
    - Timeout and stall statistics
    - Average episode length and reward

Typical usage:
    python eval_falcon.py artifacts/models/train/.../mppo_GvST_defaultSettings.zip


Action Masking
--------------
- Environment internally tracks legal (position, direction) moves
- Actions are flattened into a Discrete space
- get_action_mask() exposes legality
- MaskablePPO guarantees no illegal actions are sampled


Versioning Notes
----------------
- Git history is the authoritative source of truth
- Version tags in headers (env_4.0, mppo_exp2.0, etc.)
  are human-readable milestones, not strict releases
- Experiment names + variation tags uniquely identify runs


Intended Use
------------
This framework is designed for:
- Reward shaping ablation studies
- Strategy analysis against fixed opponents
- Comparing greedy vs smart adversaries
- Extensible to tiger-learning agents (future work)

NAMING CONVENTIONS (Quick Reference)
===================================

+-------------------+-------------------------------+-------------------------------+-----------------------------------------------+
| Concept           | Name / Pattern                | Example                       | Meaning                                       |
+-------------------+-------------------------------+-------------------------------+-----------------------------------------------+
| Algorithm tag     | {ALGO_TAG}                    | mppo                          | Learning algorithm (Maskable PPO)             |
| Core matchup      | {CORE}                        | GvST                          | Learner vs opponent                           |
| Experiment suite  | {EXPERIMENT_NAME}             | sparse_reward_analysis        | Collection of related training runs           |
| Variation         | {variation}                   | no_bubble                     | One training run inside an experiment         |
| Checkpoint file   | cp_{ALGO}_{CORE}_{variation}* | cp_mppo_GvST_no_bubble_3.zip  | Periodic checkpoint during training           |
| Final model file  | {ALGO}_{CORE}_{variation}.zip | mppo_GvST_defaultSettings.zip | Final trained policy                          |
| TensorBoard run   | {ALGO}_{CORE}_{variation}     | mppo_GvST_no_block            | One variation’s TensorBoard stream            |
| Model directory   | {ALGO}/{CORE}/{EXPERIMENT}    | mppo/GvST/sparse_reward_...   | All models for one experiment                 |
| Log directory     | {ALGO}/{CORE}/{EXPERIMENT}    | mppo/GvST/sparse_reward_...   | All logs for one experiment                   |
+-------------------+-------------------------------+-------------------------------+-----------------------------------------------+


CORE TAG MEANINGS
================

+------+----------------------------------------------+
| CORE | Description                                  |
+------+----------------------------------------------+
| GvNT | Goat learning vs Normal / Greedy Tiger       |
| GvST | Goat learning vs Smart Tiger                 |
| TvNG | (future) Tiger learning vs Normal Goat       |
| TvSG | (future) Tiger learning vs Smart Goat        |
+------+----------------------------------------------+

