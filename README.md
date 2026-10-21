# Tigers and Goats

A reinforcement learning project using Maskable PPO. Play the game, evaluate pretrained policies, and train goat and tiger agents.

## Start here

Follow the [quickstart](Important_Docs/QUICKSTART.md) to install dependencies, play, evaluate a model, and run a short training experiment.

For Colab or Jupyter, open the standalone [training and evaluation notebook](TnG_Notebook.ipynb). Edit the configuration cell near the top, then run the notebook in order. It includes the environment, training, and a separate final evaluation section; no repository clone is required.

**Version 1.0 · Final preparation: October 20, 2026**

An experiment contains named `TRAINING_RUNS`. Each run config defines an independent training job; phase configs continue the same model through stages. See the [run config reference](Important_Docs/TRAINING_RUN_CONFIGS.md) for notebook/script differences.

[Future Plans](Important_Docs/FUTURE_PLANS.ipynb) covers proposed saved Tiger training, opponent pools, sticky sampling, and randomized starts. These are planned features, not current training options.

## Capabilities

- Placement and movement phases with legal-action masking.
- Goat and tiger training, scripted and model opponents, and opponent mixing.
- Configurable handcrafted rewards, experiment training runs, and curricula.
- Checkpoints, resumed training, TensorBoard metrics, and run metadata.
- Single-model evaluation and matchup sweeps with CSV/JSON exports.
- GUI play as either side, player versus player, model assistance, and replays.

## Project files

| Path | Purpose |
| --- | --- |
| `env_tng.py` | Game rules, environment, action masks, and rewards |
| [`TnG_Notebook.ipynb`](TnG_Notebook.ipynb) | Standalone Colab/Jupyter environment, training, evaluation, and interactive play |
| `train.py` | PPO training and experiment runner |
| `eval.py` | Evaluation and matchup sweeps |
| `gui.py` | Tkinter game and replay viewer |
| `model_summary.py` | Policy architecture inspection |
| `requirements.txt` | Python dependencies needed to run the project |
| `.gitignore` | Shared rules that keep generated outputs and local files out of Git |
| `Important_Docs/` | Student setup, training guides, and the shared progress notebook |
| `docs/` | Technical references for the code and policy architecture |
| `stable_models/` | Saved reference policies and historical training logs |
| `artifacts/` | Local training and evaluation outputs; ignored by Git |

## Important_Docs: guides and progress

Start here when setting up the project or planning a training experiment.

| Document | What it covers |
| --- | --- |
| [Quickstart](Important_Docs/QUICKSTART.md) | Installation, playing the game, evaluating models, and starting training |
| [Training guide](Important_Docs/TRAIN_SCRIPT_DETAILED_GUIDE.md) | Training settings, opponents, logging, checkpoints, and the training workflow |
| [Training runs and curricula](Important_Docs/TRAINING_RUN_CONFIGS.md) | Configuring experiment training runs and training across multiple phases |
| [Reward reference](Important_Docs/REWARD_REFERENCE.md) | Reward components and settings for designing experiments |
| [Robust goat training strategy](Important_Docs/ROBUST_GOAT_TRAINING_STRATEGY.txt) | Experiment notes on continued training and mixing Greedy and Smart Tiger opponents |
| [Versions and progress](Important_Docs/VERSIONS_AND_PROGRESS.ipynb) | Shared version history and a simple template for students to record changes, results, and next steps |
| [Future plans](Important_Docs/FUTURE_PLANS.ipynb) | Proposed opponent pools, sticky opponent sampling, and randomized starting states, with complexity and success checks |

## docs: technical references

Use these when you want to understand how the code and models work.

- [API index](docs/API_Index.md): reference to the main scripts, classes, functions, and command-line entry points.
- [Network architecture and model combiner](docs/NETWORK_ARCHITECTURE_AND_MODEL_COMBINER.md): explanations of the policy network and discussion of combining models.

## stable_models: reference models and logs

This folder contains existing policies students can evaluate, use in the GUI, or use as starting points for continued training.

| Folder | Contents |
| --- | --- |
| `stable_models/models/Goats/` | Saved goat policies in `.zip` format |
| `stable_models/models/Tigers/` | Saved tiger policies in `.zip` format |
| `stable_models/Logging/` | Historical TensorBoard logs from benchmark goat and tiger training |

Existing model filenames retain their original names, including older labels such as `NormalGoat`. The current scripted tiger opponents are called **Greedy** and **Smart**.

To view the historical logs, run this from the repository root:

```powershell
tensorboard --logdir stable_models/Logging --port 6006
```

## artifacts: your experiment outputs

New training runs save their configuration metadata, TensorBoard logs, checkpoints, and models under `artifacts/<experiment_name>/`. Evaluation also writes results under `artifacts/`.

Give each experiment a distinct name so you can identify its results. This folder is ignored by Git, so your outputs stay on your computer and are not included in commits. Record a short summary in the progress notebook and share model files separately when teammates need them.

The original ECE445 research deliverable is preserved on branch `445`.
