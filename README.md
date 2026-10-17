# Tigers and Goats

A reinforcement learning project using Maskable PPO. Play the game, evaluate pretrained policies, and train goat and tiger agents.

## Start here

Follow the [quickstart](Important_Docs/QUICKSTART.md) to install dependencies, play, evaluate a model, and run a short training experiment.

## Capabilities

- Placement and movement phases with legal-action masking.
- Goat and tiger training, scripted and model opponents, and opponent mixing.
- Configurable handcrafted rewards, experiment variations, and curricula.
- Checkpoints, resumed training, TensorBoard metrics, and run metadata.
- Single-model evaluation and matchup sweeps with CSV/JSON exports.
- GUI play as either side, player versus player, model assistance, and replays.

## Project files

| Path | Purpose |
| --- | --- |
| `env_tng.py` | Game rules, environment, action masks, and rewards |
| `train.py` | PPO training and experiment runner |
| `eval.py` | Evaluation and matchup sweeps |
| `gui.py` | Tkinter game and replay viewer |
| `model_summary.py` | Policy architecture inspection |
| `stable_models/models/` | Pretrained goat and tiger policies |
| `artifacts/` | Generated training and evaluation outputs |

## Documentation

- [Quickstart](Important_Docs/QUICKSTART.md)
- [Training guide](Important_Docs/TRAIN_SCRIPT_DETAILED_GUIDE.md)
- [Variations and curricula](Important_Docs/VARIATION_CONFIGS.md)
- [Reward reference](Important_Docs/REWARD_REFERENCE.md)
- [Network architecture](docs/NETWORK_ARCHITECTURE_AND_MODEL_COMBINER.md)

The original ECE445 research deliverable is preserved on branch `445`.
