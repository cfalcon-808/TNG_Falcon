# Training Goat Learners Against Saved Tiger Models — Future Upgrade

**Status:** Proposed; deferred until after the Version 1 notebook release.  
**Date:** 2026-10-17

## Objective and scope

Let students train fresh or resumed goat policies against frozen saved Tiger policies, then evaluate model-versus-model performance. Implement this later in both `train.py` and `TnG_Notebook.ipynb`, with compatible evaluation.

Use one fixed Tiger model per training phase. Curriculum phases may select different saved models. Generalized opponent mixing, model pools, simultaneous training of both sides, and automatic self-play or opponent replacement are deferred.

The current Version 1 notebook will be published separately. This document is a future implementation specification, not a claim that Model Tiger training is already available.

## Existing capabilities

- `env_tng.py` already accepts `tiger_ai="model"` and a `tiger_model_predict_fn` callback. The environment constructs a legal Tiger action mask and maps the predicted action to a legal move.
- The repository's `eval.py` already has Model Tiger prediction helpers and model-versus-model single and sweep evaluation paths.
- The training script and notebook currently expose Greedy and Smart Tigers for goat training. Their training configuration and model-loading paths still need the proposed extension.

## Proposed implementation

### 1. Configuration

- Add `TIGER_MODEL_PATH = None` beside the existing Goat model path.
- Allow notebook settings `LEARNER_ROLE = "goat"` and `OPPONENT = "model"`.
- Add `OPP_TIGER_MODEL = "tiger_model"` to the script's opponent selector.
- Support `tiger_model_path` overrides in training runs and curriculum phases; inherit the global path when omitted.
- Require `MIX_PROB = None` for Model Tiger phases. Reject Model Tiger plus mixing explicitly, while preserving existing scripted Tiger and Random/Model Goat mixing.
- Preserve current default matchups and keep notebook settings in its single configuration cell, with basic settings first.

### 2. Training

- Reuse the existing environment callback without changing game rules, reward behavior, or action encoding.
- Load the MaskablePPO Tiger opponent once per environment instance and use deterministic predictions with the Tiger legal-action mask.
- Keep opponent weights frozen; never update or overwrite the opponent checkpoint.
- Preserve sequential notebook environments and subprocess workers in the script. Load models inside each worker rather than passing loaded models between processes.
- Support fresh and resumed goat learners and continuation of the same learner across curriculum phases, including changes between Tiger checkpoints.
- Close environments on completion, failure, and interruption.

### 3. Validation and failures

- Validate all requested Tiger model files before starting training: require an existing `.zip`, successful loading, and compatible observation/action spaces.
- Explain that matching spaces cannot prove a policy was trained as a Tiger. Students must select a Tiger policy compatible with the current game environment; do not reject legacy models solely because role metadata is absent.
- Report missing, corrupt, incompatible, or invalid-action opponents explicitly. Never silently replace a requested Model Tiger with a scripted opponent.

### 4. Evaluation

- Extend notebook evaluation to accept `"model"` for goat learners.
- By default, evaluate against the final training phase's opponent and Tiger checkpoint. Allow an explicit `TIGER_MODEL_PATH` override when an opponent file has been moved or uploaded elsewhere.
- Support explicit comparisons against Greedy, Smart, and Model Tigers, including evaluation of an existing goat model without retraining.
- Verify the script's existing single and sweep model-versus-model evaluation, changing only what is needed for validation and consistent reporting.

### 5. Metadata and naming

- Record the effective Tiger opponent path and file SHA-256 in training and evaluation metadata so results identify the exact checkpoint.
- Treat absent `tiger_model_path` fields in older settings files as `None`; require a path only when the selected matchup uses a Model Tiger.
- Use **Model Tiger** in descriptions and `GvMT` for fixed Model Tiger training tags.
- For curricula involving Model Tigers, construct tags from actual opponent families in stable Greedy/Smart/Model order. Preserve existing tags for existing combinations.

### 6. Student documentation

- Update notebook instructions, inline options, overview tables, and evaluation guidance, plus repository training/configuration documentation and README capability descriptions.
- Provide commented examples for fixed Model Tiger training and a Greedy → Smart → Model Tiger curriculum.
- Explain the manual improvement cycle: train and save a Tiger, train a Goat against that frozen Tiger, evaluate the Goat, then optionally use the saved Goat as a frozen opponent for subsequent Tiger training.
- Explain that improvement against one fixed opponent does not establish generalization. Compare outcomes against scripted and saved opponents separately; generalized strategy experiments remain future work.

## Test and acceptance plan

- Complete short goat-training runs against a saved Tiger policy in both the notebook and script, including a subprocess-worker run.
- Verify opponent predictions receive Tiger masks and return legal actions. Confirm opponent weights and checkpoint bytes remain unchanged.
- Exercise fresh training, resumed goat training, scripted-to-model curricula, and switching between two saved Tiger policies.
- Test final-phase default evaluation, explicit Greedy/Smart/Model comparisons, and existing-model evaluation without training.
- Test missing and corrupt files, incompatible spaces, invalid predicted actions, and rejection of Model Tiger mixing.
- Check global/phase path inheritance, old metadata compatibility, opponent hashes, matchup tags, checkpoint saving, and exported results.
- Run existing notebook regression checks and execute the optional Model Tiger configuration in a fresh kernel.
- Report Colab, GPU, or other platform checks explicitly as verified or untested; do not infer their success from local CPU tests.

## Delivery boundaries

This documentation task creates only this Markdown plan. It does not modify code, the Version 1 notebook, README, or configuration; train models; commit; push; or tag a release.

Implementation is a separate future task. Before considering this document complete, verify that it records the agreed scope, requirements, test cases, deferred features, and distinction between existing environment hooks and proposed training support.
