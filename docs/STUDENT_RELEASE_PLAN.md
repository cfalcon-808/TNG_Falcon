# Student release plan

## Agreed direction

Preserve the ECE445 deliverable on branch `445`. Prepare a student-centered
`main` with all existing non-VAE capabilities. Once that version is validated,
consider exporting it into a fresh repository as release `v1.0.0`.

The fresh repository is a later decision, not part of the initial cleanup.

## Starting point

- Local branch `445` preserves commit `aaf7c40` (Final Delivarable for ECE445).
- `final-project-deliverable` also points to that commit.
- At inspection, local `main` is three commits behind, with no divergent commits.
- An existing uncommitted whitespace edit in `gui_abc.py` remains untouched.
- Branches preserve tracked commits, not ignored files or uncommitted changes.
- No cleanup or remote publication has been performed during planning.

## 1. Preserve and establish a baseline

- Preserve any needed ignored datasets or local research outputs separately.
- Publish the archival branch when ready to back it up remotely.
- Start a cleanup branch from the preserved deliverable; integrate the reviewed
  result into `main` after verification.
- Inventory capabilities and run a small non-VAE baseline before editing.
- Review general analysis notebooks present on the older `main`: retain useful
  non-VAE analysis workflows where appropriate instead of overlooking them
  because the final deliverable removed them.

## 2. Remove VAE functionality

- Remove VAE imports, configuration, checkpoint loading, reward wrappers,
  startup checks, logging, metadata, and ablation presets from `train_abc.py`.
- Remove VAE code, notebooks, datasets, checkpoints, and documentation from
  the student tree. They remain available in archival history.
- Preserve handcrafted rewards and the general reward customization hook.
- Check all retained entry points for references to removed paths.
- Keep behavior changes separate from broad file renaming or module splitting.

## 3. Preserve the non-VAE feature set

Use this as the acceptance checklist, verifying actual behavior:

- Game rules, placement and movement, legal-action masking, and end conditions.
- Goat and tiger learning; scripted and model-based opponents.
- Opponent mixing, experiment variations, and multi-phase curricula.
- Checkpoints, saved-model loading, and resumed training.
- TensorBoard metrics and experiment metadata.
- Single-model evaluation and evaluation sweeps with exported results.
- GUI play as either side, player versus player, model assistance, and replay.
- Model architecture inspection and curated pretrained goat/tiger policies.

## 4. Build the student experience

The first-run sequence should be:

1. Install into a clean environment.
2. Play a game in the GUI.
3. Evaluate a supplied model.
4. Complete a short CPU training run.
5. Change one reward or opponent setting and compare results.

- Replace the VAE-centered README and quickstart with this sequence.
- Provide small CPU defaults, with explicit larger/GPU experiment examples.
- Provide named examples for basic goat training, tiger training, curricula,
  opponent mixing, resume, and evaluation sweeps.
- Explain observations, actions, rewards, outcomes, and the main source files.
- Consolidate useful documentation into `docs/` and repair links.
- Separate normal generated outputs from curated example models; ignore new
  run outputs in Git.
- Keep ECE445 research records in the archive; retain applicable attribution
  and decide on a suitable reuse license before publishing a new repository.

## 5. Verify and integrate into main

- Test installation and documented commands in a fresh environment.
- Verify rules, masks, and terminal/truncated behavior with focused checks.
- Run short training for both learner roles, then save, load, and resume.
- Exercise curriculum transitions and opponent mixing.
- Evaluate supplied models and a freshly trained model; exercise sweep export.
- Manually verify GUI play and replay.
- Confirm non-VAE workflows need no archived VAE assets.
- Record tested dependency versions and any platform limitations.
- Integrate the verified cleanup into `main`; keep `445` as the archive.

Preserving workflows does not imply identical learned policies or performance
after removal of VAE reward shaping.

## 6. Optional fresh repository and version 1.0

- Select the validated student commit and record its source repository/commit.
- Export only intended tracked student files, excluding the old `.git`, local
  environments, caches, research data, and generated outputs.
- Retain relevant attribution and license information.
- Initialize fresh history, repeat the fresh-install checks, and publish under
  the chosen repository name and visibility.
- Tag the validated initial release `v1.0.0`.
- Make the new repository the authoritative student version and document how
  shared fixes will move between the student and research repositories.

Deleting files on this repository's `main` does not remove old objects from its
history. The later export is what gives the new repository fresh history.

## Decisions to settle during cleanup

- Supported student platforms and Python version.
- Which pretrained models and general analysis notebooks to ship.
- Whether to retain current script names initially or introduce clearer names
  after behavior is verified.
- New repository name, visibility, ownership, and license, if proceeding to v1.0.
