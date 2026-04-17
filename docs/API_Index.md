# API Index

This index reflects the current repo-root `*_abc.py` files. Older `*_falcon.py` names still appear in some comments and historical docs, but they are not the active entry points.

# `env_tng_abc.py`

Unified Gymnasium environment for Tigers and Goats with native `Discrete(115)` actions, role-aware masking, and configurable scripted or model-driven opponents.

### Constants and enums

- `BOARD_SIZE = 23`
- `DIR_CODES = 5`
- `TOTAL_GOATS_TO_PLACE = 15`
- `GOATS_EATEN_FOR_TIGER_WIN = 6`
- `TIGER_START_POSITIONS = [0, 3, 4]`
- `KEY_CENTERS = [0, 9, 10, 15, 16]`
- `TIGER_AI_GREEDY = "greedy"`
- `TIGER_AI_SMART = "smart"`
- `TIGER_AI_MODEL = "model"`
- `VALID_TIGER_AI = {"greedy", "smart", "model"}`
- `GOAT_LEARNER = "goat"`
- `TIGER_LEARNER = "tiger"`
- `GOAT_AI_RANDOM = "random"`
- `GOAT_AI_MODEL = "model"`

### Tuning

- `DEFAULT_KNOBS`
  Base reward and shaping configuration.
- Main knob families:
  terminal rewards, tiger capture rewards, timeout scaling, move-step penalties, center and mobility shaping, repeat penalties, tiger capture bias.
- `reward_weights`
  Optional constructor override map. Known keys overwrite `DEFAULT_KNOBS`.
- `max_turns`
  Explicit constructor override that also synchronizes `MAX_TURNS`.

### Top-level helper

- `idx_to_coord(idx: int) -> str`
  Converts a board index to its human-readable coordinate label.

### Class `TnGEnv`

Full environment with:

- placing phase and moving phase
- goat learner and tiger learner modes
- greedy, smart, or model tiger opponents
- random or model goat opponents
- native flat action encoding
- action masking for `MaskablePPO`
- repeat-state tracking and truncation handling
- reward hook via `reward_fn`

#### Constructor

- `__init__(reward_weights=None, tiger_ai=TIGER_AI_GREEDY, learner_role=GOAT_LEARNER, goat_opponent_ai=GOAT_AI_RANDOM, goat_model_predict_fn=None, tiger_model_predict_fn=None, reward_fn=None, max_turns=None)`
  Configures knobs, role, opponents, action and observation spaces, board topology, and cached geometry helpers.

#### Public Gym API

- `reset(seed=None, options=None)`
  Resets board state, counters, repeat tracking, and caches. Returns `(obs, info)`.
- `step(action)`
  Routes to goat-learner or tiger-learner transition logic, then applies the configured reward function.
- `get_state()`
  Returns the flattened observation vector: `board[23] + goats_eaten + phase`.
- `render(show_coords=True)`
  Prints an ASCII board with phase, goats eaten, turn count, and last-move text.
- `get_action_mask()`
  Returns a role-aware legal-action mask of shape `(115,)`.

#### Action helpers

- `encode_action(pos: int, dir_code: int) -> int`
  Encodes `(pos, dir)` into the flat action id.
- `decode_action(action)`
  Decodes a flat action id, and also accepts legacy 2-item action vectors.
- `_action_to_flat(action)`
  Best-effort normalization into a flat action id.
- `_safe_decode_action(action)`
  Defensive decode helper used by transitions.

#### Info and reward pipeline

- `_build_step_info(**extra)`
  Builds the standard info payload.
- Standard info fields include:
  `reason`, `winner`, `goat_action`, `goat_decoded`, `tiger_move`, `phase`, `goats_eaten`, `goats_placed`, `turn_counter`, `learner_role`, `action_mask`, opponent identifiers, and reward breakdown fields.
- `_set_reward_components(info, **components)`
  Writes scalar reward breakdown values into `info`.
- `_apply_reward_fn(prev_obs, action, result)`
  Applies `reward_fn` while preserving compatibility with older callback signatures.
- `sparse_reward(...)`
  Dispatches to the role-specific reward builder.
- `_sparse_reward_goat(...)`
  Goat-learner shaping and terminal reward logic.
- `_sparse_reward_tiger(...)`
  Tiger-learner capture, mobility, center-control, and terminal reward logic.

#### Transition engines

- `_step_goat_transition(action)`
  Applies a goat turn, then a tiger reply, then emits a structured transition summary.
- `_step_tiger_transition(action)`
  Applies a tiger turn, then a goat reply, then emits a structured transition summary.
- Invalid actions and timeouts return `truncated=True`.
- Game outcomes return `terminated=True`.

#### Opponent selection and move generation

- `_select_tiger_move(...)`
  Dispatches to greedy, smart, or model tiger control.
- `_model_tiger_move(...)`
  Uses `tiger_model_predict_fn(obs, mask)` and maps the chosen flat action back to a legal tiger move.
- `_smart_tiger_move(...)`
  Heuristic tiger policy with capture priority, anchor control, and positional roaming.
- `_greedy_tiger_move(...)`
  Capture-biased random tiger policy controlled by `BASE_TIGER_CAPTURE_BIAS`.
- `_tiger_moves(include_dir: bool = False)`
  Source of truth for legal tiger moves and captures.
- `_update_valid_moves()`
  Recomputes legal goat actions for the current phase.

#### Board metrics and graph helpers

- `_find_jumped_goat(t_from: int, t_to: int)`
  Resolves the jumped goat for tiger captures.
- `_compute_unreachable_safe_cells()`
  Estimates goat-safe bubble territory.
- `_precompute_all_pairs_dist()`
  Precomputes shortest-path distances on the board graph.
- `_shortest_path_len(start: int, goal: int) -> int`
  Distance lookup into the cached graph matrix.
- `_tiger_spread() -> float`
  Sum of pairwise tiger distances.
- `_w(key) -> float`
  Strict knob lookup helper.

# `eval_abc.py`

Evaluation runner for both single-matchup debug/eval flows and full goat-model versus tiger-model sweeps.

### CLI and config

- `parse_cli_args()`
  Supports the legacy convenience CLI patterns:
  `python eval_abc.py`
  `python eval_abc.py model.zip`
  `python eval_abc.py normal`
  `python eval_abc.py battle`
  `python eval_abc.py battle model.zip`
- `EVAL_MODE`
  `"single"` or `"sweep"`.
- `SINGLE_CONFIG`
  Configures one learner model against one opponent setup.
- `SWEEP_CONFIG`
  Configures goat-model and tiger-model permutation sweeps plus export settings.

### Logging utilities

- `class Tee`
  Mirrors stdout into one or more streams.
- `log(...)`
  Writes only to the log stream.
- `log_and_console(...)`
  Writes to both the log and the real console.
- `build_log_path(model_path, mode_tag="single")`
  Creates a debug log filename under `artifacts/logging/eval_debug/`.
- `setup_logging(log_path, log_tag="single")`
  Installs tee-based logging and returns the original stdout plus the open file handle.

### Model and environment helpers

- `load_latest_model_path(model_dir=MODELS_DIR)`
  Finds the newest `.zip` model in the target directory.
- `_mask_fn(env)`
  Delegates mask generation to `env.unwrapped.get_action_mask()`.
- `make_goat_model_predict_fn(model_path)`
  Lazy goat-model opponent loader for tiger-learner evaluation.
- `make_tiger_model_predict_fn(model_path)`
  Lazy tiger-model opponent loader for goat-learner evaluation.
- `describe_matchup(...)`
  Builds a human-readable matchup label for logs and summaries.
- `make_debug_env(...)`
  Builds `TnGEnv -> Monitor -> ActionMasker`.
- `make_eval_env(...)`
  Builds `TnGEnv -> ActionMasker`.
- `decode_flat_action(flat_action, dir_codes=DIR_CODES)`
  Utility for rendering flat policy outputs as `(pos, dir)`.

### Evaluation runners

- `run_debug_episodes(...)`
  Runs step-by-step deterministic or stochastic debug episodes with rendered boards and decoded actions.
- `classify_outcome(episode_reward, info) -> str`
  Uses `info["winner"]` first, then reward thresholds as a fallback.
- `run_batch_evaluation(...)`
  Runs `n_games` episodes, aggregates outcomes, and returns a metric dictionary.
- `save_sweep_results(results)`
  Writes JSON and CSV summaries under `artifacts/eval_sweeps/`.
- `run_sweep_mode()`
  Runs all configured goat-model x tiger-model permutations.
- `run_single_mode()`
  Resolves the active learner model, optional debug pass, then batch evaluation.

### Outputs

- Single-run debug logs:
  `artifacts/logging/eval_debug/`
- Sweep exports:
  `artifacts/eval_sweeps/sweep_results_<timestamp>.json`
  `artifacts/eval_sweeps/sweep_results_<timestamp>.csv`

# `train_abc.py`

MaskablePPO training runner for experiment suites made of one or more variations, where each variation can be a single phase or a multi-phase curriculum.

### Core user config

- `EXPERIMENT_NAME`
  Top-level experiment folder name under `artifacts/`.
- `LEARNER_ROLE`
  `GOAT_LEARNER` or `TIGER_LEARNER`.
- `OPPONENT_AI`
  Unified opponent selector:
  goat learner uses `tiger_greedy` or `tiger_smart`
  tiger learner uses `goat_random` or `goat_model`
- `MIX_PROB`
  Per-reset opponent mixing probability.
- `RESUME_MODEL_PATH`
  Optional resume source; accepts either a `.zip` file or a directory containing checkpoints.
- `GOAT_MODEL_PATH`
  Path for model-goat opponents when training the tiger learner.
- `DEVICE_MODE`, `TIMESTEPS`, `NUM_CPU`, `SEED`
  Scale and hardware controls.
- PPO hyperparameters:
  `GAMMA`, `LEARNING_RATE`, `ENT_COEF`, `N_STEPS`, `BATCH_SIZE`, `N_EPOCHS`.

### Variation schema

- `CONTROL_KEYS`
  Phase-level control keys: `timesteps`, `opponent_ai`, `mix_prob`, `checkpoints_per_run`, `goat_model_path`.
- `LEGACY_VARIATION_KEYS`
  Explicitly rejected older keys: `tiger_ai`, `reward_weights`.
- `@dataclass(frozen=True) PhaseConfig`
  Fields:
  `timesteps`, `opponent_ai`, `mix_prob`, `checkpoints_per_run`, `goat_model_path`, `knobs`.
- `default_phase_config()`
  Builds a `PhaseConfig` from the global script defaults.
- `normalize_phase_dict(...)`
  Validates a phase dict and splits control keys from reward knobs.
- `normalize_variation_config(...)`
  Accepts `None`, a single dict, or a list of dict phases.

### Matchup and naming helpers

- `core_tag(learner_role, opponent_ai, mix=False, goat_opponent_ai=None) -> str`
  Builds tags like `GvNT`, `GvST`, `GvMixT`, `TvRG`, or `TvMG`.
- `resolve_opponent_or_exit(learner_role, opponent_ai) -> tuple[str, str]`
  Converts the unified opponent selector into `(tiger_ai_mode, goat_opponent_ai)`.
- `resolve_variation_core(phases) -> str`
  Chooses the variation-level matchup tag across one or more phases.
- `_slugify(text)`
  Filesystem-safe experiment and variation naming helper.

### Model and mask helpers

- `make_goat_model_predict_fn(model_path)`
  Lazy-loads a goat opponent model for tiger-learner runs.
- `warn_if_missing_goat_model_path(goat_opponent_ai, model_path, context)`
  Warns when a model-goat opponent was requested but no model path was supplied.
- `resolve_phase_matchup(phase) -> tuple[str, str]`
  Resolves the phase-specific opponent modes.
- `checkpoint_save_freq(phase) -> int`
  Converts phase length and checkpoint count into per-worker save frequency.
- `mask_fn(env)`
  Exposes `env.unwrapped.get_action_mask()` to `ActionMasker`.

### Callback and wrapper classes

- `class WinStatsCallback(BaseCallback)`
  Tracks total and rolling-window outcome rates, realized mixed-opponent fractions, reward components, and episode reward summaries.
- `class TigerMixWrapper(Monitor)`
  Re-samples `tiger_ai` on every reset.
- `class GoatMixWrapper(Monitor)`
  Re-samples `goat_opponent_ai` on every reset.

### Environment factory

- `make_env(rank, seed, knobs, tiger_ai_mode=None, mix_prob=None, resolved_goat_ai=None, goat_model_path=None)`
  Builds one worker environment:
  `TnGEnv -> (Monitor or mix wrapper) -> ActionMasker`

### Resume, artifact, and metadata helpers

- `resolve_resume_path(path)`
  Resolves a resume target from a file or directory.
- `unique_filename(directory, base_name, ext=".zip")`
  Prevents output file collisions.
- `build_run_layout(experiment_name, artifacts_root="artifacts")`
  Creates:
  `meta/`, `tb/`, `checkpoints/`, `models/`, `eval/`, `logs/`
- `_git_commit_or_unknown()`
  Best-effort git hash capture for reproducibility.
- `write_run_metadata(layout, variation_name, variation_core, phases, run_stamp)`
  Writes config, git commit, and notes files under `meta/`.
- `log_matchup_info(model_obj, learner_role, tiger_ai, resolved_goat_ai, mix_prob)`
  Records matchup scalars into TensorBoard.
- `configure_model_for_phase(...)`
  Creates a fresh `MaskablePPO` model, resumes an existing one, or reattaches the same instance for later phases.

### Core runner

- `run_single_variation(variation_name, variation_config)`
  Main training entry point for one variation.
- Per phase it:
  resolves the matchup, builds `SubprocVecEnv`, installs callbacks, configures or resumes the model, trains, saves checkpoints, saves phase finals, and tracks the best phase model.
- Final return payload includes:
  variation name, episode count, goat and tiger rates, timeout rates, and the final saved model path.

### Main flow

- Prints the experiment header and resolved matchup tags for every variation.
- Iterates `VARIATIONS.items()`.
- Prints a final per-variation result table.

# `gui_abc.py`

Tkinter GUI for live play, model-assisted play, tiger-play mode, PvP mode, replay browsing, and replay recording.

### CLI

- `python gui_abc.py --env {normal|battle} [--model path/to/model.zip]`
- `--env normal`
  Starts with greedy tiger defaults.
- `--env battle`
  Starts with smart tiger defaults.
- `--model`
  Optional goat model to preload at startup.

### Top-level helpers

- `build_edges(move_map)`
  Converts move-map adjacency into unique undirected draw edges.
- `pack_action_flat(pos: int, dir_code: int) -> int`
  Encodes a flat action id matching the environment.
- `unpack_action_flat(a: int) -> tuple[int, int]`
  Decodes the flat action id.

### Class `ReplayTimeline`

- `__init__(source)`
  Loads replay JSON or an in-memory replay dict.
- `current_board()`
  Returns the board for the current replay index.
- `cumulative_reward(idx: int) -> float`
  Returns prefix reward total up to the requested step.
- `step(delta)`
  Moves the replay cursor with bounds clamping.
- `info_text()`
  Builds the replay status string used by the UI.

### Class `TigersGoatsGUI`

Primary window/controller class. The home screen switches between:

- goat-play mode
- tiger-play mode
- PvP mode
- replay mode

#### Initialization and view management

- `__init__(tiger_ai: str, model_path=None)`
  Creates the root window, stores initial model state, and opens the home view.
- `switch_view(view_name: str)`
  Swaps the active UI view and cleans up game state when needed.
- `_build_home_view()`
  Builds the mode-selection screen.
- `_build_game_view()`
  Builds the live game or replay screen.

#### Environment and runtime setup

- `_init_env_or_replay()`
  Chooses live env, PvP env, or replay setup.
- `_init_live_env()`
  Creates the normal live environment and optional model handles.
- `_init_pvp_env()`
  Creates the local PvP state.
- `_init_replay_env()`
  Loads replay-only state.
- `_init_runtime_state()`
  Resets transient GUI state such as selections, replay flags, and timers.

#### Drawing and overlays

- `_draw_static()`
  Draws the board graph, static labels, and UI shell.
- `_draw_board(board_override=None)`
  Draws pieces, labels, highlights, and board state.
- `_init_canvas_overlays()`
  Creates top-of-canvas counters and labels.
- `_update_canvas_overlays(...)`
  Refreshes turn, phase, blocked tiger count, winner text, and related status labels.

#### Labels, snapshots, and replay text

- `_init_live_labels()`
  Creates live piece labels.
- `_update_live_labels(before_board, after_board, action, phase)`
  Keeps piece IDs visually stable across transitions.
- `_snapshot_live_state()`
  Captures a replay-ready state snapshot.
- `_describe_transition(...)`
  Produces readable action descriptions for replay and live logs.

#### Model and replay loading

- `_load_model(path)`
  Loads a goat model for live goat-play mode.
- `_load_tiger_model(path)`
  Loads a tiger model for tiger-play mode.
- `load_model_from_entry()`, `browse_model()`
  Goat model UI loaders.
- `load_tiger_model_from_entry()`, `browse_tiger_model()`
  Tiger model UI loaders.
- `_load_replay(path)`
  Loads a replay file into the replay viewer.
- `load_replay_from_entry()`, `browse_replay()`
  Replay UI loaders.

#### Live play and replay actions

- `reset_env()`
  Resets the current live environment.
- `record_episode(max_steps=200)`
  Runs a fresh episode and stores a replay timeline in memory.
- `save_live_replay()`
  Saves the recorded in-memory replay to disk.
- `model_move()`
  Executes one deterministic goat-model action in live goat-play mode.
- `on_canvas_click(event)`
  Primary click handler for goat-play interaction.
- `_handle_tiger_play_click(idx)`
  Click handler for tiger-play interaction.
- `_handle_pvp_click(idx)`
  Click handler for PvP interaction.
- `_apply_action(action_flat: int)`
  Applies one goat-side action and updates the live UI.
- `_apply_tiger_action(action_flat: int)`
  Applies one tiger-side action and updates the live UI.
- `_after_tiger_step(...)`
  Handles post-step winner text, overlays, and end-of-episode flow.

#### Highlighting and validation helpers

- `_is_flat_goat_action_valid(action_flat: int) -> bool`
  Validates goat actions directly against the environment mask.
- `_find_live_tiger_action(from_idx: int, to_idx: int)`
  Maps a clicked tiger move back to an environment action.
- `_highlight_tiger_moves(tiger_idx)`
  Highlights legal tiger destinations.
- `_highlight_moves(goat_idx)`
  Highlights legal goat destinations.

#### Replay transport controls

- `jump_replay(delta)`
  Steps replay backward or forward.
- `play()`
  Starts auto-advance for replay or live auto-play.
- `pause()`
  Stops auto-advance.
- `_play_tick()`
  Main playback timer callback.
- `_on_speed_change(val)`
  Updates playback speed.
- `_on_timeline_change(val)`
  Scrubs replay position.

#### Miscellaneous controls

- `toggle_tiger_ai()`
  Switches between greedy and smart tiger behavior in goat-play mode.
- `_set_goat_play_tiger_model_controls_visible(visible: bool)`
  Shows or hides tiger-model controls for the current mode.
- `_update_timeline_ui()`
  Refreshes replay slider state and text.
- `run()`
  Starts the Tk main loop.

# `model_summary_abc.py`

Utility script for printing `torchsummary` reports for a fresh or saved `MaskablePPO` model.

### CLI helpers

- `parse_net_arch(raw_value: str) -> list[int]`
  Parses comma-separated hidden-layer sizes.
- `parse_input_size(raw_value: str) -> tuple[int, ...]`
  Parses manual summary input sizes.
- `resolve_device(device_arg: str) -> str`
  Expands `auto` to `cuda` or `cpu`.
- `build_parser()`
  Defines CLI arguments such as `--model-path`, `--module`, `--input-size`, `--device`, `--net-arch`, `--learner-role`, `--tiger-ai`, and `--goat-opponent-ai`.

### Model helpers

- `build_fresh_model(args, device: str) -> MaskablePPO`
  Constructs a new policy using `TnGEnv`.
- `load_or_build_model(args, device: str) -> MaskablePPO`
  Loads a saved `.zip` model or falls back to a fresh model.
- `resolve_module(model, module_path: str) -> nn.Module`
  Resolves dotted module paths such as `policy.mlp_extractor.policy_net`.
- `summary_device(device: str) -> str`
  Maps the runtime device string to the format expected by `torchsummary`.
- `infer_input_size(model, module_path, module) -> tuple[int, ...]`
  Infers common summary input sizes automatically.
- `print_component_summaries(model, device: str)`
  Prints per-component summaries if a full policy summary is not practical.

### Main flow

- `main()`
  Parses CLI args, loads or builds the model, resolves the target module, infers input size, and runs `torchsummary.summary(...)`.

