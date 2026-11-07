# tngviz

A desktop Tigers & Goats board that shows **perfect play** for both sides: the exact result of every
move, the best moves, and whether either side has a **forced win** and in how many plies. It reads the
solved tables from [`perfect_play/`](../perfect_play/README.md), so the values are exact, not estimates.
The layout follows the `chessviz` visualizer.

- **Verdict** above the move lists: "Goats can force a win in 37 plies", "Tigers can force a win in 5
  plies" or "Draw with perfect play". It also shows how many moves keep that result, the only winning move
  when there is one, and whether TnGEnv's turn limit cuts the win off.
- **Eval bar** next to the board: goats (cream) at the bottom, tigers (orange) at the top. It fills
  completely for a forced win and shows the plies to go. A forced win beyond the 200-ply limit fills three
  quarters of the bar. A draw sits in the middle.
- **Two move panels.** The side to move gets every legal move, best first, each with its exact result:
  `W37` wins in 37 plies, `L12` loses in 12, `=` draws. After each move comes the perfect continuation.
  The other side's panel lists the moves it would make if it were its turn (its threats).
- **Ordering:** the fastest win comes first and the slowest loss leads the losing moves. Drawing moves are
  ordered by the share of the opponent's replies that would lose ("3/12 replies lose"), so the moves that
  set traps come first.
- **On the board:**
  - Green arrows show the best move. Faint green arrows show equally fast wins.
  - A red arrow shows the other side's threat, and a blue arrow the line under the mouse.
  - Placements appear as rings, and captures bend through the jumped goat (hub jumps are not straight).
- **Values on the board:**
  - In the placement phase, every empty square carries the result of placing a goat there.
  - Select a piece to see the result of each of its moves.
  - With no piece selected, each movable piece gets a ring in the colour of its best move.
  - Outlined badges and pills are forced results beyond the turn limit.
- **Players:** the *Goats* and *Tigers* buttons open a drop-down: **Human**, **Perfect**, **Random**, or
  any trained model in `stable_models/models/Goats` or `stable_models/models/Tigers`. Perfect takes the
  fastest win or the slowest loss. In a draw it prefers positions not seen yet, then the moves that set
  the most traps. A model plays its deterministic MaskablePPO move from TnGEnv's observation and action
  mask, as in training; it loads in the background the first time it is picked (needs
  `requirements-models.txt`).
- **Move list:** click a move to jump to it, or step with Back and Next. `?` marks a move that threw away
  a win or a draw, and `??` a move that turned a win into a loss.
- **Rules:**
  - *Limits: None* (the default) is the solved game, which a draw makes endless.
  - *Limits: TnGEnv* plays like `env_tng.py`: 100 turns (200 plies), and with goats moving first, the 4th
    repeat of a board ends the game as a draw.
  - *First* chooses who moves first.
- **Copy and paste** a position (`tigers=4,12,13 goats=3,5,6,22 captured=0 mover=goat`) or a whole game.

## Setup

Requires Python 3.10 or newer and the solved tables in `perfect_play/tables` (21 GB). Build the tables
with the rebuild commands in [`perfect_play/README.md`](../perfect_play/README.md#rebuild), or pass
`--tables PATH`. Without them, the board still opens for human and random play.

The setup scripts create the virtual environment and install everything, including CPU torch and
sb3-contrib for the trained models (about 200 MB). Pass `--no-models` to skip those; the model entries
then report that the packages are missing when picked.

| | Linux / macOS | Windows (cmd or PowerShell) |
| --- | --- | --- |
| Set up | `tngviz/setup.sh` | `tngviz\setup.cmd` |
| Set up without models | `tngviz/setup.sh --no-models` | `tngviz\setup.cmd --no-models` |
| Choose the Python | `PYTHON=python3.12 tngviz/setup.sh` | `set PYTHON=py -3.12` then `tngviz\setup.cmd` |
| Environment | `tngviz/.venv` | `tngviz\.venv-windows` |
| Run | `tngviz/run.sh [options]` | `tngviz\run.cmd [options]` |
| Tests | `PYTHONPATH=. tngviz/.venv/bin/python -m tngviz.test_tngviz` | `set PYTHONPATH=.` then `tngviz\.venv-windows\Scripts\python.exe -m tngviz.test_tngviz` |

Both environments can live side by side in one checkout (for example on a drive shared with WSL).

<details><summary>Manual setup</summary>

```bash
python3 -m venv tngviz/.venv                                  # Windows: py -3 -m venv tngviz\.venv-windows
tngviz/.venv/bin/pip install -r tngviz/requirements.txt       # Windows: tngviz\.venv-windows\Scripts\python.exe -m pip ...
tngviz/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: models
tngviz/.venv/bin/pip install -r tngviz/requirements-models.txt                        # optional: models
```

</details>

## Run

```bash
tngviz/run.sh                                      # you play the goats, the perfect tigers reply
tngviz/run.sh --goats perfect --tigers human       # you play the tigers
tngviz/run.sh --goats human --tigers human         # analysis board: move both sides yourself
tngviz/run.sh --goats perfect --tigers random --delay 0.25
tngviz/run.sh --first tiger                        # the tigers move first
tngviz/run.sh --goats RobustGoat --tigers perfect  # a trained goat model (unique part of its file name)
tngviz/run.sh --goats SmartGoat --tigers RobustTiger
tngviz/run.sh --limits tngenv                      # TnGEnv's turn limit and repeat rule
tngviz/run.sh --position "tigers=4,12,13 goats=3,5,6,22 captured=0 mover=goat"   # the 215-ply win
tngviz/run.sh --help                               # all options
```

On Windows, run `tngviz\run.cmd` with the same options. You can also start it from the repository root
with `PYTHONPATH=. tngviz/.venv/bin/python -m tngviz`.

## Controls

| Action | Mouse / button | Key |
| --- | --- | --- |
| Place a goat | click an empty square | |
| Move a piece | click-click or drag | Esc deselects |
| Play a listed move | click it in the side-to-move panel | |
| Goats / Tigers: Human, Perfect, Random or a model | top-row drop-downs (cycle with keys) | `1` / `G`, `2` / `T` |
| Who moves first | First (starts a new game if no move was made) | |
| TnGEnv limits on / off | Limits | `R` |
| Step back / forward | Back / Next, mouse wheel on the board | `←` `→` |
| Start / end of the game | | `Home` / `End` |
| Jump to a move | click it in the move list | |
| Pause / resume engines | Pause | `Space` |
| New game | New game | `N` |
| Moves listed (1–25) | Lines − + | `+` / `-` |
| Continuation length (0–30 plies) | Line length − + | |
| Engine delay per move | Delay − + | |
| Analysis on / off | Analysis | `E` |
| Arrows on / off | Arrows | `A` |
| Values on the board on / off | Values | `V` |
| Square labels: names, numbers, off | Labels | `L` |
| Copy position / copy game | Copy position, Copy game | `Ctrl+C` / `Ctrl+Shift+C` |
| Paste a position or game | Paste | `Ctrl+V` |
| Quit | | `Ctrl+Q` |

Stepping back while an engine plays pauses the engines, so the game is not overwritten. Make a move, or
press Space, to continue. While paused you can move for either side.

## Notes

- **Notation:** squares use `env_tng.py`'s names, from `b0` at the apex down to `e4` (Labels shows the
  numbers 0–22 instead). `@c2` places a goat, `c2-c3` slides a piece, `b1xd1` is a capture.
- **Ply counts:** a ply is one move by one side. Plies played are counted from the standard start. A
  position pasted in the placement phase gets its exact count, since every goat turn places a piece. A
  pasted movement-phase position gets the earliest possible count.
- **Turn limit:** with *Limits: TnGEnv*, a forced win in d plies still happens only if plies played +
  d ≤ 200 (see `PerfectPlay.evaluate_with_limit`). The repeat rule never interrupts a perfect winner,
  because along a winning line the plies to go fall every ply.
- **End of the game:** six captures win for the tigers. A side with no legal move loses. With the limits
  on, the 200th ply or the 4th repeat ends the game as a draw. Switch to *Limits: None* (the default) to play on.
- **Tests:** `PYTHONPATH=. tngviz/.venv/bin/python -m tngviz.test_tngviz` from the repository root.
  They cover:
  - moves and captures against TnGEnv's action mask;
  - the end rules;
  - position text;
  - analysis against the oracle;
  - the app driven headless.
