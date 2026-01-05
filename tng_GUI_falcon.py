# ============================================================
#  Project    : Tigers & Goats – Falcon Branch
#  Module     : Tkinter GUI (Live Play + Model + Replay Viewer)
#  File       : tng_GUI_falcon.py
#  Version    : gui1.0  (set this however you version your tools)
#  Last Update: 2025-12-29
#
#  Overview
#  --------
#  This script provides a lightweight Tkinter GUI for interacting with the
#  unified Tigers & Goats environment in three modes:
#
#    (1) LIVE HUMAN PLAY (default)
#        - You click nodes to place/move goats.
#        - The tiger plays automatically using the env’s built-in tiger AI.
#
#    (2) LIVE AUTO-PLAY (optional)
#        - A trained MaskablePPO goat model can make goat moves on demand
#          ("Model Move") or continuously ("Play") with action masking.
#
#    (3) REPLAY VIEWER (optional)
#        - Browse an in-memory replay timeline using:
#            Prev / Next buttons, a timeline slider, and Play/Pause.
#        - Each step shows phase, goats eaten/placed, blocked tigers, action,
#          plus cumulative reward and outcome (when available).
#
#  Purpose
#  -------
#  • Debug tactics visually (goat placements, traps, tiger captures).
#  • Demo trained goat policies against greedy/smart tigers.
#  • Inspect recorded episodes frame-by-frame to understand "why" a model
#    won/lost (including sacrifice patterns and capture events).
#  • Provide a quick “Record Episode” pipeline to generate a timeline
#    immediately after running a fresh game.
#
#  Key Features
#  ------------
#  • Clickable graph board (nodes + edges) based on NODE_LAYOUT coordinates.
#  • Unified environment switch:
#       --env normal  -> greedy tiger (TIGER_AI_GREEDY)
#       --env battle  -> smart tiger  (TIGER_AI_SMART)
#  • Optional MaskablePPO model loading:
#       - Uses env.get_action_mask() so the model only selects legal actions.
#  • ReplayTimeline helper:
#       - Loads JSON with "timeline" entries containing {before, action, reward, info}
#       - Builds prefix reward sums for fast "running reward" display
#       - Attempts to infer winner if missing
#  • “Record Episode”:
#       - Runs a new episode (model-driven or random-goat) and stores an
#         in-memory replay with per-step snapshots for immediate browsing.
#  • Playback controls:
#       - Prev/Next stepping with goat-only preview animation on forward
#       - Timeline slider scrubbing
#       - Play/Pause with adjustable speed (ms delay)
#
#  Quick Start
#  ----------
#  1) Human vs Greedy Tiger (normal):
#       python gui_tigers_goats.py --env normal
#
#  2) Human vs Smart Tiger (battle):
#       python gui_tigers_goats.py --env battle
#
#  3) Load a trained goat model and record + browse an episode:
#       python gui_tigers_goats.py --env battle --model artifacts/models/train/mppo/GvST/.../your_model.zip
#       (Then click "Load + Record" or "Record Episode")
#
#  Workflow (recommended usage)
#  ----------------------------
#  1) Visual sanity check (live mode):
#     - Run with --env normal or --env battle
#     - Play a few turns manually to confirm:
#         • node clicks map to expected placements/moves
#         • tiger AI responds as expected
#         • phase switching (Place -> Move) looks correct
#
#  2) Model smoke test:
#     - Provide --model path/to/model.zip
#     - Use "Model Move" to step the model once at a time.
#     - If it makes illegal moves, your masking pipeline is broken:
#         • check env.get_action_mask()
#         • check Flatten action packing/unpacking vs wrapper assumptions
#
#  3) Generate a replay quickly:
#     - Click "Record Episode"
#     - The GUI runs a fresh episode using:
#         • the loaded model (deterministic=True), OR
#         • random legal actions if no model is loaded
#     - Immediately browse the recorded timeline (Prev/Next, slider, Play).
#
#  4) Replay deep dive:
#     - Use the step text to track phase, action, blocked tigers, captures.
#     - Use "Running reward" and move descriptions to correlate reward spikes
#       with tactical events (capture avoidance, traps, sacrifices).
#
#  5) Compare tiger modes:
#     - Use the "Tiger toggle" button to switch greedy <-> smart.
#     - Re-run Record Episode for side-by-side qualitative comparisons.
#
#  Notes / Practical Tips
#  ----------------------
#  • The GUI uses FlattenTnGActionWrapper and pack/unpack helpers so that:
#       action_flat = pos * N_DIR_CODES + dir
#    matches the Discrete action space used by training.
#  • Forward replay shows a goat-only preview first, then updates to the
#    post-tiger board after a short delay (speed slider controls delay).
#  • Live labeling tracks piece IDs (G1.., T1..) to make trajectories easier
#    to follow across moves/captures.
#
# ============================================================


import argparse
import json
import os
import tkinter as tk
from tkinter import ttk, filedialog

import numpy as np

try:
    from sb3_contrib import MaskablePPO
except Exception:  # allow GUI without SB3 installed
    MaskablePPO = None

from env_goat_falcon import (
    TnGEnv,
    FlattenTnGActionWrapper,
    DIR_CODES as N_DIR_CODES,
    TIGER_AI_GREEDY,
    TIGER_AI_SMART,
    TOTAL_GOATS_TO_PLACE,
    idx_to_coord,
)

EATEN_DISPLAY_CAP = 6  # show goats eaten progress out of N
TIGER_COUNT = 3
# ============================================================
# GUI Layout: node pixel positions (matches your board indexing)
# ============================================================

NODE_LAYOUT = {
    0: (280, 40),
    1: (80, 120), 2: (160, 120), 3: (240, 120), 4: (320, 120), 5: (400, 120), 6: (480, 120),
    7: (80, 200), 8: (160, 200), 9: (240, 200), 10: (320, 200), 11: (400, 200), 12: (480, 200),
    13: (80, 280), 14: (160, 280), 15: (240, 280), 16: (320, 280), 17: (400, 280), 18: (480, 280),
    19: (160, 360), 20: (240, 360), 21: (320, 360), 22: (400, 360),
}

DRAW_OFFSET_Y = 40  # shift board down to clear top overlays

def build_edges(move_map):
    edges = set()
    for src, dests in move_map.items():
        for dest in dests.values():
            if dest is None:
                continue
            edge = tuple(sorted((src, dest)))
            edges.add(edge)
    return sorted(edges)


def pack_action_flat(pos: int, dir_code: int) -> int:
    """Flatten (pos, dir) into the Discrete action index used by FlattenTnGActionWrapper."""
    return int(pos) * int(N_DIR_CODES) + int(dir_code)


def unpack_action_flat(a: int) -> tuple[int, int]:
    """Inverse of pack_action_flat."""
    pos = int(a) // int(N_DIR_CODES)
    d = int(a) % int(N_DIR_CODES)
    return pos, d


class ReplayTimeline:
    """
    Thin helper to navigate a recorded episode.
    Accepts either a path to JSON or an in-memory dict with keys:
      - timeline: list of {before, action, ...}
      - final: optional terminal board state
    """

    def __init__(self, source):
        if isinstance(source, str):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("ReplayTimeline expects a file path or dict.")

        self.timeline = data.get("timeline", [])
        self.final = data.get("final", None)
        # Derive winner/result if missing
        inferred_result = data.get("result")
        if not inferred_result:
            inferred_result = "Undetermined"
            for step in reversed(self.timeline):
                w = step.get("info", {}).get("winner")
                if w:
                    inferred_result = w
                    break
            if not self.timeline and self.final:
                inferred_result = self.final.get("winner", inferred_result)

        self.result = inferred_result
        if not self.timeline:
            raise ValueError("Replay JSON missing 'timeline' entries.")
        self.idx = 0
        self.length = len(self.timeline)
        self.prefix_rewards = [0.0]
        for step in self.timeline:
            self.prefix_rewards.append(self.prefix_rewards[-1] + float(step.get("reward", 0.0)))

    def current_board(self):
        if self.idx < self.length:
            return self.timeline[self.idx]["before"]["board"]
        if self.final:
            return self.final.get("board")
        return None  # pragma: no cover

    def cumulative_reward(self, idx: int) -> float:
        """Return cumulative reward up to and including step idx."""
        if idx < 0:
            return 0.0
        if idx >= self.length:
            return self.prefix_rewards[-1]
        return self.prefix_rewards[idx + 1]

    def step(self, delta):
        self.idx = min(max(self.idx + delta, 0), self.length)

    def info_text(self):
        if self.idx >= self.length:
            return (
                "Final state\n"
                f"Winner (spoiler): {self.result}\n"
                f"Total reward (spoiler): {self.prefix_rewards[-1]:.3f}"
            )

        step = self.timeline[self.idx]
        before = step.get("before", {})
        action = step.get("action", {})
        winner = step.get("info", {}).get("winner") or self.result
        base = (
            f"Step {self.idx+1}/{self.length} | "
            f"Phase: {before.get('phase','?')} | "
            f"Goats Eaten: {min(before.get('goats_eaten',0), EATEN_DISPLAY_CAP)}/{EATEN_DISPLAY_CAP} | "
            f"Goats Placed: {before.get('goats_placed','?')}/{TOTAL_GOATS_TO_PLACE} | "
            f"Tigers Blocked: {before.get('tigers_blocked','?')}/{TIGER_COUNT} | "
            f"Action: {action.get('pos','?')},{action.get('dir','?')}"
        )
        return (
            base
            + f"\nWinner (look ahead): {winner}"
            + f"\nTotal reward: {self.prefix_rewards[-1]:.3f}"
        )


class TigersGoatsGUI:
    def __init__(self, tiger_ai: str, model_path=None):
        self.tiger_ai = tiger_ai
        self.model_path = model_path
        self.input_locked = False

        self.root = tk.Tk()
        self.root.title("Tigers & Goats GUI")

        self.current_view = None
        self._game_active = False
        self.home_frame = None
        self.game_frame = None
        self._views = {
            "home": self._build_home_view,
            "goat_play": self._build_game_view,
        }

        self.switch_view("home")

    def switch_view(self, view_name: str):
        if view_name == self.current_view:
            return

        if self.current_view == "goat_play":
            self._deactivate_game_view()

        for child in self.root.winfo_children():
            child.destroy()

        self.home_frame = None
        self.game_frame = None
        self.current_view = view_name

        build_view = self._views.get(view_name)
        if build_view is None:
            raise ValueError(f"Unknown view: {view_name}")
        build_view()

    def _build_home_view(self):
        self.home_frame = ttk.Frame(self.root)
        self.home_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        title = ttk.Label(self.home_frame, text="Tigers & Goats", font=("Arial", 18, "bold"))
        title.pack(pady=(10, 10))
        subtitle = ttk.Label(self.home_frame, text="Choose a mode to begin")
        subtitle.pack(pady=(0, 20))

        buttons_frame = ttk.Frame(self.home_frame)
        buttons_frame.pack(pady=(0, 10))

        self.btn_home_goat = ttk.Button(
            buttons_frame,
            text="Play as Goat",
            command=self._start_goat_mode,
        )
        self.btn_home_goat.pack(fill="x", pady=6, ipadx=10)

        self.btn_home_tiger = ttk.Button(
            buttons_frame,
            text="Play as Tiger (coming soon)",
            state="disabled",
        )
        self.btn_home_tiger.pack(fill="x", pady=6, ipadx=10)

        self.btn_home_cvc = ttk.Button(
            buttons_frame,
            text="Computer vs Computer (coming soon)",
            state="disabled",
        )
        self.btn_home_cvc.pack(fill="x", pady=6, ipadx=10)

    def _start_goat_mode(self):
        self.switch_view("goat_play")

    def _quit_to_home(self):
        self.switch_view("home")

    def _build_game_view(self):
        self._game_active = True
        self.input_locked = False

        self.game_frame = ttk.Frame(self.root)
        self.game_frame.grid(row=0, column=0, sticky="nsew")
        self.game_frame.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        nav_frame = ttk.Frame(self.game_frame)
        nav_frame.grid(row=0, column=0, sticky="we", padx=10, pady=(10, 0))
        nav_frame.columnconfigure(0, weight=1)
        self.btn_quit_home = ttk.Button(nav_frame, text="Quit to Home", command=self._quit_to_home)
        self.btn_quit_home.grid(row=0, column=0, sticky="e")

        self.status_var = tk.StringVar()
        self.reward_var = tk.StringVar(value="Running reward: 0.000")
        self.move_var = tk.StringVar(value="Last move: -")
        self.model_name_var = tk.StringVar(value="Loaded model: (none)")

        # Model browse section above game stats
        model_top_frame = ttk.Frame(self.game_frame)
        model_top_frame.grid(row=1, column=0, sticky="we", padx=10, pady=5)
        for c in range(3):
            model_top_frame.columnconfigure(c, weight=1)
        ttk.Label(model_top_frame, text="Model path:").grid(row=0, column=0, columnspan=3, sticky="w")
        self.model_entry = ttk.Entry(model_top_frame, width=50)
        self.model_entry.grid(row=1, column=0, columnspan=3, sticky="we", pady=(2, 2))
        self.btn_browse_model = ttk.Button(model_top_frame, text="Browse", command=self.browse_model)
        self.btn_browse_model.grid(row=2, column=0, sticky="we")
        self.btn_record = ttk.Button(model_top_frame, text="Record Episode", command=self.record_episode)
        self.btn_record.grid(row=2, column=1, sticky="we")
        self.btn_load_model = ttk.Button(model_top_frame, text="Load + Record", command=self.load_model_from_entry)
        self.btn_load_model.grid(row=2, column=2, sticky="we")
        # Difficulty selector directly under Load + Record
        self.btn_toggle_tiger = ttk.Button(model_top_frame, text="", command=self.toggle_tiger_ai)
        self.btn_toggle_tiger.grid(row=3, column=0, sticky="w", pady=(5, 0))

        # Game stats just below model browse
        metrics_frame = ttk.Frame(self.game_frame)
        metrics_frame.grid(row=2, column=0, columnspan=2, sticky="we", padx=10, pady=5)
        ttk.Label(metrics_frame, textvariable=self.status_var, width=100, anchor="w").pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.reward_var, width=100, anchor="w").pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.move_var, width=100, anchor="w").pack(anchor="w")

        self.canvas_width = 600
        self.canvas_height = 520
        self.canvas = tk.Canvas(
            self.game_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="#222222",
            highlightthickness=0,
        )
        self.canvas.grid(row=3, column=0, padx=10, pady=(10, 0))
        # Canvas overlays: turn indicator, phase text, and on-canvas stats
        self.overlay_items = {}
        self._init_canvas_overlays()
        self.model_name_canvas = self.canvas.create_text(
            self.canvas_width / 2,
            12,
            text=self.model_name_var.get(),
            fill="white",
            anchor="n",
        )
        self.model_name_var.trace_add("write", lambda *_: self._update_model_name_on_canvas())

        # Controls stacked under the canvas
        control_frame = ttk.Frame(self.game_frame)
        control_frame.grid(row=4, column=0, sticky="we", padx=10, pady=(5, 2))
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        self.btn_reset = ttk.Button(control_frame, text="Reset", command=self.reset_env)
        self.btn_reset.grid(row=0, column=1, sticky="we", padx=5)
        self.btn_prev = ttk.Button(control_frame, text="Prev", command=lambda: self.jump_replay(-1), state="disabled")
        self.btn_prev.grid(row=0, column=0, sticky="w")
        self.btn_play = ttk.Button(control_frame, text="Play", command=self.play)
        self.btn_play.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.btn_model = ttk.Button(control_frame, text="Model Move", command=self.model_move, state="disabled")
        self.btn_model.grid(row=1, column=1, sticky="we", padx=5, pady=(2, 0))
        self.btn_next = ttk.Button(control_frame, text="Next", command=lambda: self.jump_replay(1), state="disabled")
        self.btn_next.grid(row=0, column=2, sticky="e")
        self.btn_pause = ttk.Button(control_frame, text="Pause", command=self.pause)
        self.btn_pause.grid(row=1, column=2, sticky="e", pady=(2, 0))

        self.speed_label_var = tk.StringVar(value="Speed (ms): 200")
        ttk.Label(self.game_frame, textvariable=self.speed_label_var).grid(row=5, column=0, sticky="w", padx=10, pady=2)
        self.speed_var = tk.IntVar(value=200)
        self.speed_scale = ttk.Scale(
            self.game_frame,
            from_=50,
            to=1000,
            orient="horizontal",
            variable=self.speed_var,
            command=self._on_speed_change,
        )
        self.speed_scale.grid(row=6, column=0, sticky="we", padx=10, pady=(0, 2))

        # Timeline slider (replay)
        self.timeline_label_var = tk.StringVar(value="Timeline: 0/0")
        ttk.Label(self.game_frame, textvariable=self.timeline_label_var).grid(row=7, column=0, sticky="w", padx=10, pady=2)
        self.timeline_var = tk.IntVar(value=0)
        self.timeline_scale = ttk.Scale(
            self.game_frame,
            from_=0,
            to=0,
            orient="horizontal",
            variable=self.timeline_var,
            command=self._on_timeline_change,
            state="disabled",
        )
        self.timeline_scale.grid(row=8, column=0, sticky="we", padx=10, pady=(0, 2))

        self._init_runtime_state()

        # Initialize environment/replay and initial draw
        self._init_env_or_replay()
        self._update_tiger_button_text()
        self._draw_static()
        self._draw_board()
        self._update_timeline_ui()

        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def _init_runtime_state(self):
        self.node_items = {}
        self.edge_items = []
        self.selected_goat = None
        self.replay = None
        self.replay_animating = False
        self._move_map_cache = None
        self.cumulative_reward = 0.0
        self.playing = False
        self.piece_labels_live = {}
        self.next_goat_id = 1
        self.last_winner = None
        self._pending_before = None
        self._pending_action = None
        self._pending_phase = None
        self.base_env = None
        self.env = None
        self.obs = None
        self.model = None

    def _deactivate_game_view(self):
        if not self._game_active:
            return
        self._game_active = False
        self.playing = False
        self.replay_animating = False
        self.input_locked = True
        self.replay = None
        self.base_env = None
        self.env = None
        self.obs = None
        self.model = None

    def _update_model_name_on_canvas(self):
        if not self._game_active or not hasattr(self, "model_name_canvas"):
            return
        if self.canvas:
            self.canvas.itemconfig(self.model_name_canvas, text=self.model_name_var.get())

    def _init_canvas_overlays(self):
        # Game state (top-left)
        self.overlay_items["game_state"] = self.canvas.create_text(
            12, 12, text="IN PLAY", fill="#2ecc71", anchor="nw", font=("Arial", 10, "bold")
        )

        # Stats bottom row
        x_left, x_mid, x_right = 110, self.canvas_width / 2, self.canvas_width - 110
        label_y = self.canvas_height - 50
        value_y = self.canvas_height - 30
        self.overlay_items["lbl_blocked"] = self.canvas.create_text(
            x_left, label_y, text="TIGERS BLOCKED", fill="white", anchor="center"
        )
        self.overlay_items["val_blocked"] = self.canvas.create_text(
            x_left, value_y, text="-/-", fill="#f4d03f", anchor="center"
        )
        self.overlay_items["lbl_eaten"] = self.canvas.create_text(
            x_mid, label_y, text="GOATS EATEN", fill="white", anchor="center"
        )
        self.overlay_items["val_eaten"] = self.canvas.create_text(
            x_mid, value_y, text="-/-", fill="#f4d03f", anchor="center"
        )
        self.overlay_items["lbl_placed"] = self.canvas.create_text(
            x_right, label_y, text="GOATS PLACED", fill="white", anchor="center"
        )
        self.overlay_items["val_placed"] = self.canvas.create_text(
            x_right, value_y, text="-/-", fill="#f4d03f", anchor="center"
        )

        # Turn indicator circle (top-right)
        cx, cy, r = self.canvas_width - 60, 45, 24
        self.overlay_items["turn_circle"] = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r, fill="#4da6ff", outline="white", width=2
        )
        self.overlay_items["turn_text"] = self.canvas.create_text(
            cx, cy, text="0", fill="black", font=("Arial", 12, "bold")
        )

        # Phase text just below
        self.overlay_items["phase_text"] = self.canvas.create_text(
            cx, cy + 32, text="Phase: -", fill="white", anchor="center", font=("Arial", 10, "bold")
        )

    def _update_canvas_overlays(self, placed, eaten, blocked, turn, phase, actor="Goat", live=True):
        placed_txt = f"{placed}/{TOTAL_GOATS_TO_PLACE}"
        eaten_txt = f"{min(eaten, EATEN_DISPLAY_CAP)}/{EATEN_DISPLAY_CAP}"
        blocked_txt = f"{blocked}/{TIGER_COUNT}"
        if self.overlay_items:
            if "game_state" in self.overlay_items:
                if self.replay:
                    state_text = "REPLAY"
                    state_color = "#f4d03f"
                elif self.last_winner:
                    state_text = f"GAME OVER: {self.last_winner}"
                    state_color = "#ff4d4d"
                else:
                    state_text = "IN PLAY"
                    state_color = "#2ecc71"
                self.canvas.itemconfig(
                    self.overlay_items["game_state"], text=state_text, fill=state_color
                )

            self.canvas.itemconfig(self.overlay_items["val_blocked"], text=blocked_txt)
            self.canvas.itemconfig(self.overlay_items["val_eaten"], text=eaten_txt)
            self.canvas.itemconfig(self.overlay_items["val_placed"], text=placed_txt)

            # Turn indicator
            turn_color = "#4da6ff" if actor.lower().startswith("g") else "#ff7f50"
            self.canvas.itemconfig(self.overlay_items["turn_circle"], fill=turn_color)
            self.canvas.itemconfig(self.overlay_items["turn_text"], text=str(turn))

            # Phase indicator
            phase_label = "Place" if int(phase) == 0 else "Move"
            self.canvas.itemconfig(self.overlay_items["phase_text"], text=f"Phase: {phase_label}")

    def _init_env_or_replay(self):
        if self.replay:
            self.base_env = None
            self.env = None
            self.btn_prev["state"] = "normal"
            self.btn_next["state"] = "normal"
            self.btn_model["state"] = "disabled"
            self.status_var.set("Replay mode")
            self._update_timeline_ui()
            return

        self._init_live_env()

    def _init_live_env(self):
        # Unified env: choose tiger behavior via tiger_ai
        self.base_env = TnGEnv(tiger_ai=self.tiger_ai)
        self.env = FlattenTnGActionWrapper(self.base_env)
        self.last_winner = None

        self.obs, _ = self.env.reset()
        self._init_live_labels()
        self.btn_prev["state"] = "disabled"
        self.btn_next["state"] = "disabled"
        self.btn_model["state"] = "normal" if self.model_path else "disabled"
        self._update_tiger_button_text()

        mode_label = "battle/smart" if self.tiger_ai == TIGER_AI_SMART else "normal/greedy"
        self.status_var.set(f"Live mode ({mode_label} tiger): click a node to place/move goats.")
        self.move_var.set("Last move: -")

        # Optional PPO model (trained on flattened wrapper)
        self._load_model(self.model_path)
        self._update_tiger_button_text()

    def _draw_static(self):
        # Graph is independent of tiger mode; pull from a temp env if replay
        if self.replay:
            tmp = TnGEnv(tiger_ai=TIGER_AI_GREEDY)
            move_map = tmp.move_map
        else:
            move_map = self.base_env.move_map

        edges = build_edges(move_map)
        for a, b in edges:
            x1, y1 = self._node_xy(a)
            x2, y2 = self._node_xy(b)
            line = self.canvas.create_line(x1, y1, x2, y2, fill="#555555", width=2)
            self.edge_items.append(line)

        radius = 22
        for idx in NODE_LAYOUT:
            x, y = self._node_xy(idx)
            circle = self.canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill="#444444", outline="#888888", width=2
            )
            label = self.canvas.create_text(x, y, text=idx_to_coord(idx), fill="white")
            self.node_items[idx] = (circle, label)
        


    def _draw_board(self, board_override=None):
        if not self._game_active or self.canvas is None:
            return
        board = board_override if board_override is not None else self._current_board()
        if board is None:
            return

        labels = self._current_labels()
        for idx, val in enumerate(board):
            circle, label = self.node_items[idx]
            if val == 0:
                color = "#444444"
            elif val == 1:
                color = "#4da6ff"   # goats
            elif val == 2:
                color = "#ff7f50"   # tigers
            else:
                color = "#aaaaaa"
            # reset outline each draw to clear old highlights
            self.canvas.itemconfig(circle, fill=color, outline="#888888", width=2)
            text = labels.get(idx, idx_to_coord(idx))
            self.canvas.itemconfig(label, text=text)

        if self.replay:
            self.status_var.set(self.replay.info_text())
            # last move description for replay
            if self.replay.idx == 0:
                self.move_var.set("Last move: start")
            else:
                prev_board = self.replay.timeline[self.replay.idx - 1]["before"]["board"]
                if self.replay.idx < self.replay.length:
                    curr_board = self.replay.timeline[self.replay.idx]["before"]["board"]
                else:
                    curr_board = self.replay.final.get("board", prev_board)
                action = self.replay.timeline[self.replay.idx - 1]["action"]
                phase = self.replay.timeline[self.replay.idx - 1]["before"].get("phase", 0)
                desc = self._describe_transition(prev_board, curr_board, action, phase)
                self.move_var.set(f"Last move: {desc}")
        else:
            phase = getattr(self.base_env, "phase", 0)
            eaten = getattr(self.base_env, "eaten", getattr(self.base_env, "goats_eaten", "?"))
            placed = getattr(self.base_env, "goats_placed", "?")
            blocked = self._blocked_tigers_live()
            turns = getattr(self.base_env, "turns", getattr(self.base_env, "turn_count", "?"))
            phase_s = "Place" if phase == 0 else "Move"
            self.status_var.set(
                f"Phase: {phase_s} | "
                f"Goats Placed: {placed}/{TOTAL_GOATS_TO_PLACE} | "
                f"Goats Eaten: {min(eaten, EATEN_DISPLAY_CAP)}/{EATEN_DISPLAY_CAP} | "
                f"Tigers Blocked: {blocked}/{TIGER_COUNT} | "
                f"Turn: {turns}"
            )

        # Running reward viewer
        if self.replay:
            running_r = self.replay.cumulative_reward(self.replay.idx)
        else:
            running_r = self.cumulative_reward
        self.reward_var.set(f"Running reward: {running_r:.3f}")

        # On-canvas overlay updates
        if self.replay:
            if self.replay.idx < self.replay.length:
                before = self.replay.timeline[self.replay.idx]["before"]
            else:
                before = self.replay.final or {}
            phase_val = before.get("phase", 0)
            eaten_val = before.get("goats_eaten", 0)
            placed_val = before.get("goats_placed", 0)
            blocked_val = before.get("tigers_blocked", 0)
            turn_val = before.get("turn", self.replay.idx)
            actor = "Tiger" if self.replay_animating else "Goat"
        else:
            phase_val = getattr(self.base_env, "phase", 0)
            eaten_val = getattr(self.base_env, "eaten", getattr(self.base_env, "goats_eaten", 0))
            placed_val = getattr(self.base_env, "goats_placed", 0)
            blocked_val = self._blocked_tigers_live()
            turn_val = getattr(self.base_env, "turns", getattr(self.base_env, "turn_count", 0))
            actor = "Tiger" if self.input_locked else "Goat"
        self._update_canvas_overlays(placed_val, eaten_val, blocked_val, turn_val, phase_val, actor=actor)

    def _preview_board_after_goat(self, action_flat: int):
        """
        Return a board list representing the immediate goat move result,
        WITHOUT tiger response/capture. This is purely for GUI animation.
        """
        # snapshot BEFORE step
        board = self.base_env.board.copy()

        pos, d = unpack_action_flat(action_flat)

        phase = getattr(self.base_env, "phase", 0)

        if phase == 0:
            # placing phase: drop goat at pos
            if board[pos] == 0:
                board[pos] = 1
            return board.tolist()

        # moving phase: move goat from pos -> dest
        dest = self.base_env.move_map.get(pos, {}).get(d, None)
        if dest is None:
            return board.tolist()

        # only move if it looks like a goat is actually there
        if board[pos] == 1 and board[dest] == 0:
            board[pos] = 0
            board[dest] = 1

        return board.tolist()


    def _current_board(self):
        if self.replay:
            return self.replay.current_board()
        return self.base_env.board.tolist()

    def _current_labels(self):
        if self.replay:
            return self._labels_for_replay(self.replay.idx)
        return self.piece_labels_live

    def _labels_for_replay(self, idx):
        """
        Reconstruct piece labels by simulating actions up to step idx.
        Labels:
          - Tigers: T1..T3 based on initial tiger positions
          - Goats:  G1.. placed order
        """
        if not self.replay:
            return {}

        # start from first board
        first_board = self.replay.timeline[0]["before"]["board"]
        labels = {}
        goat_id = 1

        # initial tigers
        tiger_positions = [i for i, v in enumerate(first_board) if v == 2]
        for t_idx, pos in enumerate(sorted(tiger_positions), 1):
            labels[pos] = f"T{t_idx}"

        # simulate steps up to idx
        for i in range(min(idx, self.replay.length)):
            curr = self.replay.timeline[i]["before"]["board"]
            next_board = (
                self.replay.timeline[i + 1]["before"]["board"]
                if (i + 1) < self.replay.length
                else (self.replay.final.get("board") if self.replay.final else curr)
            )
            action = self.replay.timeline[i]["action"]
            phase = self.replay.timeline[i]["before"].get("phase", 0)
            pos = int(action.get("pos", 0))
            d = int(action.get("dir", 0))

            # Goat move/placement
            if phase == 0:
                labels[pos] = f"G{goat_id}"
                goat_id += 1
            else:
                dest = self._get_move_map().get(pos, {}).get(d, None)
                if dest is not None and pos in labels:
                    labels[dest] = labels.pop(pos)

            # Tiger move (identify moved tiger)
            curr_t = {p for p, v in enumerate(curr) if v == 2}
            next_t = {p for p, v in enumerate(next_board) if v == 2}
            if curr_t != next_t and len(curr_t) == len(next_t):
                moved_from = list(curr_t - next_t)
                moved_to = list(next_t - curr_t)
                if len(moved_from) == 1 and len(moved_to) == 1:
                    f_pos = moved_from[0]
                    t_pos = moved_to[0]
                    if f_pos in labels:
                        labels[t_pos] = labels.pop(f_pos)

            # Capture: remove goats that vanished
            curr_goats = {p for p, v in enumerate(curr) if v == 1}
            next_goats = {p for p, v in enumerate(next_board) if v == 1}
            captured = curr_goats - next_goats
            for c in captured:
                labels.pop(c, None)

            # Also clean labels for emptied cells
            for p, v in enumerate(next_board):
                if v == 0:
                    if p in labels and (labels[p].startswith("G") or labels[p].startswith("T")):
                        labels.pop(p, None)
        return labels

    def _init_live_labels(self):
        self.piece_labels_live = {}
        self.next_goat_id = 1
        for idx, v in enumerate(self.base_env.board):
            if v == 2:
                self.piece_labels_live[idx] = f"T{len(self.piece_labels_live) + 1}"

    def _update_live_labels(self, before_board, after_board, action, phase):
        pos = int(action.get("pos", 0))
        d = int(action.get("dir", 0))
        move_map = self._get_move_map()

        # Goat placement / move
        if phase == 0:
            self.piece_labels_live[pos] = f"G{self.next_goat_id}"
            self.next_goat_id += 1
        else:
            dest = move_map.get(pos, {}).get(d, None)
            if dest is not None and pos in self.piece_labels_live:
                self.piece_labels_live[dest] = self.piece_labels_live.pop(pos)

        # Tiger move
        before_t = {i for i, v in enumerate(before_board) if v == 2}
        after_t = {i for i, v in enumerate(after_board) if v == 2}
        if before_t != after_t and len(before_t) == len(after_t):
            moved_from = list(before_t - after_t)
            moved_to = list(after_t - before_t)
            if len(moved_from) == 1 and len(moved_to) == 1:
                f_pos = moved_from[0]
                t_pos = moved_to[0]
                if f_pos in self.piece_labels_live:
                    self.piece_labels_live[t_pos] = self.piece_labels_live.pop(f_pos)

        # Captures / cleanup
        for idx, val in enumerate(after_board):
            if val == 0:
                self.piece_labels_live.pop(idx, None)
            elif val == 1:
                # ensure goats have label
                if idx not in self.piece_labels_live or not self.piece_labels_live[idx].startswith("G"):
                    self.piece_labels_live[idx] = self.piece_labels_live.get(idx, f"G{self.next_goat_id}")
            elif val == 2:
                if idx not in self.piece_labels_live or not self.piece_labels_live[idx].startswith("T"):
                    # fallback tiger label
                    self.piece_labels_live[idx] = self.piece_labels_live.get(idx, "T?")

    def reset_env(self):
        # Clear replay and re-enter live (human) mode
        self.replay = None
        self.replay_animating = False
        self.selected_goat = None
        self.cumulative_reward = 0.0
        self.playing = False
        self.last_winner = None
        self._init_live_env()
        self._draw_board()
        self._update_timeline_ui()

    def _get_move_map(self):
        if self.base_env is not None:
            return self.base_env.move_map
        if self._move_map_cache is None:
            self._move_map_cache = TnGEnv(tiger_ai=self.tiger_ai).move_map
        return self._move_map_cache

    def _preview_from_record(self, step):
        """
        Build a goat-only preview board from a recorded step (pre-tiger).
        """
        board = np.array(step["before"]["board"], dtype=int)
        pos = int(step["action"].get("pos", 0))
        d = int(step["action"].get("dir", 0))
        phase = int(step.get("before", {}).get("phase", 0))
        move_map = self._get_move_map()

        if phase == 0:
            if 0 <= pos < board.size and board[pos] == 0:
                board[pos] = 1
            return board.tolist()

        dest = move_map.get(pos, {}).get(d, None)
        if dest is not None and 0 <= dest < board.size and board[pos] == 1 and board[dest] == 0:
            board[pos] = 0
            board[dest] = 1
        return board.tolist()

    def record_episode(self, max_steps: int = 200):
        """
        Run a fresh episode (using the same tiger AI and optional goat model),
        store it as an in-memory replay, and switch the UI into replay mode.
        """
        self.cumulative_reward = 0.0
        rec_env_base = TnGEnv(tiger_ai=self.tiger_ai)
        rec_env = FlattenTnGActionWrapper(rec_env_base)
        obs, _ = rec_env.reset()

        timeline = []
        steps = 0

        last_info = {}
        terminated = False
        truncated = False

        while steps < max_steps:
            mask = rec_env_base.get_action_mask()
            if mask is None or not np.any(mask):
                action_flat = 0
            elif self.model is not None:
                action_flat, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
                action_flat = int(action_flat)
            else:
                valid = np.flatnonzero(mask)
                action_flat = int(np.random.choice(valid)) if valid.size else 0

            tiger_moves = rec_env_base._tiger_moves()
            blocked = self._blocked_from_moves(rec_env_base.board, tiger_moves)

            before = {
                "board": rec_env_base.board.tolist(),
                "phase": int(getattr(rec_env_base, "phase", 0)),
                "goats_eaten": int(getattr(rec_env_base, "eaten", 0)),
                "goats_placed": int(getattr(rec_env_base, "goats_placed", 0)),
                "tigers_blocked": int(blocked),
                "turn": int(getattr(rec_env_base, "turns", steps)),
            }
            timeline.append({
                "before": before,
                "action": {
                    "pos": int(action_flat) // int(N_DIR_CODES),
                    "dir": int(action_flat) % int(N_DIR_CODES),
                },
            })

            obs, reward, terminated, truncated, info = rec_env.step(action_flat)
            clean_info = {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in info.items()
            }
            last_info = clean_info
            timeline[-1]["reward"] = float(reward)
            timeline[-1]["info"] = clean_info
            steps += 1
            if terminated or truncated:
                break

        final = {
            "board": rec_env_base.board.tolist(),
            "phase": int(getattr(rec_env_base, "phase", 0)),
            "goats_eaten": int(getattr(rec_env_base, "eaten", 0)),
            "goats_placed": int(getattr(rec_env_base, "goats_placed", 0)),
            "tigers_blocked": int(self._blocked_from_moves(rec_env_base.board, rec_env_base._tiger_moves())),
            "turn": int(getattr(rec_env_base, "turns", steps)),
            "winner": last_info.get("winner", "Undetermined") if (terminated or truncated) else "Undetermined",
        }

        # Outcome text for quick summary in replay
        if terminated or truncated:
            result = last_info.get("winner", "Unknown") if timeline else "Unknown"
        else:
            result = "Undetermined"

        replay_data = {"timeline": timeline, "final": final, "result": result}
        self.replay = ReplayTimeline(replay_data)
        self.btn_prev["state"] = "normal"
        self.btn_next["state"] = "normal"
        self.btn_model["state"] = "disabled"
        self.status_var.set(f"Recorded episode (Result: {result}): use Prev/Next to browse.")
        self._draw_board()
        self._update_timeline_ui()

    def model_move(self):
        if not self.model or not self.env:
            return
        mask = self.base_env.get_action_mask()  # flattened mask should match wrapper
        action, _ = self.model.predict(self.obs, deterministic=True, action_masks=mask)
        self._apply_action(int(action))

    def jump_replay(self, delta):
        if not self.replay or self.replay_animating:
            return

        # Forward: show goat-only preview, then advance to tiger-updated board
        if delta > 0:
            if self.replay.idx >= self.replay.length:
                return
            self.replay_animating = True
            self.btn_prev["state"] = "disabled"
            self.btn_next["state"] = "disabled"

            step = self.replay.timeline[self.replay.idx]
            preview = self._preview_from_record(step)
            self._draw_board(board_override=preview)

            self.root.after(self._delay_ms(), self._finish_replay_forward)
            return

        # Backward: step back then pause 200 - 500ms (no goat preview needed)
        if delta < 0:
            if self.replay.idx <= 0:
                return
            self.replay_animating = True
            self.btn_prev["state"] = "disabled"
            self.btn_next["state"] = "disabled"
            self.replay.idx = max(self.replay.idx - 1, 0)
            self._draw_board()
            self.root.after(self._delay_ms(), self._finish_replay_complete)

    def _finish_replay_forward(self):
        if not self._game_active:
            return
        # Advance to the board AFTER tiger response (next step's before board)
        self.replay.idx = min(self.replay.idx + 1, self.replay.length)
        self._draw_board()
        self._finish_replay_complete()

    def _finish_replay_complete(self):
        if not self._game_active:
            return
        self.replay_animating = False
        if self.replay:
            self.btn_prev["state"] = "normal" if self.replay.idx > 0 else "disabled"
            self.btn_next["state"] = "normal" if self.replay.idx < self.replay.length else "disabled"
            self._update_timeline_ui()

    def _update_tiger_button_text(self):
        label = "Tiger: SMART (click -> NORMAL)" if self.tiger_ai == TIGER_AI_SMART else "Tiger: NORMAL (click -> SMART)"
        self.btn_toggle_tiger["text"] = label

    def _update_timeline_ui(self):
        if not self._game_active:
            return
        if self.replay:
            self.timeline_scale.configure(state="normal", to=self.replay.length)
            self.timeline_var.set(self.replay.idx)
            self.timeline_label_var.set(f"Timeline: {self.replay.idx}/{self.replay.length}")
        else:
            self.timeline_scale.configure(state="disabled", to=0)
            self.timeline_var.set(0)
            self.timeline_label_var.set("Timeline: 0/0")

    def toggle_tiger_ai(self):
        # Switch tiger AI and reset into live mode (clears replay)
        self.tiger_ai = TIGER_AI_SMART if self.tiger_ai == TIGER_AI_GREEDY else TIGER_AI_GREEDY
        self.reset_env()

    def _load_model(self, path: str | None):
        """Load a PPO model if a path is provided; clear if invalid or None."""
        self.model = None
        if not path:
            self.model_name_var.set("Loaded model: (none)")
            return
        if MaskablePPO is None:
            self.status_var.set("sb3_contrib not installed; cannot load model.")
            return
        if not os.path.isfile(path):
            self.status_var.set(f"Model not found: {path}")
            self.model_name_var.set("Loaded model: (none)")
            return
        try:
            self.model = MaskablePPO.load(path, env=self.env)
            self.model_path = path
            # Pre-fill entry so user sees what is loaded
            if hasattr(self, "model_entry"):
                self.model_entry.delete(0, tk.END)
                self.model_entry.insert(0, path)
            self.model_name_var.set(f"Loaded model: {os.path.basename(path)}")
        except Exception as e:
            self.model = None
            self.status_var.set(f"Failed to load model: {e}")
            self.model_name_var.set("Loaded model: (none)")

    def load_model_from_entry(self):
        """Load model from entry text, then record an episode for browsing."""
        path = self.model_entry.get().strip()
        # Update model path and reset to live (clears any replay)
        self.model_path = path
        self.reset_env()
        # _load_model is called inside _init_live_env during reset
        if self.model:
            self.status_var.set(f"Loaded model: {os.path.basename(path)}; recording episode...")
            self.record_episode()
        else:
            self.status_var.set(f"Failed to load model: {path}")

    def browse_model(self):
        """Open file picker for a model zip, load it, and record an episode."""
        path = filedialog.askopenfilename(
            title="Select model (.zip)",
            filetypes=[("Model zip", "*.zip"), ("All files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "artifacts"),
        )
        if not path:
            return
        self.model_entry.delete(0, tk.END)
        self.model_entry.insert(0, path)
        self.load_model_from_entry()

    def on_canvas_click(self, event):
        if self.input_locked:
            return
        if self.replay or not self.base_env:
            return

        idx = self._loc_to_node(event.x, event.y)
        if idx is None:
            return

        phase = getattr(self.base_env, "phase", 0)
        valid_moves = getattr(self.base_env, "valid_moves", [])

        # ------------------------------------------------------------
        # Placing phase: click destination node to place a goat
        # In your env this is represented as (pos=idx, dir=0) style
        # ------------------------------------------------------------
        if phase == 0:
            # find a valid place action (idx, 0)
            if [idx, 0] in valid_moves or (idx, 0) in valid_moves:
                flat = pack_action_flat(idx, 0)
                self._apply_action(flat)
            return

        # ------------------------------------------------------------
        # Moving phase: click goat to select, then click destination
        # ------------------------------------------------------------
        if self.selected_goat is None:
            if self.base_env.board[idx] == 1:
                self.selected_goat = idx
                self._highlight_moves(idx)
        else:
            from_idx = self.selected_goat
            self.selected_goat = None
            self._draw_board()

            dir_code = None
            for d, dest in self.base_env.move_map.get(from_idx, {}).items():
                if dest == idx:
                    if [from_idx, d] in valid_moves or (from_idx, d) in valid_moves:
                        dir_code = d
                        break

            if dir_code is not None:
                flat = pack_action_flat(from_idx, dir_code)
                self._apply_action(flat)

    def _apply_action(self, action_flat: int):
        if self.input_locked:
            return
        self.input_locked = True

        # Draw goat-only preview FIRST (before stepping env)
        preview = self._preview_board_after_goat(action_flat)
        self.selected_goat = None
        self._draw_board(board_override=preview)

        # store pre-step context for labeling / descriptions
        self._pending_before = self.base_env.board.copy()
        pos, d = unpack_action_flat(action_flat)
        self._pending_action = {"pos": pos, "dir": d}
        self._pending_phase = getattr(self.base_env, "phase", 0)

        # Now step the real env (this includes tiger response/capture)
        self.obs, reward, terminated, truncated, info = self.env.step(action_flat)

        # After delay, draw the real (post-tiger) state
        self.root.after(
            self._delay_ms(),
            lambda: self._after_tiger_step(reward, terminated, truncated, info),
        )


    def _after_tiger_step(self, reward, terminated, truncated, info):
        if not self._game_active:
            return
        self.cumulative_reward += float(reward)
        # update live labels and move description
        if self._pending_before is not None and self._pending_action is not None:
            after_board = self.base_env.board.copy()
            self._update_live_labels(self._pending_before, after_board, self._pending_action, self._pending_phase or 0)
            desc = getattr(self.base_env, "last_move_desc", "") or self._describe_transition(
                self._pending_before, after_board, self._pending_action, self._pending_phase or 0
            )
            self.move_var.set(f"Last move: {desc}")
        else:
            self.move_var.set("Last move: -")

        # clear pending
        self._pending_before = None
        self._pending_action = None
        self._pending_phase = None

        # unlock before drawing so turn indicator reflects goat turn
        self.input_locked = False

        if terminated or truncated:
            winner = info.get("winner", "Done")
            self.last_winner = winner
            self.status_var.set(f"Game over: {winner} | reward {reward:.3f}")

        # show the real env board now
        self._draw_board()
        self._update_timeline_ui()



    def _highlight_moves(self, goat_idx):
        # redraw to clear any previous highlights
        self._draw_board()
        circle, _ = self.node_items[goat_idx]
        self.canvas.itemconfig(circle, outline="#ffff00", width=3)

        valid_moves = getattr(self.base_env, "valid_moves", [])
        for d, dest in self.base_env.move_map.get(goat_idx, {}).items():
            if [goat_idx, d] in valid_moves or (goat_idx, d) in valid_moves:
                circle2, _ = self.node_items[dest]
                self.canvas.itemconfig(circle2, outline="#ffd700", width=3)

    def _loc_to_node(self, x, y):
        for idx in NODE_LAYOUT:
            nx, ny = self._node_xy(idx)
            if (x - nx) ** 2 + (y - ny) ** 2 <= 18 ** 2:
                return idx
        return None

    def _node_xy(self, idx):
        x, y = NODE_LAYOUT[idx]
        return x, y + DRAW_OFFSET_Y

    def _on_speed_change(self, val):
        """Update speed label when slider moves."""
        try:
            ms = int(float(val))
        except Exception:
            ms = 200
        self.speed_label_var.set(f"Speed (ms): {ms}")
        # If playing live/replay, no need to interrupt; next tick will use new delay

    def _on_timeline_change(self, val):
        """Seek within replay using the timeline slider."""
        if not self.replay or self.replay_animating:
            return
        try:
            idx = int(float(val))
        except Exception:
            return
        idx = min(max(idx, 0), self.replay.length)
        self.playing = False  # stop auto-play when scrubbing
        self.replay.idx = idx
        self._draw_board()
        self._update_timeline_ui()

    def _describe_transition(self, before_board, after_board, action, phase):
        pos = int(action.get("pos", 0))
        d = int(action.get("dir", 0))
        pos_c = idx_to_coord(pos)
        move_map = self._get_move_map()

        # Goat part
        if phase == 0:
            goat_part = f"Goat placed at {pos_c}"
        else:
            dest = move_map.get(pos, {}).get(d, None)
            dest_c = idx_to_coord(dest) if dest is not None else str(dest)
            goat_part = f"Goat moved {idx_to_coord(pos)} -> {dest_c}"

        # Tiger move detection
        tiger_part = ""
        before_t = {i for i, v in enumerate(before_board) if v == 2}
        after_t = {i for i, v in enumerate(after_board) if v == 2}
        moved_from = list(before_t - after_t)
        moved_to = list(after_t - before_t)
        if len(moved_from) == 1 and len(moved_to) == 1:
            tiger_part = f"Tiger moved {idx_to_coord(moved_from[0])} -> {idx_to_coord(moved_to[0])}"

        # Capture detection
        before_goats = {i for i, v in enumerate(before_board) if v == 1}
        after_goats = {i for i, v in enumerate(after_board) if v == 1}
        captured = before_goats - after_goats
        captured_flag = False
        if captured and (len(after_goats) < len(before_goats)):
            captured_flag = True
            cap_txt = ", ".join(idx_to_coord(c) for c in captured)
            if tiger_part:
                tiger_part += f" \nand captured goat at {cap_txt}"
            else:
                tiger_part = f"\nTiger captured goat at {cap_txt}"

        parts = [goat_part]
        if tiger_part:
            parts.append(tiger_part)
        first_line = " | ".join(parts)
        eaten_line = "Goat eaten: Yes" if captured_flag else "Goat eaten: No"
        return f"{first_line}\n{eaten_line}"

    def _blocked_from_moves(self, board, tiger_moves):
        tigers = [i for i, v in enumerate(board) if v == 2]
        movable = {m[0] for m in tiger_moves} if tiger_moves else set()
        blocked = len([t for t in tigers if t not in movable])
        return max(0, min(len(tigers), blocked))

    def _blocked_tigers_live(self):
        if not self.base_env or not hasattr(self.base_env, "_tiger_moves"):
            return "?"
        moves = self.base_env._tiger_moves()
        return self._blocked_from_moves(self.base_env.board, moves)

    # ------------------------------------------------------------
    # Playback helpers
    # ------------------------------------------------------------
    def _delay_ms(self):
        try:
            return max(10, int(self.speed_var.get()))
        except Exception:
            return 200

    def play(self):
        self.playing = True
        self._play_tick()

    def pause(self):
        self.playing = False

    def _play_tick(self):
        if not self._game_active:
            self.playing = False
            return
        if not self.playing:
            return

        # In replay mode: auto-advance
        if self.replay:
            if self.replay.idx >= self.replay.length:
                self.playing = False
                return
            if not self.replay_animating:
                self.jump_replay(1)
        else:
            # Live mode: auto-step using model or random legal move
            if self.base_env is None:
                self.playing = False
                return
            mask = self.base_env.get_action_mask()
            if mask is None or not np.any(mask):
                self.playing = False
                return
            if self.model is not None:
                act, _ = self.model.predict(self.obs, deterministic=True, action_masks=mask)
                self._apply_action(int(act))
            else:
                valid = np.flatnonzero(mask)
                if valid.size == 0:
                    self.playing = False
                    return
                self._apply_action(int(np.random.choice(valid)))

        # schedule next tick
        self.root.after(self._delay_ms(), self._play_tick)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Tkinter GUI for Tigers & Goats (unified env).")
    parser.add_argument("--env", choices=["normal", "battle"], default="normal",
                        help="Opponent mode: normal=greedy tiger, battle=smart tiger.")
    parser.add_argument("--model", type=str, default=None, help="Optional MaskablePPO goat model path (.zip).")
    args = parser.parse_args()

    tiger_ai = TIGER_AI_GREEDY if args.env == "normal" else TIGER_AI_SMART

    gui = TigersGoatsGUI(
        tiger_ai=tiger_ai,
        model_path=args.model,
    )
    gui.run()


if __name__ == "__main__":
    main()
