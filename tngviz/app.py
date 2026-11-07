"""Tigers & Goats board with perfect-play analysis for both sides, forced-win display and engine play."""

from __future__ import annotations

import argparse
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from . import ui  # noqa: E402
from .game import (DEFAULT_TABLES, FLIP, GOAT, OTHER, PLY_LIMIT, RANK, SIDE_NAME, TIGER, Analysis,  # noqa: E402
                   Analyzer, Game, Line, Move, Outcome, Position, describe, edges, goat_count_text,
                   legal_moves, parse_position, pill_text, position_to_text, square_name, start_position)
from .models import PREFIX as MODEL_PREFIX, ModelInfo, ModelPlayer, discover  # noqa: E402

HUMAN, PERFECT, RANDOM = "human", "perfect", "random"
PLAYER_LABEL = {HUMAN: "Human", PERFECT: "Perfect", RANDOM: "Random"}
PLAYER_NOTE = {HUMAN: "you move", PERFECT: "solved tables", RANDOM: "uniform legal move"}
DELAYS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
ANIM_SECONDS = 0.18
FLASH_SECONDS = 4.0
PV_BUDGET = 0.006            # seconds of continuation lookups per frame
LINES_MAX = 25
PV_MAX = 30
MIN_WINDOW = (960, 600)
MODAL_KEYS_REPEAT = (300, 45)

FOOTER_ITEMS = (
    "Click or drag to move", "←/→ or wheel: step", "Home/End", "Space pause", "N new game",
    "1/G goats, 2/T tigers: player", "R limits", "A arrows", "E analysis", "V values", "L labels",
    "+/- lines", "Ctrl+C position", "Ctrl+Shift+C game", "Ctrl+V paste",
)

MARGIN = 16
BAR_W = 28
BAR_GAP = 10
PANEL_MIN = 440
PANEL_PREF = 560
ROW_H = 32
GAP = 6
LINE_H = 24
MOVE_ROW_H = 22

# Node layout of gui.py (board units): the apex b0 above a 6 x 3 grid and a 4-node bottom row.
LAYOUT = {
    0: (280, 40),
    1: (80, 120), 2: (160, 120), 3: (240, 120), 4: (320, 120), 5: (400, 120), 6: (480, 120),
    7: (80, 200), 8: (160, 200), 9: (240, 200), 10: (320, 200), 11: (400, 200), 12: (480, 200),
    13: (80, 280), 14: (160, 280), 15: (240, 280), 16: (320, 280), 17: (400, 280), 18: (480, 280),
    19: (160, 360), 20: (240, 360), 21: (320, 360), 22: (400, 360),
}
X0, Y0 = 80, 40
PAD_U = 48                    # board padding in units (room for pieces and labels)
BOARD_UW, BOARD_UH = 400 + 2 * PAD_U, 320 + 2 * PAD_U
PIECE_U = 25                  # piece radius in units (nodes are 80 units apart)

LASTMOVE_RGBA = (215, 200, 40, 150)
SELECT_RGBA = (30, 120, 50, 210)
TARGET_RGBA = (30, 110, 45, 120)
HOVER_RGBA = (255, 255, 255, 60)
CAPTURED_RGBA = (200, 40, 40, 150)
ARROW_BEST = (21, 135, 45, 200)
ARROW_ALT = (21, 135, 45, 90)
ARROW_THREAT = (210, 50, 50, 170)
ARROW_HOVER = (40, 115, 225, 210)
RESULT_COLOR = {"win": ui.WIN_C, "draw": ui.DRAW_C, "loss": ui.LOSS_C}


@dataclass
class Anim:
    move: Move
    start: float


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class App:
    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        pygame.init()
        pygame.display.set_caption("Tigers & Goats · perfect play")
        self.screen = pygame.display.set_mode(self._initial_size(args.size), pygame.RESIZABLE)
        self._apply_min_size()
        pygame.key.set_repeat(*MODAL_KEYS_REPEAT)
        self.fonts = ui.Fonts()
        self.clock = pygame.time.Clock()
        self.rng = random.Random()

        # perfect-play tables
        self.an: Optional[Analyzer] = None
        self.an_error = ""
        try:
            self.an = Analyzer(Path(args.tables))
        except Exception as exc:  # missing or unreadable tables: the board still works
            self.an_error = f"Perfect play unavailable: {exc}"
            print(self.an_error, file=sys.stderr)

        # game and options
        self.first = args.first
        self.limits = args.limits == "tngenv"
        self.models: Dict[str, List[ModelInfo]] = discover()
        self.model_players: Dict[str, ModelPlayer] = {}
        self.players: Dict[str, str] = {GOAT: HUMAN, TIGER: HUMAN}
        for side, spec in ((GOAT, args.goats), (TIGER, args.tigers)):
            kind = self._resolve_player(side, spec)
            if kind == PERFECT and self.an is None:
                kind = HUMAN
            self._set_player(side, kind, announce=False)
        self.paused = False
        self.lines_max = _clamp(args.lines, 1, LINES_MAX)
        self.pv_len = _clamp(args.pv, 0, PV_MAX)
        self.delays = sorted(set(DELAYS + [max(0.0, args.delay)]))
        self.delay_i = self.delays.index(max(0.0, args.delay))
        self.analysis_on = not args.no_analysis
        self.show_arrows = True
        self.show_values = True
        self.labels = args.labels
        self.game = Game(start_position(self.first), self.first)
        self.outcome: Optional[Outcome] = None
        self.cur: Optional[Analysis] = None
        self.threat: Optional[Analysis] = None
        self.pv_jobs: List[Iterator[None]] = []
        self.engine_due = 0.0
        self.annotations: Dict[int, str] = {}

        # interaction state
        self.mouse = (0, 0)
        self.selected: Optional[int] = None
        self.dragging = False
        self.press_pos = (0, 0)
        self.anim: Optional[Anim] = None
        self.flash_msg = ""
        self.flash_color = ui.MUTED
        self.flash_until = 0.0
        self.bar_frac = 0.5
        self.bar_target = 0.5
        self.bar_label = ""
        self.line_rows: List[Tuple[pygame.Rect, Line, bool]] = []
        self.hover_line: Optional[Line] = None
        self.move_cells: List[Tuple[pygame.Rect, int]] = []
        self.movelist_rect = pygame.Rect(0, 0, 0, 0)
        self.movelist_scroll = 0
        self.movelist_follow = True
        self.u = 1.0
        self.edge_list = edges()

        self._build_controls()
        self._layout(*self.screen.get_size())
        if args.position or args.game:
            text = args.position
            if args.game:
                try:
                    text = Path(args.game).read_text()
                except OSError as exc:
                    self._flash(f"Could not read {args.game}: {exc}", ui.BAD)
                    text = None
            if text and not self._load_text(text):
                print(f"Could not load {text!r}", file=sys.stderr)
        self._on_position_changed()
        if self.an_error:
            self._flash(self.an_error, ui.BAD)

    @staticmethod
    def _initial_size(size_arg: Optional[str]) -> Tuple[int, int]:
        if size_arg:
            try:
                w, h = (int(v) for v in size_arg.lower().split("x"))
                return max(MIN_WINDOW[0], w), max(MIN_WINDOW[1], h)
            except ValueError:
                print(f"Ignoring bad --size {size_arg!r}; expected WxH", file=sys.stderr)
        try:
            dw, dh = pygame.display.get_desktop_sizes()[0]
        except Exception:
            dw, dh = 1600, 1000
        h = _clamp(dh - 100, MIN_WINDOW[1], 900)
        u = (h - 2 * MARGIN) / BOARD_UH
        w = round(MARGIN + BAR_W + BAR_GAP + BOARD_UW * u + MARGIN + PANEL_PREF + MARGIN)
        return min(w, max(MIN_WINDOW[0], dw - 40)), h

    @staticmethod
    def _apply_min_size() -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                pygame.Window.from_display_module().minimum_size = MIN_WINDOW
        except Exception:
            pass

    def _build_controls(self) -> None:
        B = ui.Button
        has_moves = lambda: self.game.ply < len(self.game.moves)  # noqa: E731
        self.btn_goats = B(lambda: self._fit(f"Goats: {self._player_label(self.players[GOAT])} ▾",
                                             f"G: {self._player_label(self.players[GOAT])} ▾", self.btn_goats),
                           lambda: None, active=lambda: self.players[GOAT] != HUMAN)
        self.btn_tigers = B(lambda: self._fit(f"Tigers: {self._player_label(self.players[TIGER])} ▾",
                                              f"T: {self._player_label(self.players[TIGER])} ▾", self.btn_tigers),
                            lambda: None, active=lambda: self.players[TIGER] != HUMAN)
        self.dropdowns = {
            side: ui.Dropdown(btn, lambda side=side: self._player_options(side),
                              lambda side=side: self.players[side],
                              lambda kind, side=side: self._set_player(side, kind))
            for side, btn in ((GOAT, self.btn_goats), (TIGER, self.btn_tigers))
        }
        self.btn_first = B(lambda: self._fit(f"First: {SIDE_NAME[self.first]}",
                                             f"{SIDE_NAME[self.first]} first", self.btn_first), self._toggle_first)
        self.btn_limits = B(lambda: self._fit(f"Limits: {'TnGEnv' if self.limits else 'None'}",
                                              "TnGEnv" if self.limits else "No limits", self.btn_limits),
                            self._toggle_limits, active=lambda: self.limits)
        self.btn_new = B(lambda: "New game", self._new_game)
        self.btn_back = B(lambda: "◀ Back", lambda: self._goto(self.game.ply - 1),
                          enabled=lambda: self.game.ply > 0)
        self.btn_fwd = B(lambda: "Next ▶", lambda: self._goto(self.game.ply + 1), enabled=has_moves)
        self.btn_pause = B(lambda: "Resume" if self.paused else "Pause", self._toggle_pause,
                           enabled=self._any_engine, active=lambda: self.paused)
        self.btn_analysis = B(lambda: f"Analysis: {'On' if self.analysis_on else 'Off'}", self._toggle_analysis,
                              enabled=lambda: self.an is not None, active=lambda: self.analysis_on)
        self.btn_arrows = B(lambda: f"Arrows: {'On' if self.show_arrows else 'Off'}", self._toggle_arrows,
                            active=lambda: self.show_arrows)
        self.btn_values = B(lambda: f"Values: {'On' if self.show_values else 'Off'}", self._toggle_values,
                            active=lambda: self.show_values)
        self.btn_labels = B(lambda: self._fit(f"Labels: {self.labels.title()}", self.labels.title(),
                                              self.btn_labels), self._cycle_labels)
        self.btn_copy = B(lambda: "Copy position", self._copy_position)
        self.btn_copy_game = B(lambda: "Copy game", self._copy_game)
        self.btn_paste = B(lambda: "Paste", self._paste)
        self.step_lines = ui.Stepper("Lines", lambda: str(self.lines_max),
                                     lambda: self._set_lines(self.lines_max - 1),
                                     lambda: self._set_lines(self.lines_max + 1))
        self.step_pv = ui.Stepper("Line length", lambda: str(self.pv_len),
                                  lambda: self._set_pv(self.pv_len - 1), lambda: self._set_pv(self.pv_len + 1))
        self.step_delay = ui.Stepper("Delay", lambda: f"{self.delay:g}s",
                                     lambda: self._set_delay(-1), lambda: self._set_delay(+1))
        self.button_rows: List[Sequence[ui.Button]] = [
            (self.btn_goats, self.btn_tigers, self.btn_first, self.btn_limits),
            (self.btn_new, self.btn_back, self.btn_fwd, self.btn_pause),
            (self.btn_analysis, self.btn_arrows, self.btn_values, self.btn_labels),
            (self.btn_copy, self.btn_copy_game, self.btn_paste),
        ]
        self.steppers = (self.step_lines, self.step_pv, self.step_delay)

    def _fit(self, full: str, short: str, button: ui.Button) -> str:
        return full if self.fonts.regular.size(full)[0] <= button.rect.w - 12 else short

    def _all_buttons(self) -> List[ui.Button]:
        out = [b for row in self.button_rows for b in row]
        for st in self.steppers:
            out.extend(st.buttons())
        return out

    def _layout(self, w: int, h: int) -> None:
        fixed = MARGIN * 3 + BAR_W + BAR_GAP
        avail_h = h - 2 * MARGIN
        u_fit = avail_h / BOARD_UH
        u_max = (w - fixed - PANEL_MIN) / BOARD_UW          # the panel at its minimum width
        u_pref = (w - fixed - PANEL_PREF) / BOARD_UW        # the panel at its preferred width
        u_big = min(u_fit, u_max)
        u = max(0.5, u_big if u_pref < 0.85 * u_big else min(u_big, u_pref))   # wide panel if it costs little
        bw, bh = round(BOARD_UW * u), round(BOARD_UH * u)
        top = MARGIN + max(0, (avail_h - bh) // 2)
        self.eval_rect = pygame.Rect(MARGIN, top, BAR_W, bh)
        self.board_rect = pygame.Rect(MARGIN + BAR_W + BAR_GAP, top, bw, bh)
        px = self.board_rect.right + MARGIN
        self.panel_rect = pygame.Rect(px, MARGIN, max(200, w - px - MARGIN), h - 2 * MARGIN)
        self.overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        if abs(u - self.u) > 1e-6 or not hasattr(self, "sprites"):
            self.u = u
            self._build_board_assets()
        self.node_xy = {i: (self.board_rect.x + (PAD_U + x - X0) * u, self.board_rect.y + (PAD_U + y - Y0) * u)
                        for i, (x, y) in LAYOUT.items()}

        p = self.panel_rect
        y = p.y
        rows: List = [self.button_rows[0], self.button_rows[1], self.steppers, self.button_rows[2],
                      self.button_rows[3]]
        for row in rows:
            n = len(row)
            bw_ = (p.w - GAP * (n - 1)) / n
            for i, item in enumerate(row):
                r = pygame.Rect(round(p.x + i * (bw_ + GAP)), y, round(bw_), ROW_H)
                if isinstance(item, ui.Stepper):
                    item.layout(r)
                else:
                    item.rect = r
            y += ROW_H + GAP
        self.controls_bottom = y

    def _build_board_assets(self) -> None:
        u = self.u
        self.R = PIECE_U * u
        bold = self.fonts._bold_path or self.fonts._sans_path
        self.label_font = pygame.font.Font(bold, max(9, round(11 * u)))
        self.badge_font = pygame.font.Font(bold, max(9, round(12 * u)))
        self.sprites = {GOAT: self._piece_sprite(GOAT), TIGER: self._piece_sprite(TIGER)}

    def _piece_sprite(self, side: str) -> pygame.Surface:
        """Draw a piece at 3x and scale down for smooth edges."""
        k = 3
        r = self.R * k
        size = math.ceil(2 * r + 4 * k)
        big = pygame.Surface((size, size), pygame.SRCALPHA)
        c = size / 2
        if side == TIGER:
            fill, dark = ui.TIGER_FILL, ui.TIGER_DARK
        else:
            fill, dark = ui.GOAT_FILL, ui.GOAT_DARK
        pygame.draw.circle(big, (0, 0, 0, 70), (c + 1.5 * k, c + 2 * k), r)            # shadow
        pygame.draw.circle(big, dark, (c, c), r)
        pygame.draw.circle(big, fill, (c, c), r - 2.2 * k)
        w = max(2, round(r * 0.13))
        if side == TIGER:
            for sgn in (-1, 1):                        # stripes from both sides
                for ang in (-32, 0, 32):
                    a = math.radians(ang)
                    x0 = c + sgn * (r - 2.2 * k) * math.cos(a)
                    y0 = c + (r - 2.2 * k) * math.sin(a)
                    x1 = c + sgn * (r * 0.55) * math.cos(a * 0.8)
                    y1 = c + (r * 0.55) * math.sin(a * 0.8)
                    pygame.draw.line(big, dark, (x0, y0), (x1, y1), w)
            for sgn in (-1, 1):                        # eyes and nose
                pygame.draw.circle(big, dark, (c + sgn * r * 0.2, c - r * 0.12), r * 0.08)
            pygame.draw.polygon(big, dark, [(c - r * 0.1, c + r * 0.12), (c + r * 0.1, c + r * 0.12),
                                            (c, c + r * 0.25)])
        else:
            for sgn in (-1, 1):                        # horns, eyes
                pygame.draw.line(big, dark, (c + sgn * r * 0.16, c - r * 0.3), (c + sgn * r * 0.42, c - r * 0.66), w)
                pygame.draw.circle(big, dark, (c + sgn * r * 0.22, c - r * 0.02), r * 0.075)
            pygame.draw.ellipse(big, (210, 205, 195), (c - r * 0.2, c + r * 0.14, r * 0.4, r * 0.3))
        small = math.ceil(size / k)
        return pygame.transform.smoothscale(big, (small, small))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def delay(self) -> float:
        return self.delays[self.delay_i]

    def _any_engine(self) -> bool:
        return any(v != HUMAN for v in self.players.values())

    def _flash(self, msg: str, color=ui.MUTED) -> None:
        self.flash_msg = msg
        self.flash_color = color
        self.flash_until = time.monotonic() + FLASH_SECONDS

    def _node_at(self, pos: Tuple[int, int]) -> Optional[int]:
        best, best_d = None, (self.R * 1.15) ** 2
        for i, (x, y) in self.node_xy.items():
            d = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _can_human_move(self) -> bool:
        return (self.outcome is None and self.anim is None
                and (self.players[self.game.pos.mover] == HUMAN or self.paused))

    def _engine_should_move(self) -> bool:
        return (self.outcome is None and not self.paused and self.anim is None
                and self.players[self.game.pos.mover] != HUMAN)

    def _moves_from(self, sq: Optional[int]) -> List[Move]:
        return [m for m in legal_moves(self.game.pos) if m.frm == sq]

    def _line_for(self, move: Move) -> Optional[Line]:
        if self.cur is None:
            return None
        for ln in self.cur.lines:
            if ln.move.action == move.action:
                return ln
        return None

    # ------------------------------------------------------------------
    # game flow
    # ------------------------------------------------------------------
    def _on_position_changed(self) -> None:
        g = self.game
        self.selected = None
        self.dragging = False
        self.outcome = g.outcome(limits=self.limits)
        self.cur = self.threat = None
        self.pv_jobs = []
        pos = g.pos
        if self.an is not None and pos.captured < 6:
            plies = g.plies_played()
            self.cur = self.an.analyse(pos, plies, self.limits)
            if self.outcome is None or self.outcome.winner is None:
                swapped = Position(pos.tm, pos.gm, pos.captured, pos.placed, OTHER[pos.mover])
                self.threat = self.an.analyse(swapped, plies, self.limits)
            self._start_pv_jobs()
            for i in range(1, g.ply + 1):
                self._annotate(i)
        self.engine_due = time.monotonic() + self.delay

    def _start_pv_jobs(self) -> None:
        self.pv_jobs = []
        if self.an is None or self.pv_len == 0:
            return
        g = self.game
        seen = g.seen()
        for an in (self.cur, self.threat):
            if an is None:
                continue
            for ln in an.lines[: self.lines_max]:
                self.pv_jobs.append(self.an.pv(ln, g.plies_played(), seen, self.pv_len, self.rng))

    def _annotate(self, i: int) -> None:
        """'?' marks a move that throws away a win or a draw, '??' one that turns a win into a loss."""
        if i in self.annotations or self.an is None:
            return
        before = self.game.positions[i - 1]
        mv = self.game.moves[i - 1]
        if before.captured >= 6:
            return
        was = self.an.value(before)[0]
        now = "win" if mv.nxt is None else FLIP[self.an.value(mv.nxt)[0]]
        drop = RANK[now] - RANK[was]
        self.annotations[i] = "??" if drop == 2 else "?" if drop == 1 else ""

    def _play_move(self, move: Move, *, animate: bool, human: bool) -> None:
        g = self.game
        if g.ply < len(g.moves):
            for i in range(g.ply + 1, len(g.moves) + 1):
                self.annotations.pop(i, None)
        g.push(move)
        if human and self.paused and self.players[OTHER[move.side]] != HUMAN:
            self.paused = False
        self.anim = Anim(move, time.monotonic()) if animate else None
        self.movelist_follow = True
        self._on_position_changed()

    def _goto(self, ply: int) -> None:
        g = self.game
        ply = _clamp(ply, 0, len(g.moves))
        if ply == g.ply:
            return
        animate = ply == g.ply + 1
        g.ply = ply
        if ply < len(g.moves) and self._any_engine() and not self.paused:
            self.paused = True
            self._flash("Engines paused while you look back. Move, or press Space, to continue.")
        self.anim = Anim(g.moves[ply - 1], time.monotonic()) if animate else None
        self.movelist_follow = True
        self._on_position_changed()

    def _new_game(self, root: Optional[Position] = None) -> None:
        self.game = Game(root or start_position(self.first), self.first)
        self.annotations = {}
        self.anim = None
        self.paused = False
        self.movelist_scroll = 0
        self._on_position_changed()

    # ------------------------------------------------------------------
    # players
    # ------------------------------------------------------------------
    def _model_info(self, kind: str) -> Optional[ModelInfo]:
        for infos in self.models.values():
            for info in infos:
                if info.key == kind:
                    return info
        return None

    def _player_label(self, kind: str) -> str:
        info = self._model_info(kind)
        return info.label if info else PLAYER_LABEL.get(kind, kind)

    def _player_options(self, side: str) -> List[Tuple[str, str, str]]:
        kinds = [HUMAN, PERFECT, RANDOM] if self.an is not None else [HUMAN, RANDOM]
        out = [(k, PLAYER_LABEL[k], PLAYER_NOTE[k]) for k in kinds]
        out += [(info.key, info.label, "trained model") for info in self.models[side]]
        return out

    def _resolve_player(self, side: str, spec: str) -> str:
        """A --goats/--tigers value: human, perfect, random, or a model of that side by file name or label
        (case-insensitive; a unique substring is enough)."""
        s = spec.strip().lower()
        if s in PLAYER_LABEL:
            return s
        s = s[len(MODEL_PREFIX):] if s.startswith(MODEL_PREFIX) else s
        infos = self.models[side]
        exact = [i for i in infos if s in (i.path.stem.lower(), i.label.lower())]
        part = exact or [i for i in infos if s in i.path.stem.lower() or s in i.label.lower()]
        if len(part) == 1:
            return part[0].key
        names = ", ".join(i.path.stem for i in infos) or "none found"
        why = f"matches several {side} models" if part else f"matches no {side} model"
        raise SystemExit(f"--{SIDE_NAME[side].lower()} {spec!r} {why}. Use human, perfect, random or one of: {names}")

    def _set_player(self, side: str, kind: str, announce: bool = True) -> None:
        info = self._model_info(kind)
        if info is not None:
            mp = self.model_players.get(kind)
            if mp is None or mp.error:
                mp = self.model_players[kind] = ModelPlayer(info)
            mp.start()
            if announce and not mp.ready:
                self._flash(f"Loading {info.label} for the {SIDE_NAME[side].lower()}…")
        self.players[side] = kind
        if announce:
            self.engine_due = time.monotonic() + self.delay

    def _cycle_player(self, side: str) -> None:
        kinds = [k for k, _, _ in self._player_options(side)]
        cur = self.players[side]
        self._set_player(side, kinds[(kinds.index(cur) + 1) % len(kinds)] if cur in kinds else HUMAN)

    def _engine_move(self, side: str) -> Optional[Move]:
        """The engine's move for `side`, or None while a model is still loading."""
        kind = self.players[side]
        moves = legal_moves(self.game.pos)
        if kind == PERFECT and self.cur is not None:
            return self.an.choose(self.cur, self.game.seen(), self.rng)
        mp = self.model_players.get(kind)
        if mp is not None:
            if mp.error:
                self._flash(f"{mp.info.label}: {mp.error}. The {SIDE_NAME[side].lower()} are now Human.", ui.BAD)
                print(mp.error, file=sys.stderr)
                self.players[side] = HUMAN
                return None
            if not mp.ready:
                return None
            mv = mp.choose(self.game.pos, moves)
            if mv is not None:
                return mv
            self._flash(f"{mp.info.label} proposed an illegal move; played a random one.", ui.WARN)
        return self.rng.choice(moves)

    def _toggle_first(self) -> None:
        self.first = OTHER[self.first]
        if not self.game.moves and self.game.root == start_position(OTHER[self.first]):
            self._new_game()
            self._flash(f"New game: the {SIDE_NAME[self.first].lower()} move first.")
        elif not self.game.moves:
            self.game = Game(self.game.root, self.first)
            self._on_position_changed()
            self._flash(f"Plies are now counted as if the {SIDE_NAME[self.first].lower()} moved first.")
        else:
            self._flash(f"The {SIDE_NAME[self.first].lower()} move first from the next new game (N).")

    def _toggle_limits(self) -> None:
        self.limits = not self.limits
        self._on_position_changed()
        self._flash("TnGEnv limits: 100 turns (200 plies) and, with goats first, the 4th repeat of a board end "
                    "the game as a draw." if self.limits else "No limits: the solved game, which can go on forever.")

    def _toggle_pause(self) -> None:
        if self._any_engine():
            self.paused = not self.paused
            self.engine_due = time.monotonic() + self.delay

    def _toggle_analysis(self) -> None:
        self.analysis_on = not self.analysis_on

    def _toggle_arrows(self) -> None:
        self.show_arrows = not self.show_arrows

    def _toggle_values(self) -> None:
        self.show_values = not self.show_values

    def _cycle_labels(self) -> None:
        order = ["names", "numbers", "off"]
        self.labels = order[(order.index(self.labels) + 1) % len(order)]

    def _set_lines(self, n: int) -> None:
        self.lines_max = _clamp(n, 1, LINES_MAX)
        self._start_pv_jobs()

    def _set_pv(self, n: int) -> None:
        self.pv_len = _clamp(n, 0, PV_MAX)
        self._start_pv_jobs()

    def _set_delay(self, step: int) -> None:
        self.delay_i = _clamp(self.delay_i + step, 0, len(self.delays) - 1)

    # ------------------------------------------------------------------
    # clipboard
    # ------------------------------------------------------------------
    def _clip_put(self, text: str) -> bool:
        try:
            pygame.scrap.put_text(text)
            return True
        except Exception:
            pass
        for cmd in (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-b", "-i"]):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, input=text.encode(), timeout=2, check=True)
                    return True
                except Exception:
                    continue
        return False

    def _clip_get(self) -> str:
        try:
            text = pygame.scrap.get_text()
            if text:
                return text
        except Exception:
            pass
        for cmd in (["wl-paste", "-n"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "-b", "-o"]):
            if shutil.which(cmd[0]):
                try:
                    out = subprocess.run(cmd, capture_output=True, timeout=2, check=True).stdout
                    if out:
                        return out.decode(errors="replace")
                except Exception:
                    continue
        return ""

    def _copy_position(self) -> None:
        text = position_to_text(self.game.pos)
        ok = self._clip_put(text)
        print(text)
        self._flash("Position copied." if ok else "No clipboard; the position was printed to the terminal.",
                    ui.GOOD if ok else ui.WARN)

    def game_text(self) -> str:
        g = self.game
        parts = []
        turn = 1
        side_first = g.first
        for i, mv in enumerate(g.moves):
            if mv.side == side_first or i == 0:
                parts.append(f"{turn}.")
                if mv.side != side_first:
                    parts.append("…")
                turn += 1
            parts.append(mv.text("numbers"))
        lines = [f"start: {position_to_text(g.root)}", f"first: {g.first}", "moves: " + " ".join(parts)]
        if self.outcome is not None and g.ply == len(g.moves):
            who = SIDE_NAME[self.outcome.winner] + " win" if self.outcome.winner else "draw"
            lines.append(f"result: {who} ({self.outcome.reason})")
        return "\n".join(lines)

    def _copy_game(self) -> None:
        text = self.game_text()
        ok = self._clip_put(text)
        print(text)
        self._flash("Game copied." if ok else "No clipboard; the game was printed to the terminal.",
                    ui.GOOD if ok else ui.WARN)

    def _paste(self) -> None:
        text = self._clip_get()
        if not text.strip():
            self._flash("The clipboard is empty.", ui.WARN)
            return
        self._load_text(text)

    def _load_text(self, text: str) -> bool:
        """Load a position ('tigers=… goats=… captured=… mover=…') or a game copied with Copy game."""
        try:
            start = re.search(r"start:\s*(.+)", text)
            first = re.search(r"first:\s*(goat|tiger)", text)
            moves = re.search(r"moves:\s*(.*)", text)
            if moves is None:
                root = parse_position(text)
                self._new_game(root)
                self._flash("Position loaded.", ui.GOOD)
                return True
            if first:
                self.first = first.group(1)
            root = parse_position(start.group(1)) if start else start_position(self.first)
            self.game = Game(root, self.first)
            self.annotations = {}
            for tok in moves.group(1).split():
                if re.fullmatch(r"\d+\.|…|\.\.\.", tok):
                    continue
                match = [m for m in legal_moves(self.game.pos)
                         if tok in (m.text("numbers"), m.text("names"))]
                if not match or self.game.outcome(limits=False) is not None:
                    raise ValueError(f"illegal move {tok!r} at ply {self.game.ply + 1}")
                self.game.push(match[0])
            self.game.ply = 0 if self.game.moves else self.game.ply
            self.paused = bool(self.game.moves) and self._any_engine()
            self.anim = None
            self._on_position_changed()
            self._flash(f"Game loaded: {len(self.game.moves)} plies. Use → / End to step through.", ui.GOOD)
            return True
        except Exception as exc:
            self._flash(f"Could not load: {exc}", ui.BAD)
            return False

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def _on_left_down(self, pos: Tuple[int, int]) -> None:
        h = self.screen.get_height()
        for dd in self.dropdowns.values():
            if dd.open:
                # a click outside an open menu only closes it, unless it opens the other menu
                if not dd.click(pos, h):
                    for other in self.dropdowns.values():
                        if other is not dd and other.button.hit(pos):
                            other.toggle()
                return
        for b in self._all_buttons():
            if b.hit(pos):
                b.on_click()
                return
        for rect, line, playable in self.line_rows:
            if rect.collidepoint(pos):
                if playable and self._can_human_move():
                    self._play_move(line.move, animate=True, human=True)
                elif playable:
                    self._flash("It is the engine's move. Pause (Space) to move for it.", ui.WARN)
                return
        for rect, ply in self.move_cells:
            if rect.collidepoint(pos):
                self._goto(ply)
                return
        if self.board_rect.collidepoint(pos):
            self._board_press(pos)

    def _board_press(self, pos: Tuple[int, int]) -> None:
        sq = self._node_at(pos)
        if sq is None:
            self.selected = None
            return
        if self.outcome is not None:
            self._flash(f"Game over: {self._outcome_text()}. Step back or start a new game.", ui.WARN)
            return
        if not self._can_human_move():
            if self.anim is None:
                self._flash("It is the engine's move. Pause (Space) to move for it.", ui.WARN)
            return
        g = self.game
        moves = legal_moves(g.pos)
        if self.selected is not None:
            for m in moves:
                if m.frm == self.selected and m.to == sq:
                    self._play_move(m, animate=True, human=True)
                    return
        place = [m for m in moves if m.frm is None and m.to == sq]
        if place:
            self._play_move(place[0], animate=True, human=True)
            return
        if any(m.frm == sq for m in moves):
            self.selected = sq
            self.dragging = True
            self.press_pos = pos
            return
        own = g.pos.gm if g.pos.mover == GOAT else g.pos.tm
        if (own >> sq) & 1:
            self._flash("That piece has no legal move.", ui.WARN)
        self.selected = None

    def _board_release(self, pos: Tuple[int, int]) -> None:
        if not self.dragging:
            return
        self.dragging = False
        if self.selected is None:
            return
        sq = self._node_at(pos)
        if sq is None or sq == self.selected:
            return
        for m in legal_moves(self.game.pos):
            if m.frm == self.selected and m.to == sq:
                self._play_move(m, animate=False, human=True)
                return

    def _on_wheel(self, dy: int) -> None:
        if self.movelist_rect.collidepoint(self.mouse):
            self.movelist_follow = False
            self.movelist_scroll -= dy
        elif self.board_rect.collidepoint(self.mouse) or self.eval_rect.collidepoint(self.mouse):
            self._goto(self.game.ply - dy)

    def _on_key(self, ev: pygame.event.Event) -> bool:
        k = ev.key
        ctrl = bool(ev.mod & pygame.KMOD_CTRL)
        shift = bool(ev.mod & pygame.KMOD_SHIFT)
        if ctrl and k == pygame.K_q:
            return False
        if ctrl and k == pygame.K_c:
            self._copy_game() if shift else self._copy_position()
        elif ctrl and k == pygame.K_v:
            self._paste()
        elif k == pygame.K_LEFT:
            self._goto(self.game.ply - 1)
        elif k == pygame.K_RIGHT:
            self._goto(self.game.ply + 1)
        elif k == pygame.K_HOME:
            self._goto(0)
        elif k == pygame.K_END:
            self._goto(len(self.game.moves))
        elif k == pygame.K_SPACE:
            self._toggle_pause()
        elif k == pygame.K_n:
            self._new_game()
        elif k in (pygame.K_1, pygame.K_g):
            self._cycle_player(GOAT)
        elif k in (pygame.K_2, pygame.K_t):
            self._cycle_player(TIGER)
        elif k == pygame.K_r:
            self._toggle_limits()
        elif k == pygame.K_a:
            self._toggle_arrows()
        elif k == pygame.K_e and self.an is not None:
            self._toggle_analysis()
        elif k == pygame.K_v:
            self._toggle_values()
        elif k == pygame.K_l:
            self._cycle_labels()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._set_lines(self.lines_max + 1)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._set_lines(self.lines_max - 1)
        elif k == pygame.K_ESCAPE:
            for dd in self.dropdowns.values():
                dd.open = False
            self.selected = None
            self.dragging = False
        return True

    # ------------------------------------------------------------------
    # per-frame update
    # ------------------------------------------------------------------
    def _update(self, dt: float) -> None:
        now = time.monotonic()
        if self.anim and now - self.anim.start >= ANIM_SECONDS:
            self.anim = None
            self.engine_due = max(self.engine_due, now + min(self.delay, 0.05))
        if self._engine_should_move() and now >= self.engine_due:
            mv = self._engine_move(self.game.pos.mover)
            if mv is not None:
                self._play_move(mv, animate=self.delay > 0, human=False)

        t_end = time.perf_counter() + PV_BUDGET
        while self.pv_jobs and time.perf_counter() < t_end:
            try:
                next(self.pv_jobs[0])
            except StopIteration:
                self.pv_jobs.pop(0)

        # eval bar, from the goats' side (goats at the bottom)
        o, an = self.outcome, self.cur
        if o is not None and o.winner is not None:
            self.bar_target = 1.0 if o.winner == GOAT else 0.0
            self.bar_label = "G" if o.winner == GOAT else "T"
        elif o is not None:
            self.bar_target, self.bar_label = 0.5, "="
        elif not self.analysis_on or an is None:
            self.bar_target, self.bar_label = 0.5, ""
        elif an.result == "draw":
            self.bar_target, self.bar_label = 0.5, "="
        else:
            goats_win = (an.result == "win") == (an.pos.mover == GOAT)
            full = 0.75 if an.over_limit else 1.0
            self.bar_target = full if goats_win else 1.0 - full
            self.bar_label = str(an.d)
        self.bar_frac += (self.bar_target - self.bar_frac) * min(1.0, dt * 8)

        self.hover_line = None
        for rect, line, _ in self.line_rows:
            if rect.collidepoint(self.mouse):
                self.hover_line = line
                break

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _draw(self) -> None:
        self.screen.fill(ui.BG)
        self._draw_eval_bar()
        self._draw_board()
        self._draw_panel()
        for dd in self.dropdowns.values():
            dd.draw_menu(self.screen, self.fonts, self.mouse)

    def _draw_eval_bar(self) -> None:
        r, s = self.eval_rect, self.screen
        goat_h = round(r.h * _clamp(self.bar_frac, 0.0, 1.0))
        pygame.draw.rect(s, ui.TIGER_FILL, r)
        pygame.draw.rect(s, ui.GOAT_FILL, pygame.Rect(r.x, r.bottom - goat_h, r.w, goat_h))
        pygame.draw.line(s, (90, 90, 96), (r.x, r.centery), (r.right - 1, r.centery), 2)
        pygame.draw.rect(s, ui.BORDER, r, width=1)
        ui.draw_text(s, self.fonts.bar, "T", ui.MUTED, (r.centerx, r.y - 2), anchor="midbottom")
        ui.draw_text(s, self.fonts.bar, "G", ui.MUTED, (r.centerx, r.bottom + 2), anchor="midtop")
        if self.bar_label:
            goats_ahead = self.bar_target > 0.5
            if self.bar_target == 0.5:
                img = self.fonts.bar.render(self.bar_label, True, (30, 30, 30))
                rect = img.get_rect(center=(r.centerx, r.centery + 10))
            elif goats_ahead:
                img = self.fonts.bar.render(self.bar_label, True, (30, 30, 30))
                rect = img.get_rect(midbottom=(r.centerx, r.bottom - 4))
            else:
                img = self.fonts.bar.render(self.bar_label, True, (30, 20, 10))
                rect = img.get_rect(midtop=(r.centerx, r.y + 4))
            s.blit(img, rect)

    def _xy(self, sq: int) -> Tuple[float, float]:
        return self.node_xy[sq]

    def _draw_board(self) -> None:
        s, u, br, g = self.screen, self.u, self.board_rect, self.game
        pygame.draw.rect(s, ui.BOARD_BG, br, border_radius=round(16 * u))
        pygame.draw.rect(s, ui.BOARD_EDGE, br, width=2, border_radius=round(16 * u))
        lw = max(2, round(3 * u))
        for a, b in self.edge_list:
            pygame.draw.line(s, ui.BOARD_LINE, self._xy(a), self._xy(b), lw)
        ov = self.overlay
        ov.fill((0, 0, 0, 0))
        R = self.R
        pos = g.pos

        # last move
        if g.ply > 0:
            last = g.moves[g.ply - 1]
            for sq in (last.frm, last.to):
                if sq is not None:
                    pygame.draw.circle(ov, LASTMOVE_RGBA, self._xy(sq), R + 4 * u, width=max(3, round(5 * u)))
            if last.jumped is not None and (self.anim is None or self.anim.move is not last):
                x, y = self._xy(last.jumped)
                d = R * 0.45
                pygame.draw.line(ov, CAPTURED_RGBA, (x - d, y - d), (x + d, y + d), max(3, round(4 * u)))
                pygame.draw.line(ov, CAPTURED_RGBA, (x - d, y + d), (x + d, y - d), max(3, round(4 * u)))

        # empty nodes and labels
        occ = pos.tm | pos.gm
        for sq, (x, y) in self.node_xy.items():
            if not (occ >> sq) & 1:
                pygame.draw.circle(s, ui.BOARD_LINE, (x, y), max(3, 7 * u))
            if self.labels != "off":
                ui.draw_text(s, self.label_font, square_name(sq, self.labels), (120, 90, 60),
                             (x + R * 0.72, y + R * 0.72), anchor="topleft")
        hover_sq = self._node_at(self.mouse) if self.board_rect.collidepoint(self.mouse) else None
        if hover_sq is not None and self._can_human_move():
            moves = legal_moves(pos)
            if any(m.frm == hover_sq or (m.frm is None and m.to == hover_sq) for m in moves) or (
                    self.selected is not None and any(m.frm == self.selected and m.to == hover_sq for m in moves)):
                pygame.draw.circle(ov, HOVER_RGBA, self._xy(hover_sq), R + 2 * u)
        if self.selected is not None:
            pygame.draw.circle(ov, SELECT_RGBA, self._xy(self.selected), R + 5 * u, width=max(3, round(5 * u)))
        s.blit(ov, (0, 0))

        # pieces
        anim = self.anim
        t = 1.0 if anim is None else _clamp((time.monotonic() - anim.start) / ANIM_SECONDS, 0.0, 1.0)
        t = 1 - (1 - t) ** 3
        for side, mask in ((GOAT, pos.gm), (TIGER, pos.tm)):
            spr = self.sprites[side]
            for sq in range(23):
                if not (mask >> sq) & 1:
                    continue
                if anim is not None and sq == anim.move.to and anim.move.side == side:
                    continue
                if self.dragging and sq == self.selected and self._dist(self.mouse, self.press_pos) > 6:
                    ghost = spr.copy()
                    ghost.set_alpha(90)
                    s.blit(ghost, ghost.get_rect(center=self._xy(sq)))
                    continue
                s.blit(spr, spr.get_rect(center=self._xy(sq)))
        if anim is not None:
            mv = anim.move
            spr = self.sprites[mv.side]
            if mv.jumped is not None:                  # the captured goat fades out
                ghost = self.sprites[GOAT].copy()
                ghost.set_alpha(round(255 * (1 - t)))
                s.blit(ghost, ghost.get_rect(center=self._xy(mv.jumped)))
            if mv.frm is None:                         # a placed goat grows in
                size = max(2, round(spr.get_width() * (0.4 + 0.6 * t)))
                img = pygame.transform.smoothscale(spr, (size, size))
                s.blit(img, img.get_rect(center=self._xy(mv.to)))
            else:
                (x0, y0), (x1, y1) = self._xy(mv.frm), self._xy(mv.to)
                if mv.jumped is not None:              # go through the jumped square (hub jumps bend)
                    xm, ym = self._xy(mv.jumped)
                    if t < 0.5:
                        x, y = x0 + (xm - x0) * 2 * t, y0 + (ym - y0) * 2 * t
                    else:
                        x, y = xm + (x1 - xm) * (2 * t - 1), ym + (y1 - ym) * (2 * t - 1)
                else:
                    x, y = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                s.blit(spr, spr.get_rect(center=(x, y)))

        show = self.analysis_on and self.cur is not None and anim is None and self.outcome is None
        if show and self.show_arrows:
            self._draw_arrows()
        if show and self.show_values:
            self._draw_values()
        if self.dragging and self.selected is not None and self._dist(self.mouse, self.press_pos) > 6:
            side = pos.mover
            spr = self.sprites[side]
            s.blit(spr, spr.get_rect(center=self.mouse))

    @staticmethod
    def _dist(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _badge(self, center: Tuple[float, float], line: Line) -> None:
        text = pill_text(line.result, line.d)
        img = self.badge_font.render(text, True, (255, 255, 255) if not line.over_limit else ui.TEXT)
        w = max(img.get_width() + 10 * self.u, img.get_height() + 6 * self.u)
        rect = pygame.Rect(0, 0, round(w), round(img.get_height() + 4 * self.u))
        rect.center = (round(center[0]), round(center[1]))
        color = RESULT_COLOR[line.result]
        if line.over_limit:                            # forced result beyond the turn limit: outlined
            pygame.draw.rect(self.screen, ui.PANEL, rect, border_radius=rect.h // 2)
            pygame.draw.rect(self.screen, color, rect, width=2, border_radius=rect.h // 2)
        else:
            pygame.draw.rect(self.screen, color, rect, border_radius=rect.h // 2)
            pygame.draw.rect(self.screen, (30, 30, 30), rect, width=1, border_radius=rect.h // 2)
        self.screen.blit(img, img.get_rect(center=rect.center))

    def _draw_values(self) -> None:
        """Results on the board: for the selected piece's moves, every placement square, or (no selection)
        a ring on each movable piece in the colour of its best move."""
        an = self.cur
        if self.selected is not None:
            for ln in an.lines:
                if ln.move.frm == self.selected:
                    self._badge(self._xy(ln.move.to), ln)
            return
        if an.lines and an.lines[0].move.frm is None:
            for ln in an.lines:
                self._badge(self._xy(ln.move.to), ln)
            return
        best_from: Dict[int, Line] = {}
        for ln in an.lines:                            # lines are sorted best first
            best_from.setdefault(ln.move.frm, ln)
        for sq, ln in best_from.items():
            pygame.draw.circle(self.screen, RESULT_COLOR[ln.result], self._xy(sq), self.R + 3 * self.u,
                               width=max(2, round(3 * self.u)))

    def _move_arrow(self, ov: pygame.Surface, mv: Move, color, scale: float = 1.0) -> None:
        u = self.u
        width, head_l, head_w = 9 * u * scale, 24 * u * scale, 26 * u * scale
        if mv.frm is None:                             # a placement: a ring with a plus on the square
            x, y = self._xy(mv.to)
            pygame.draw.circle(ov, color, (x, y), self.R * 0.85, width=max(3, round(6 * u * scale)))
            if self.show_values and self.analysis_on:      # the badge sits in the middle
                return
            d = self.R * 0.4
            pygame.draw.line(ov, color, (x - d, y), (x + d, y), max(3, round(6 * u * scale)))
            pygame.draw.line(ov, color, (x, y - d), (x, y + d), max(3, round(6 * u * scale)))
            return
        start, end = self._xy(mv.frm), self._xy(mv.to)
        if mv.jumped is not None:                      # via the captured goat (hub jumps are not straight)
            mid = self._xy(mv.jumped)
            pygame.draw.line(ov, color, start, mid, max(2, round(width)))
            pygame.draw.circle(ov, color, mid, width / 2)
            start = mid
        dx, dy = end[0] - start[0], end[1] - start[1]
        n = math.hypot(dx, dy) or 1
        end = (end[0] - dx / n * self.R * 0.35, end[1] - dy / n * self.R * 0.35)
        ui.draw_arrow(ov, color, start, end, width, head_l, head_w)

    def _draw_arrows(self) -> None:
        ov = self.overlay
        ov.fill((0, 0, 0, 0))
        an = self.cur
        if self.threat is not None and self.threat.best:
            self._move_arrow(ov, self.threat.lines[0].move, ARROW_THREAT, 0.8)
        best = an.best
        if an.lines and an.lines[0].result != "draw":     # equally fast wins (or equally slow losses)
            for ln in an.lines[1: self.lines_max]:
                if ln in best:
                    self._move_arrow(ov, ln.move, ARROW_ALT, 0.8)
        if an.lines:
            self._move_arrow(ov, an.lines[0].move, ARROW_BEST)
        if self.hover_line is not None:
            self._move_arrow(ov, self.hover_line.move, ARROW_HOVER)
        self.screen.blit(ov, (0, 0))

    # ------------------------------------------------------------------
    # panel
    # ------------------------------------------------------------------
    def _outcome_text(self) -> str:
        o = self.outcome
        if o is None:
            return ""
        who = f"{SIDE_NAME[o.winner]} win" if o.winner else "Draw"
        return f"{who}: {o.reason}"

    def _status_line(self) -> Tuple[str, tuple]:
        if self.outcome is not None:
            return f"Game over · {self._outcome_text()}", ui.WARN
        g = self.game
        mover = g.pos.mover
        text = f"{SIDE_NAME[mover]} to move"
        kind = self.players[mover]
        if kind != HUMAN and self.paused:
            text += " · engines paused (Space to resume)"
        elif kind in self.model_players and not self.model_players[kind].ready:
            text += f" · loading {self._player_label(kind)}…"
        elif kind != HUMAN:
            text += f" · {self._player_label(kind)} engine to move"
        else:
            text += " · your move"
        return text, ui.TEXT

    def _draw_panel(self) -> None:
        s, p, f = self.screen, self.panel_rect, self.fonts
        mouse = self.mouse
        for row in self.button_rows:
            for b in row:
                b.draw(s, f, mouse)
        for st in self.steppers:
            st.draw(s, f, mouse)

        y = self.controls_bottom + 2
        text, color = self._status_line()
        ui.draw_text(s, f.bold, text, color, (p.x + 2, y), max_w=p.w - 4)
        y += 21
        g = self.game
        if self.flash_msg and time.monotonic() < self.flash_until:
            sub, sub_color = self.flash_msg, self.flash_color
        else:
            limit = f" of {PLY_LIMIT}" if self.limits else ""
            sub, sub_color = f"{goat_count_text(g.pos).capitalize()} · ply {g.plies_played()}{limit}", ui.MUTED
        ui.draw_text(s, f.small, sub, sub_color, (p.x + 2, y), max_w=p.w - 4)
        y += 22

        y = self._draw_verdict(pygame.Rect(p.x, y, p.w, 74)) + GAP

        footer = self._footer_lines(p.w - 4) if p.h >= 640 else []
        footer_h = 16 * len(footer) + 4 if footer else 0
        min_movelist = 28 + 6 + 3 * MOVE_ROW_H
        space = p.bottom - footer_h - min_movelist - y - 2 * GAP
        fit = max(2, (space - 2 * 34) // LINE_H)
        threat_n = _clamp(min(self.lines_max, 4, fit // 3), 1, LINES_MAX)
        main_n = _clamp(min(self.lines_max, fit - threat_n), 1, LINES_MAX)

        self.line_rows = []
        mover = g.pos.mover
        for an, side, n in ((self.cur, mover, main_n), (self.threat, OTHER[mover], threat_n)):
            rows = n if an is None else max(1, min(n, len(an.lines)))
            rect = pygame.Rect(p.x, y, p.w, 28 + rows * LINE_H + 6)
            self._draw_section(rect, side, an, rows, to_move=side == mover)
            y += rect.h + GAP

        ml = pygame.Rect(p.x, y, p.w, max(0, p.bottom - footer_h - y))
        self._draw_movelist(ml)
        fy = p.bottom - footer_h + 4
        for i, line in enumerate(footer):
            ui.draw_text(s, f.small, line, ui.FAINT, (p.x + 2, fy + 16 * i), max_w=p.w - 4)

    def _verdict_lines(self) -> Tuple[str, str, str, tuple]:
        """(title, best-move summary, turn-limit note, border colour)."""
        g, an = self.game, self.cur
        if self.outcome is not None:
            o = self.outcome
            color = ui.TIGER_FILL if o.winner == TIGER else ui.GOAT_FILL if o.winner == GOAT else ui.MUTED
            title = f"{SIDE_NAME[o.winner]} won" if o.winner else "Draw"
            note = f"Game over: {o.reason}."
            if o.winner is None and an is not None and an.result != "draw":
                winner = an.pos.mover if an.result == "win" else OTHER[an.pos.mover]
                return title, note, (f"Without the limits, the {SIDE_NAME[winner].lower()} could still force a "
                                     f"win in {an.d} plies (Limits: None to play on)."), color
            return title, note, "", color
        if self.an is None:
            return "No perfect-play tables", self.an_error, "Human and random play still work.", ui.BAD
        if not self.analysis_on:
            return "Analysis is off", "Press E or the Analysis button to show it.", "", ui.BORDER
        if an is None:
            return "", "", "", ui.BORDER
        mover = an.pos.mover
        title = describe(an.result, an.d, mover)
        if an.result == "draw":
            color = ui.MUTED
        else:
            winner = mover if an.result == "win" else OTHER[mover]
            color = ui.TIGER_FILL if winner == TIGER else ui.GOAT_FILL
        n = len(an.lines)
        cnt = an.counts()
        L = self.labels if self.labels != "off" else "names"
        top = an.lines[0] if an.lines else None
        if top is None:
            summary = "No legal move."
        elif an.result == "win":
            k = cnt["win"]
            fastest = top.move.text(L)
            summary = (f"Only winning move: {fastest}" if k == 1 else
                       f"{k} of {n} moves win · fastest {fastest}")
            if cnt["draw"] or cnt["loss"]:
                summary += f" · {cnt['draw']} draw, {cnt['loss']} lose"
        elif an.result == "draw":
            summary = (f"Every move keeps the draw ({n})" if cnt["draw"] == n else
                       f"{cnt['draw']} of {n} moves keep the draw · {cnt['loss']} lose")
            if cnt["draw"] == 1:
                summary = f"Only drawing move: {top.move.text(L)} · {cnt['loss']} lose"
            elif top.traps:
                summary += f" · best trap {top.move.text(L)} ({top.traps}/{top.replies})"
        else:
            summary = f"Every move loses · longest defence {top.move.text(L)} ({top.d} plies)"
        note = ""
        if an.result != "draw" and self.limits:
            end = g.plies_played() + an.d
            note = (f"Beyond TnGEnv's limit: ply {end} > {PLY_LIMIT}, so there the game ends in a draw first."
                    if an.over_limit else f"Within TnGEnv's limit: over by ply {end} of {PLY_LIMIT}.")
        elif an.result == "draw":
            note = "Neither side can force a win; with best play the game goes on forever."
        return title, summary, note, color

    def _draw_verdict(self, rect: pygame.Rect) -> int:
        s, f = self.screen, self.fonts
        title, summary, note, color = self._verdict_lines()
        pygame.draw.rect(s, ui.PANEL, rect, border_radius=8)
        pygame.draw.rect(s, color, rect, width=2, border_radius=8)
        ui.draw_text(s, f.large, title, ui.TEXT, (rect.x + 12, rect.y + 7), max_w=rect.w - 24)
        ui.draw_text(s, f.regular, summary, ui.TEXT, (rect.x + 12, rect.y + 33), max_w=rect.w - 24)
        ui.draw_text(s, f.small, note, ui.MUTED, (rect.x + 12, rect.y + 53), max_w=rect.w - 24)
        return rect.bottom

    def _footer_lines(self, max_w: int) -> List[str]:
        font = self.fonts.small
        lines: List[str] = []
        cur = ""
        for item in FOOTER_ITEMS:
            trial = f"{cur} · {item}" if cur else item
            if cur and font.size(trial)[0] > max_w:
                lines.append(cur)
                cur = item
            else:
                cur = trial
        if cur:
            lines.append(cur)
        return lines

    def _draw_section(self, rect: pygame.Rect, side: str, an: Optional[Analysis], shown: int,
                      to_move: bool) -> None:
        s, f = self.screen, self.fonts
        pygame.draw.rect(s, ui.PANEL, rect, border_radius=8)
        border = ui.ACCENT if to_move and self.outcome is None else ui.BORDER
        pygame.draw.rect(s, border, rect, width=1, border_radius=8)
        icon = pygame.transform.smoothscale(self.sprites[side], (18, 18))
        s.blit(icon, icon.get_rect(center=(rect.x + 16, rect.y + 14)))
        title = f"{SIDE_NAME[side]} · {'best moves' if to_move else 'threats'}"
        tr = ui.draw_text(s, f.bold, title, ui.TEXT, (rect.x + 30, rect.y + 5))
        sub = "to move" if to_move else f"if it were the {SIDE_NAME[side].lower()}' move"
        ui.draw_text(s, f.small, sub, ui.ACCENT if to_move else ui.MUTED, (tr.right + 8, rect.y + 7),
                     max_w=rect.right - 150 - tr.right)
        if an is not None and an.lines:
            c = an.counts()
            ui.draw_text(s, f.small, f"{c['win']} win · {c['draw']} draw · {c['loss']} lose", ui.FAINT,
                         (rect.right - 10, rect.y + 7), anchor="topright")
        y = rect.y + 28
        x0, w = rect.x + 6, rect.w - 12
        msg = None
        if self.an is None:
            msg = "Perfect-play tables unavailable."
        elif not self.analysis_on:
            msg = "Analysis is off (press E or the Analysis button)."
        elif an is None:
            msg = "Game over." if self.outcome is not None else ""
        elif not an.lines:
            msg = f"The {SIDE_NAME[side].lower()} have no legal move."
        if msg is not None:
            ui.draw_text(s, f.small, msg, ui.FAINT, (x0 + 4, y + 4), max_w=w - 8)
            return
        L = self.labels if self.labels != "off" else "names"
        playable = to_move and self.outcome is None
        for ln in an.lines[:shown]:
            row = pygame.Rect(x0, y, w, LINE_H - 2)
            if row.collidepoint(self.mouse):
                pygame.draw.rect(s, ui.PANEL_HOVER, row, border_radius=5)
            self.line_rows.append((row, ln, playable))
            pill = pygame.Rect(row.x + 2, row.y + 2, 58, row.h - 4)
            color = RESULT_COLOR[ln.result]
            if ln.over_limit:
                pygame.draw.rect(s, ui.PANEL, pill, border_radius=4)
                pygame.draw.rect(s, color, pill, width=2, border_radius=4)
            else:
                pygame.draw.rect(s, color, pill, border_radius=4)
            ui.draw_text(s, f.mono_bold, pill_text(ln.result, ln.d), ui.TEXT, pill.center, anchor="center")
            mr = ui.draw_text(s, f.bold, ln.move.text(L), ui.TEXT, (pill.right + 8, row.centery), anchor="midleft")
            if ln.result == "draw" and ln.replies:
                mr = ui.draw_text(s, f.small, f"{ln.traps}/{ln.replies} replies lose",
                                  ui.WARN if ln.traps else ui.FAINT, (mr.right + 10, row.centery), anchor="midleft")
            if ln.pv:
                rest = " ".join(m.text(L) for m in ln.pv)
                ui.draw_text(s, f.small, rest, ui.MUTED, (mr.right + 10, row.centery), anchor="midleft",
                             max_w=row.right - mr.right - 16)
            y += LINE_H

    def _draw_movelist(self, rect: pygame.Rect) -> None:
        s, f, g = self.screen, self.fonts, self.game
        self.movelist_rect = rect
        self.move_cells = []
        pygame.draw.rect(s, ui.PANEL, rect, border_radius=8)
        pygame.draw.rect(s, ui.BORDER, rect, width=1, border_radius=8)
        ui.draw_text(s, f.bold, "Moves", ui.TEXT, (rect.x + 10, rect.y + 5))
        num_w = 52
        body = pygame.Rect(rect.x + 6, rect.y + 28, rect.w - 12, rect.h - 34)
        col_w = (body.w - num_w - 8) // 2
        sides = (g.first, OTHER[g.first])
        for col, side in enumerate(sides):
            ui.draw_text(s, f.small, SIDE_NAME[side], ui.FAINT, (body.x + num_w + col * col_w + 6, rect.y + 7))
        ui.draw_text(s, f.small, f"ply {g.ply}/{len(g.moves)}", ui.FAINT, (rect.right - 10, rect.y + 7),
                     anchor="topright")
        if body.h < MOVE_ROW_H:
            return
        offset = 0 if g.root.mover == g.first else 1
        n_rows = (len(g.moves) + offset + 1) // 2
        visible = max(1, body.h // MOVE_ROW_H)
        max_scroll = max(0, n_rows - visible)
        if self.movelist_follow:
            cur_row = (g.ply - 1 + offset) // 2 if g.ply > 0 else 0
            if cur_row < self.movelist_scroll:
                self.movelist_scroll = cur_row
            elif cur_row >= self.movelist_scroll + visible:
                self.movelist_scroll = cur_row - visible + 1
        self.movelist_scroll = _clamp(self.movelist_scroll, 0, max_scroll)
        if not g.moves:
            ui.draw_text(s, f.small, "No moves yet.", ui.FAINT, (body.x + 4, body.y + 3))
            return
        L = self.labels if self.labels != "off" else "names"
        prev_clip = s.get_clip()
        s.set_clip(body)
        for row in range(self.movelist_scroll, min(n_rows, self.movelist_scroll + visible)):
            ry = body.y + (row - self.movelist_scroll) * MOVE_ROW_H
            if row % 2:
                pygame.draw.rect(s, ui.PANEL_ALT, (body.x, ry, body.w, MOVE_ROW_H))
            ui.draw_text(s, f.small, f"{row + 1}.", ui.FAINT, (body.x + 6, ry + 3))
            for col in (0, 1):
                idx = row * 2 + col - offset
                cell = pygame.Rect(body.x + num_w + col * col_w, ry, col_w, MOVE_ROW_H)
                if idx < 0:
                    ui.draw_text(s, f.regular, "…", ui.FAINT, (cell.x + 6, ry + 2))
                    continue
                if idx >= len(g.moves):
                    continue
                ply = idx + 1
                if ply == g.ply:
                    pygame.draw.rect(s, ui.ACCENT_DARK, cell.inflate(-2, -2), border_radius=4)
                elif cell.collidepoint(self.mouse):
                    pygame.draw.rect(s, ui.PANEL_HOVER, cell.inflate(-2, -2), border_radius=4)
                color = ui.TEXT if ply <= g.ply else ui.MUTED
                r = ui.draw_text(s, f.regular, g.moves[idx].text(L), color, (cell.x + 6, ry + 2),
                                 max_w=cell.w - 30)
                mark = self.annotations.get(ply, "")
                if mark:
                    ui.draw_text(s, f.bold, mark, ui.BAD if mark == "??" else ui.WARN, (r.right + 3, ry + 2))
                self.move_cells.append((cell, ply))
        s.set_clip(prev_clip)
        if max_scroll:
            track = pygame.Rect(body.right - 4, body.y, 3, body.h)
            thumb_h = max(16, body.h * visible // n_rows)
            thumb_y = body.y + (body.h - thumb_h) * self.movelist_scroll // max_scroll
            pygame.draw.rect(s, ui.PANEL_ALT, track)
            pygame.draw.rect(s, ui.FAINT, (track.x, thumb_y, 3, thumb_h))

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def step(self, dt: float = 1 / 60) -> bool:
        """Handle pending events, update and draw one frame. False when the window should close."""
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                self.screen = pygame.display.get_surface()
                self._layout(*self.screen.get_size())
            elif ev.type == pygame.MOUSEMOTION:
                self.mouse = ev.pos
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                self.mouse = ev.pos
                if ev.button == 1:
                    self._on_left_down(ev.pos)
                elif ev.button == 3:
                    self.selected = None
                    self.dragging = False
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.mouse = ev.pos
                self._board_release(ev.pos)
            elif ev.type == pygame.MOUSEWHEEL:
                self._on_wheel(ev.y)
            elif ev.type == pygame.KEYDOWN:
                if not self._on_key(ev):
                    return False
            elif ev.type == pygame.WINDOWFOCUSLOST:
                self.dragging = False
        self._update(dt)
        self._draw()
        pygame.display.flip()
        return True

    def run(self) -> None:
        try:
            while self.step(self.clock.tick(60) / 1000.0):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="tngviz",
        description="Tigers & Goats board with perfect-play analysis for both sides, forced wins and engine play.")
    found = discover()
    for side, flag in ((GOAT, "goats"), (TIGER, "tigers")):
        names = ", ".join(i.path.stem for i in found[side]) or "none found"
        p.add_argument(f"--{flag}", default=HUMAN if side == GOAT else PERFECT,
                       help=f"who plays the {flag}: human, perfect, random, or a trained model from "
                            f"stable_models/models/{flag.title()} by file name or a unique part of it "
                            f"({names}) (default: {'human' if side == GOAT else 'perfect'})")
    p.add_argument("--first", choices=[GOAT, TIGER], default=GOAT, help="who moves first (default: goat)")
    p.add_argument("--limits", choices=["tngenv", "none"], default="none",
                   help="tngenv: 200-ply turn limit and the 4th-repeat rule, as in TnGEnv; none: the solved "
                        "game without limits (default: none)")
    p.add_argument("--position", help="start from a position, e.g. 'tigers=4,12,13 goats=3,5,6,22 captured=0 "
                                      "mover=goat'")
    p.add_argument("--game", help="load a game saved with Copy game (a text file)")
    p.add_argument("--tables", default=str(DEFAULT_TABLES), help="folder with the solved tables "
                                                                 "(default: perfect_play/tables)")
    p.add_argument("--delay", type=float, default=0.5, help="seconds before each engine move (default: 0.5)")
    p.add_argument("--lines", type=int, default=8, help=f"moves listed for the side to move, 1-{LINES_MAX} "
                                                        "(default: 8)")
    p.add_argument("--pv", type=int, default=8, help=f"plies of perfect continuation per line, 0-{PV_MAX} "
                                                     "(default: 8)")
    p.add_argument("--labels", choices=["names", "numbers", "off"], default="names",
                   help="square labels: names like c2, numbers 0-22, or off (default: names)")
    p.add_argument("--no-analysis", action="store_true", help="start with the analysis hidden")
    p.add_argument("--size", help="window size as WxH, e.g. 1400x900")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    App(parse_args(argv)).run()
