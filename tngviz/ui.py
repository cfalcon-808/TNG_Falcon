"""Small pygame widgets and drawing helpers (adapted from the chessviz visualizer)."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple

import pygame

Color = Tuple[int, ...]

# ---- theme -------------------------------------------------------------
BG = (28, 29, 33)
PANEL = (38, 40, 45)
PANEL_ALT = (46, 48, 54)
PANEL_HOVER = (58, 61, 69)
BORDER = (70, 73, 82)
TEXT = (232, 233, 236)
MUTED = (150, 153, 162)
FAINT = (105, 108, 117)
ACCENT = (64, 138, 214)
ACCENT_DARK = (44, 98, 156)
GOOD = (98, 166, 70)
WARN = (226, 168, 60)
BAD = (214, 82, 76)

BOARD_BG = (236, 214, 176)       # the board, a light wood tone
BOARD_EDGE = (150, 108, 72)
BOARD_LINE = (96, 66, 42)
GOAT_FILL = (244, 242, 234)
GOAT_DARK = (92, 92, 96)
TIGER_FILL = (232, 128, 44)
TIGER_DARK = (58, 34, 16)
WIN_C = (52, 150, 72)            # result pills: the mover wins / draws / loses
DRAW_C = (128, 131, 140)
LOSS_C = (196, 70, 64)


class Fonts:
    """Font set; falls back to pygame's bundled font when DejaVu is absent."""

    def __init__(self) -> None:
        sans = pygame.font.match_font("dejavusans")
        mono = pygame.font.match_font("dejavusansmono")
        bold = pygame.font.match_font("dejavusans", bold=True)
        self.regular = pygame.font.Font(sans, 15)
        self.small = pygame.font.Font(sans, 13)
        self.tiny = pygame.font.Font(bold or sans, 11)
        self.bold = pygame.font.Font(bold or sans, 15)
        self.large = pygame.font.Font(bold or sans, 19)
        self.mono = pygame.font.Font(mono or sans, 14)
        self.mono_bold = pygame.font.Font(pygame.font.match_font("dejavusansmono", bold=True) or mono or sans, 14)
        self.coord = pygame.font.Font(bold or sans, 12)
        self.bar = pygame.font.Font(bold or sans, 11)
        self._sans_path = sans
        self._bold_path = bold

    def coord_font(self, square: int) -> pygame.font.Font:
        size = max(9, min(16, square // 6))
        return pygame.font.Font(self._bold_path or self._sans_path, size)


def fit_text(font: pygame.font.Font, text: str, max_w: int) -> str:
    """Trim ``text`` with an ellipsis so it renders within ``max_w`` pixels."""
    if max_w <= 0:
        return ""
    if font.size(text)[0] <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.size(text[:mid].rstrip() + "…")[0] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…" if lo else ""


def draw_text(surf: pygame.Surface, font: pygame.font.Font, text: str, color: Color,
              pos: Tuple[int, int], *, anchor: str = "topleft", max_w: Optional[int] = None) -> pygame.Rect:
    if max_w is not None:
        text = fit_text(font, text, max_w)
    img = font.render(text, True, color)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def wrap_text(font: pygame.font.Font, text: str, max_w: int) -> list:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.size(trial)[0] <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class Button:
    def __init__(self, label: Callable[[], str], on_click: Callable[[], None], *,
                 enabled: Callable[[], bool] = lambda: True,
                 active: Callable[[], bool] = lambda: False,
                 tooltip: str = "") -> None:
        self.label = label
        self.on_click = on_click
        self.enabled = enabled
        self.active = active
        self.tooltip = tooltip
        self.rect = pygame.Rect(0, 0, 0, 0)

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos) and self.enabled()

    def draw(self, surf: pygame.Surface, fonts: Fonts, mouse: Tuple[int, int]) -> None:
        enabled = self.enabled()
        hover = enabled and self.rect.collidepoint(mouse)
        if self.active():
            fill = ACCENT if hover else ACCENT_DARK
        else:
            fill = PANEL_HOVER if hover else PANEL_ALT
        pygame.draw.rect(surf, fill, self.rect, border_radius=6)
        pygame.draw.rect(surf, BORDER, self.rect, width=1, border_radius=6)
        color = TEXT if enabled else FAINT
        draw_text(surf, fonts.regular, self.label(), color, self.rect.center,
                  anchor="center", max_w=self.rect.w - 10)


class Stepper:
    """'Title: value' with − and + buttons."""

    def __init__(self, title: str, value: Callable[[], str], dec: Callable[[], None],
                 inc: Callable[[], None], *, enabled: Callable[[], bool] = lambda: True,
                 tooltip: str = "") -> None:
        self.title = title
        self.value = value
        self.enabled = enabled
        self.tooltip = tooltip
        self.minus = Button(lambda: "−", dec, enabled=enabled)
        self.plus = Button(lambda: "+", inc, enabled=enabled)
        self.rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, rect: pygame.Rect) -> None:
        self.rect = rect
        bw = min(28, rect.h)
        self.minus.rect = pygame.Rect(rect.x, rect.y, bw, rect.h)
        self.plus.rect = pygame.Rect(rect.right - bw, rect.y, bw, rect.h)

    def buttons(self) -> Sequence[Button]:
        return (self.minus, self.plus)

    def draw(self, surf: pygame.Surface, fonts: Fonts, mouse: Tuple[int, int]) -> None:
        mid = pygame.Rect(self.minus.rect.right + 3, self.rect.y,
                          self.plus.rect.left - self.minus.rect.right - 6, self.rect.h)
        pygame.draw.rect(surf, PANEL, mid, border_radius=6)
        pygame.draw.rect(surf, BORDER, mid, width=1, border_radius=6)
        color = TEXT if self.enabled() else FAINT
        # Always show the value; shorten or drop the title when space is tight.
        value = self.value()
        title = fit_text(fonts.small, self.title, mid.w - 10 - fonts.small.size(value)[0])
        text = f"{title} {value}" if title and not title.endswith("…") else value
        draw_text(surf, fonts.small, text, color, mid.center, anchor="center", max_w=mid.w - 4)
        self.minus.draw(surf, fonts, mouse)
        self.plus.draw(surf, fonts, mouse)


def draw_arrow(surf: pygame.Surface, color: Color, start: Tuple[float, float],
               end: Tuple[float, float], width: float, head_len: float, head_w: float) -> None:
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head_len = min(head_len, length * 0.6)
    bx, by = ex - ux * head_len, ey - uy * head_len
    hw, hh = width / 2, head_w / 2
    poly = [
        (sx + px * hw, sy + py * hw), (bx + px * hw, by + py * hw),
        (bx + px * hh, by + py * hh), (ex, ey),
        (bx - px * hh, by - py * hh), (bx - px * hw, by - py * hw),
        (sx - px * hw, sy - py * hw),
    ]
    pygame.draw.polygon(surf, color, poly)
