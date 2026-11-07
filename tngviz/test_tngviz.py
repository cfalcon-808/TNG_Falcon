"""Tests for the visualizer: moves against TnGEnv, end-of-game rules, position text, analysis, and the app
driven headless. Run from the repository root:

    PYTHONPATH=. tngviz/.venv/bin/python -m tngviz.test_tngviz      (or: python -m pytest tngviz/test_tngviz.py)

The analysis and app tests need the solved tables in perfect_play/tables and are skipped without them.
"""

from __future__ import annotations

import os
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402

from tngviz.game import (DEFAULT_TABLES, GOAT, MAX_REPEATS, PLY_LIMIT, TIGER, Analyzer, Game, Position,  # noqa: E402
                         env_tng, legal_moves, parse_position, plies_before, position_to_text, start_position)

HAVE_TABLES = (DEFAULT_TABLES / "metadata.json").exists() and (DEFAULT_TABLES / "pl_n0_G.u8").exists()


def _random_position(rng: random.Random) -> Position:
    while True:
        placing = rng.random() < 0.5
        captured = rng.randint(0, 5)
        on_board = rng.randint(0, 15 - captured) if placing else 15 - captured
        if placing and on_board + captured >= 15:
            continue
        sq = rng.sample(range(23), 3 + on_board)
        tm = sum(1 << p for p in sq[:3])
        gm = sum(1 << p for p in sq[3:])
        return Position(tm, gm, captured, on_board + captured if placing else 15, rng.choice((GOAT, TIGER)))


def _env_at(pos: Position):
    """A TnGEnv for the side to move, set to `pos`."""
    env = env_tng.TnGEnv(learner_role=env_tng.GOAT_LEARNER if pos.mover == GOAT else env_tng.TIGER_LEARNER)
    env.reset(seed=0)
    board = np.zeros(23, dtype=np.int8)
    for p in range(23):
        board[p] = 2 if (pos.tm >> p) & 1 else 1 if (pos.gm >> p) & 1 else 0
    env.board = board
    env.eaten = pos.captured
    env.goats_placed = pos.placed
    env.phase = 0 if pos.placing else 1
    env._update_valid_moves()
    return env


def test_moves_match_tngenv():
    """Every move the visualizer offers is one TnGEnv allows, and the board after it is TnGEnv's."""
    rng = random.Random(11)
    checked = 0
    for _ in range(1500):
        pos = _random_position(rng)
        env = _env_at(pos)
        mask = env.get_action_mask() if hasattr(env, "get_action_mask") else None
        moves = legal_moves(pos)
        if mask is not None:
            assert sorted(m.action for m in moves) == sorted(int(a) for a in np.flatnonzero(mask)), pos
        for mv in moves:
            if pos.mover == TIGER:
                assert mv.frm is not None
                if mv.jumped is not None:
                    assert env._find_jumped_goat(mv.frm, mv.to) == mv.jumped, (pos, mv)
            checked += 1
    assert checked > 10_000


def test_end_rules():
    # 6th capture: tigers on 0, 3, 4 ... build a position with 5 captured and a capture available
    pos = parse_position("tigers=8,20,22 goats=2,1,6,7,13,18 captured=5 mover=tiger")
    caps = [m for m in legal_moves(pos) if m.jumped is not None]
    assert caps and caps[0].nxt is None
    g = Game(pos, GOAT)
    g.push(caps[0])
    o = g.outcome()
    assert o is not None and o.winner == TIGER

    # trapped tigers lose
    trapped = parse_position("tigers=0,1,6 goats=2,3,4,5,7,8,9,10,11,12,13,18 captured=0 mover=tiger")
    assert legal_moves(trapped) == []
    o = Game(trapped, GOAT).outcome()
    assert o is not None and o.winner == GOAT

    # turn limit and repeat rule (goats first), only with limits on
    g = Game(start_position(GOAT), GOAT)
    rng = random.Random(5)
    while g.outcome(limits=False) is None and g.ply < PLY_LIMIT + 10:
        g.push(rng.choice(legal_moves(g.pos)))
        if g.outcome(limits=True) is not None:
            break
    o = g.outcome(limits=True)
    assert o is not None
    if o.winner is None:
        reason = o.reason
        assert ("turn limit" in reason and g.plies_played() == PLY_LIMIT) or (
            "repeat" in reason and g.repeats() >= MAX_REPEATS), reason


def test_position_text_round_trip():
    rng = random.Random(3)
    for _ in range(500):
        pos = _random_position(rng)
        assert parse_position(position_to_text(pos)) == pos
    assert parse_position("Tigers: b0 c1 d1  goats: a1, b2  mover: t") == Position(25, 258, 0, 2, TIGER)
    for bad in ("tigers=0,3 goats=1", "tigers=0,3,4 goats=0", "tigers=0,3,4 captured=6", "goats=1,2"):
        try:
            parse_position(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


def test_plies_before():
    assert plies_before(start_position(GOAT), GOAT) == 0
    assert plies_before(start_position(TIGER), TIGER) == 0
    pos = parse_position("tigers=4,12,13 goats=3,5,6,22 captured=0 mover=goat")
    assert plies_before(pos, GOAT) == 8 and plies_before(pos, TIGER) == 9


def test_analysis():
    if not HAVE_TABLES:
        print("  skipped: no tables")
        return
    an = Analyzer()
    a = an.analyse(start_position(GOAT), 0, True)
    assert a.result == "draw" and a.lines and all(ln.result == "draw" for ln in a.lines)
    pos = parse_position("tigers=4,12,13 goats=3,5,6,22 captured=0 mover=goat")
    a = an.analyse(pos, plies_before(pos, GOAT), True)
    assert (a.result, a.d, a.over_limit) == ("win", 215, True)
    assert [ln.move.text() for ln in a.best] == ["@b1"]
    # every line's result agrees with the oracle's move values
    rng = random.Random(9)
    for _ in range(300):
        pos = _random_position(rng)
        rows = {act: (res, d) for act, res, d in an.pp.move_values(pos)}
        for ln in an.lines(pos, 0, False):
            assert rows[ln.move.action] == (ln.result, ln.d), (pos, ln.move)
    # perfect play from a won position wins in exactly the predicted number of plies
    g = Game(pos := parse_position("tigers=8,10,17 goats=2,3,4,5,7,9,11,13,14,15,18,21 captured=3 mover=tiger"))
    want = an.value(pos)
    while g.outcome(limits=False) is None:
        g.push(an.choose(an.analyse(g.pos, g.plies_played(), False), g.seen(), rng))
    assert want == ("win", g.ply) and g.outcome(limits=False).winner == TIGER


def test_app_headless():
    if not HAVE_TABLES:
        print("  skipped: no tables")
        return
    import pygame
    from tngviz.app import App, parse_args

    assert parse_args([]).limits == "none"
    app = App(parse_args(["--size", "1200x780", "--goats", "human", "--tigers", "perfect", "--delay", "0",
                          "--limits", "tngenv"]))
    try:
        for _ in range(3):
            assert app.step()

        def click(sq):
            x, y = (int(v) for v in app.node_xy[sq])
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(x, y), button=1))
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(x, y), button=1))
            for _ in range(15):          # animations are timed, so let them finish
                app.step()
                time.sleep(0.02)

        click(9)                                     # goats place on c2, the perfect tigers reply
        assert app.game.moves[0].text() == "@c2" and len(app.game.moves) == 2
        assert app.cur is not None and app.cur.pos == app.game.pos
        text = app.game_text()
        app._new_game()
        assert app._load_text(text) and [m.text() for m in app.game.moves][:1] == ["@c2"]
        assert app._load_text("tigers=4,12,13 goats=3,5,6,22 captured=0 mover=goat")
        app.step()
        assert app.cur.d == 215 and app.bar_label == "215"
        for key in (pygame.K_r, pygame.K_a, pygame.K_v, pygame.K_l, pygame.K_e, pygame.K_e, pygame.K_PLUS):
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode=""))
            app.step()
        assert not app.limits and not app.cur.over_limit
        app.players = {GOAT: "perfect", TIGER: "random"}
        app._new_game()
        for _ in range(400):
            app.step()
            if app.outcome is not None:
                break
        assert app.outcome is not None and app.outcome.winner != TIGER   # perfect goats never lose
    finally:
        pygame.quit()


def test_model_observation_matches_tngenv():
    """The observation a model sees equals TnGEnv.get_state() on the same board."""
    from tngviz.models import observation
    rng = random.Random(17)
    for _ in range(500):
        pos = _random_position(rng)
        assert np.array_equal(observation(pos), _env_at(pos).get_state()), pos


def test_models_play():
    """Every trained model loads, fits TnGEnv's spaces, and plays a full game through the app's dropdown."""
    import importlib.util
    from tngviz.models import ModelPlayer, discover
    found = discover()
    infos = found[GOAT] + found[TIGER]
    assert infos, "no models in stable_models/models"
    if importlib.util.find_spec("sb3_contrib") is None:
        print("  skipped: sb3-contrib not installed")
        return
    rng = random.Random(23)
    for info in infos:
        mp = ModelPlayer(info)
        mp._load()
        assert mp.ready, mp.error
        assert mp.model.observation_space.shape == (25,) and mp.model.action_space.n == 115, info.path.name
        played = 0
        for _ in range(200):
            pos = _random_position(rng)
            if pos.mover != info.side or not legal_moves(pos):
                continue
            mv = mp.choose(pos, legal_moves(pos))
            assert mv is not None and mv in legal_moves(pos), (info.path.name, pos)
            played += 1
        assert played > 20

    import pygame
    from tngviz.app import App, parse_args
    goat, tiger = found[GOAT][0], found[TIGER][0]
    app = App(parse_args(["--size", "1200x780", "--goats", "human", "--tigers", "random", "--delay", "0"]))
    try:
        dd = app.dropdowns[GOAT]
        app.step()
        x, y = app.btn_goats.rect.center
        app._on_left_down((x, y))
        assert dd.open
        rows = dd._rows(app.screen.get_height())
        target = [r for r, (value, _, _) in rows if value == goat.key][0]
        app._on_left_down(target.center)                      # pick the first goat model from the list
        assert not dd.open and app.players[GOAT] == goat.key
        app._set_player(TIGER, app._resolve_player(TIGER, tiger.path.stem))
        deadline = time.monotonic() + 120
        while app.outcome is None and len(app.game.moves) < 60 and time.monotonic() < deadline:
            app.step()
        sides = {m.side for m in app.game.moves}
        assert sides == {GOAT, TIGER}, "both models moved"
    finally:
        pygame.quit()


def main() -> int:
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # report and keep going
                failed += 1
                print(f"FAIL {name}: {exc!r}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
