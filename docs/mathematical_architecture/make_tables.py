"""Generate the data tables and figures of the Mathematical Architecture PDF from the result files.

usage: python docs/mathematical_architecture/make_tables.py      (matplotlib needed for the figures)
Reads perfect_play/tables/{metadata,reachable,model_comparison}.json; writes generated/*.tex and figures/*.pdf.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = HERE.parent.parent / "perfect_play" / "tables"
GEN = HERE / "generated"
FIG = HERE / "figures"
GEN.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
meta = json.loads((T / "metadata.json").read_text())
reach = json.loads((T / "reachable.json").read_text())
comp = json.loads((T / "model_comparison.json").read_text())

SHORT = {"Perfect": "Perfect", "Perfect (plain)": "Perfect (plain)", "Random": "Random", "Greedy": "Greedy",
         "Smart": "Smart", "Tiger vs NormalGoat": "Tiger-N", "Tiger vs SmartGoat": "Tiger-S",
         "RobustTiger": "RobustTiger", "NormalGoat": "NormalGoat", "SmartGoat": "SmartGoat",
         "RobustGoat": "RobustGoat", "PPO+nll 0.5 (10M)": "PPO+nll", "ReLU Maskable (10M)": "ReLU-Mask"}


def n(x):
    return f"{x:,}".replace(",", "{,}")


def write(name, lines):
    (GEN / name).write_text("\n".join(lines) + "\n")


def res(p):
    r = p["results"]
    g, t = r.get("goat_win", 0), r.get("tiger_win", 0)
    return g, t, p["games"] - g - t


# ── layers of the solved game ────────────────────────────────────────────────────────────────────────
rows = []
for k in range(10, 16):
    m = meta["movement"][str(k)]
    rows.append(f"{k} & {m['captured']} & {n(m['boards'])} & {n(m['G']['win'])} & {n(m['G']['loss'])} & {n(m['G']['draw'])} & "
                f"{m['G']['max_dtm']} & {n(m['T']['win'])} & {n(m['T']['loss'])} & {n(m['T']['draw'])} & {m['T']['max_dtm']} \\\\")
write("tab_movement_layers.tex", rows)
rows = []
pl = meta["placement_dtm"]["layers"]
for i in range(15):
    g, t = pl[str(i)]["G"], pl[str(i)]["T"]
    tot = g["win"] + g["loss"] + g["draw"]
    rows.append(f"{i} & {n(tot)} & {n(g['win'])} & {n(g['loss'])} & {n(g['draw'])} & {g['max_dtm']} & "
                f"{n(t['win'])} & {n(t['loss'])} & {n(t['draw'])} & {t['max_dtm']} \\\\")
write("tab_placement_layers.tex", rows)

# ── reachable positions ─────────────────────────────────────────────────────────────────────────────
rows = []
for i in range(15):
    cells = [str(i), n(sum(pl[str(i)]["G"][x] for x in ("win", "loss", "draw")))]
    for first in ("goat", "tiger"):
        c = reach[f"{first}_first"]["counts"]
        cells += [n(c[f"pl_n{i}_G"]["reachable"]), n(c[f"pl_n{i}_T"]["reachable"])]
    rows.append(" & ".join(cells) + " \\\\")
rows.append("\\midrule")
for k in range(15, 9, -1):
    cells = [f"mv, $k={k}$", n(meta["movement"][str(k)]["boards"])]
    for first in ("goat", "tiger"):
        c = reach[f"{first}_first"]["counts"]
        cells += [n(c[f"mv_k{k}_G"]["reachable"]), n(c[f"mv_k{k}_T"]["reachable"])]
    rows.append(" & ".join(cells) + " \\\\")
write("tab_reachable.tex", rows)

# ── round robin ─────────────────────────────────────────────────────────────────────────────────────
tigers, goats = comp["tiger_players"], comp["goat_players"]
by = {(p["goat"], p["tiger"]): p for p in comp["pairings"]}
rows = []
for g in goats:
    cells = [SHORT[g]]
    for t in tigers:
        p = by[(g, t)]
        gw, tw, d = res(p)
        if p["games"] == 1:
            cells.append("\\textit{%s}$^\\dagger$" % ("G" if gw else "T" if tw else "D"))
        else:
            cells.append(f"{100 * gw / p['games']:.0f}\\,/\\,{100 * tw / p['games']:.0f}")
    rows.append(" & ".join(cells) + " \\\\")
write("tab_roundrobin.tex", rows)
write("tab_roundrobin_head.tex", [" & ".join(["Goats $\\downarrow$ \\ Tigers $\\rightarrow$"] + [SHORT[t] for t in tigers]) + " \\\\"])

# ── move quality per player (all its round-robin games) ──────────────────────────────────────────────
agg = {}
for p in comp["pairings"]:
    for side, name in (("goat", p["goat"]), ("tiger", p["tiger"])):
        s = p["sides"][side]
        a = agg.setdefault((side, name), dict(dec=0, win=0, draw=0, loss=0, WD=0, WL=0, DL=0, had=0, conv=0, games=0))
        a["dec"] += s["decisions"]
        a["games"] += p["games"]
        for k in ("win", "draw", "loss"):
            a[k] += s["by_value"].get(k, 0)
        for k, kk in (("W>D", "WD"), ("W>L", "WL"), ("D>L", "DL")):
            a[kk] += s["blunders"].get(k, 0)
        a["had"] += s["had_win_games"]
        a["conv"] += s["converted_games"]
rows = []
for side, names in (("goat", goats), ("tiger", tigers)):
    for name in names:
        a = agg[(side, name)]
        wd = 100 * (a["WD"] + a["WL"]) / a["win"] if a["win"] else None
        dl = 100 * a["DL"] / a["draw"] if a["draw"] else None
        rows.append(" & ".join([
            ("Goats" if side == "goat" else "Tigers") if name == names[0] else "", SHORT[name], n(a["dec"]),
            f"{100 * a['win'] / a['dec']:.1f}", f"{100 * a['draw'] / a['dec']:.1f}", f"{100 * a['loss'] / a['dec']:.1f}",
            "--" if wd is None else f"{wd:.1f}", "--" if dl is None else f"{dl:.1f}",
            f"{100 * (a['WD'] + a['WL'] + a['DL']) / a['dec']:.2f}",
            f"{a['conv']}/{a['had']}" + (f" ({100 * a['conv'] / a['had']:.0f}\\%)" if a["had"] else "")]) + " \\\\")
    if side == "goat":
        rows.append("\\midrule")
write("tab_quality.tex", rows)

# ── models against the perfect player ────────────────────────────────────────────────────────────────
rows = []
for g in goats[2:]:
    for t in ("Perfect", "Perfect (plain)"):
        p = by[(g, t)]
        gw, tw, d = res(p)
        s = p["sides"]["goat"]
        fb = s["mean_first_blunder_ply"]
        rows.append(" & ".join([SHORT[g] if t == "Perfect" else "", "trap-seeking" if t == "Perfect" else "plain",
                                f"{100 * gw / p['games']:.0f}", f"{100 * d / p['games']:.1f}", f"{100 * tw / p['games']:.1f}",
                                f"{p['mean_plies']:.0f}", f"{s['blunders'].get('D>L', 0) / p['games']:.2f}",
                                "--" if fb is None else f"{fb:.0f}"]) + " \\\\")
write("tab_goats_vs_perfect.tex", rows)
rows = []
for t in tigers[2:]:
    for g in ("Perfect", "Perfect (plain)"):
        p = by[(g, t)]
        gw, tw, d = res(p)
        s = p["sides"]["tiger"]
        fb = s["mean_first_blunder_ply"]
        rows.append(" & ".join([SHORT[t] if g == "Perfect" else "", "trap-seeking" if g == "Perfect" else "plain",
                                f"{100 * tw / p['games']:.0f}", f"{100 * d / p['games']:.1f}", f"{100 * gw / p['games']:.1f}",
                                f"{p['mean_plies']:.0f}", f"{s['blunders'].get('D>L', 0) / p['games']:.2f}",
                                "--" if fb is None else f"{fb:.0f}"]) + " \\\\")
write("tab_tigers_vs_perfect.tex", rows)

# ── models against models from balanced random openings ──────────────────────────────────────────────
rows = []
last = None
for p in comp["models_from_random_openings"]:
    gw, tw, d = res(p)
    bl = [100 * sum(p["sides"][s]["blunders"].values()) / max(p["sides"][s]["decisions"], 1) for s in ("goat", "tiger")]
    rows.append(" & ".join([SHORT[p["goat"]] if p["goat"] != last else "", SHORT[p["tiger"]],
                            f"{100 * gw / p['games']:.1f}", f"{100 * d / p['games']:.1f}", f"{100 * tw / p['games']:.1f}",
                            f"{p['mean_plies']:.0f}", f"{bl[0]:.1f}", f"{bl[1]:.1f}"]) + " \\\\")
    last = p["goat"]
write("tab_openings.tex", rows)

# ── effect of the capture fix ───────────────────────────────────────────────────────────────────────
STUDY = {("NormalGoat", "Greedy"): 0.796, ("NormalGoat", "Smart"): 0.016, ("SmartGoat", "Greedy"): 0.067,
         ("SmartGoat", "Smart"): 0.944, ("RobustGoat", "Greedy"): 0.798, ("RobustGoat", "Smart"): 0.857,
         ("PPO+nll 0.5 (10M)", "Greedy"): 0.754, ("PPO+nll 0.5 (10M)", "Smart"): 0.947,
         ("ReLU Maskable (10M)", "Greedy"): 0.854, ("ReLU Maskable (10M)", "Smart"): 0.856}
fix = {(p["goat"], p["tiger"], p["legacy_capture"]): p for p in comp["capture_fix"]}
rows = []
for g in goats[3:]:
    cells = [SHORT[g]]
    for t in ("Greedy", "Smart"):
        new, old = fix[(g, t, False)], fix[(g, t, True)]
        wn, wo = res(new)[0] / new["games"], res(old)[0] / old["games"]
        cells += [f"{STUDY[(g, t)]:.3f}", f"{wo:.3f}", f"{wn:.3f}", f"{100 * (wn - wo):+.1f}"]
    rows.append(" & ".join(cells) + " \\\\")
write("tab_capturefix.tex", rows)

# ── the longest-win games, copyable ──────────────────────────────────────────────────────────────────
GAMES = [("goats_first_goats_win_213.txt", "Goats first: the goats' longest forced win (213 plies from ply 10)"),
         ("tigers_first_goats_win_214.txt", "Tigers first: the goats' longest forced win (214 plies from ply 14)"),
         ("goats_first_goats_win_within_200_plies_191.txt", "Goats first: longest win inside the 200-ply limit (191 plies from ply 6)"),
         ("tigers_first_goats_win_within_200_plies_193.txt", "Tigers first: longest win inside the 200-ply limit (193 plies from ply 7)"),
         ("goats_first_tigers_win_67.txt", "Goats first: the tigers' longest forced win (67 plies from ply 19)"),
         ("tigers_first_tigers_win_67.txt", "Tigers first: the tigers' longest forced win (67 plies from ply 20)")]
out = []
for fname, title in GAMES:
    text = [ln for ln in (T.parent / "games" / fname).read_text().splitlines() if ln and not ln.startswith("#")]
    out += [f"\\paragraph{{{title}.}} File \\code{{perfect\\_play/games/{fname.replace('_', chr(92) * 2 + '_')}}}.".replace("\\\\", "\\"),
            "\\begin{gamelisting}"] + [ln + "\\par" for ln in text] + ["\\end{gamelisting}", ""]
write("games.tex", [x.replace("\\\\", "\\") for x in out])

# ── figures ─────────────────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib not available: tables written, figures skipped")
    raise SystemExit

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
C_G, C_D, C_T = "#4c78a8", "#b8b8b8", "#e4572e"

# outcomes of every goat player against four reference tigers
opps = ["Perfect", "Random", "Greedy", "Smart"]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
for ax, t in zip(axes, opps):
    names = goats[::-1]
    g = np.array([100 * res(by[(x, t)])[0] / by[(x, t)]["games"] for x in names])
    tw = np.array([100 * res(by[(x, t)])[1] / by[(x, t)]["games"] for x in names])
    d = 100 - g - tw
    y = np.arange(len(names))
    ax.barh(y, g, color=C_G, label="goats win")
    ax.barh(y, d, left=g, color=C_D, label="draw")
    ax.barh(y, tw, left=g + d, color=C_T, label="tigers win")
    ax.set_title(f"vs {t} tiger", fontsize=9)
    ax.set_yticks(y, [SHORT[x] for x in names])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of games")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "outcomes_goat_players.pdf", bbox_inches="tight")
plt.close(fig)

# blunders per 100 decisions
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, side, names in ((axes[0], "goat", goats), (axes[1], "tiger", tigers)):
    names = names[::-1]
    wd = [100 * (agg[(side, x)]["WD"] + agg[(side, x)]["WL"]) / agg[(side, x)]["dec"] for x in names]
    dl = [100 * agg[(side, x)]["DL"] / agg[(side, x)]["dec"] for x in names]
    y = np.arange(len(names))
    ax.barh(y, wd, color="#f2a541", label="threw away a win")
    ax.barh(y, dl, left=wd, color=C_T, label="turned a draw into a loss")
    ax.set_yticks(y, [SHORT[x] for x in names])
    ax.set_xlabel("blunders per 100 decisions")
    ax.set_title("goat players" if side == "goat" else "tiger players", fontsize=9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "blunders.pdf", bbox_inches="tight")
plt.close(fig)

# plies-to-the-end histogram over all placement positions with a forced result
hist = {}
for layer in pl.values():
    for side in ("G", "T"):
        for d, c in layer[side]["dtm_histogram"].items():
            hist[int(d)] = hist.get(int(d), 0) + c
xs = np.array(sorted(hist))
fig, ax = plt.subplots(figsize=(7.2, 2.4))
ax.bar(xs, [hist[x] for x in xs], width=1.0, color=C_G)
ax.set_yscale("log")
ax.set_xlabel("plies to the end with perfect play")
ax.set_ylabel("placement positions")
ax.axvline(200, color="k", lw=0.6, ls="--")
ax.text(198, max(hist.values()) / 3, "200", ha="right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "dtm_histogram.pdf", bbox_inches="tight")
plt.close(fig)
print("tables and figures written")
