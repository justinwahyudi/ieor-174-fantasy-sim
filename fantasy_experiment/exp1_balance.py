"""Experiment 1: where do the two picks disagree? Breakdown by matchup balance.

Run (after run_exp1_leagues; reads its saved output, no new simulation):
    python -m fantasy_experiment.exp1_balance

Balance = held-out P(win) of Team X's expected-points lineup. Decisions are
binned by it, and for each bin and lineup size k we report the share of
decisions where the win-probability pick differs from the points pick and the
mean held-out gain when it does, with 95% CIs from a bootstrap over leagues.

Saves outputs/exp1_balance.csv and outputs/exp1_balance.png.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .multileague import balance_breakdown  # noqa: E402

# Reference palette (light mode): categorical slots 1-2 and chart chrome.
SERIES = {10: "#2a78d6", 5: "#eb6834"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def plot_balance(table, path):
    bins = list(dict.fromkeys(table["Bin"]))
    x = np.arange(len(bins))
    ks = sorted(table["Starters"].unique(), reverse=True)
    dodge = {k: off for k, off in zip(ks, np.linspace(-0.08, 0.08, len(ks)))}
    panels = [("Share_Differ", 100, "Decisions where the picks differ (%)",
               "How often the win-probability pick differs"),
              ("Gain_When_Differ", 100, "Held-out gain in P(win) (pct. points)",
               "Gain when they differ")]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    for ax, (col, scale, ylabel, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for k in ks:
            t = table[table["Starters"] == k].set_index("Bin").loc[bins]
            y = t[col].to_numpy() * scale
            lo = y - t[col + "_CI_low"].to_numpy() * scale
            hi = t[col + "_CI_high"].to_numpy() * scale - y
            xs = x + dodge[k]
            ax.plot(xs, y, color=SERIES[k], lw=1.5, zorder=2)
            ax.errorbar(xs, y, yerr=[lo, hi], fmt="o", ms=7, color=SERIES[k], ecolor=SERIES[k],
                        elinewidth=1.2, capsize=0, mec=SURFACE, mew=1.5, zorder=3,
                        label=f"{k} starters")
            # direct label at the right end, in ink (the marker carries identity)
            ax.annotate(f"k = {k}", (xs[-1], y[-1]), xytext=(24, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_2)
        ax.set_xticks(x, bins)
        ax.set_xlabel("P(win) of the expected-points lineup", fontsize=9, color=INK_2)
        ax.set_xlim(-0.4, len(bins) - 0.1)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)       # the zero line is the baseline
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center")
    n = table.groupby("Starters")["Decisions"].sum().iloc[0]
    mid = table[table["Bin"] == bins[len(bins) // 2]].set_index("Starters")["Differ"]
    fig.text(0.01, 0.045, f"{n:,} decisions per lineup size (50 leagues x 30 ordered pairs), "
             "Jan 12-18, 2026. Bars: 95% CI, bootstrap over leagues.", fontsize=8, color=MUTED)
    fig.text(0.01, 0.01, f"Right panel, middle bin: only {mid.get(10, 0)} (k = 10) and "
             f"{mid.get(5, 0)} (k = 5) disagreements, so its gains are imprecise.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detail", default=str(config.OUTPUT_DIR / "exp1_leagues_detail.csv"))
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)

    detail = pd.read_csv(args.detail)
    table = balance_breakdown(detail, np.random.default_rng(args.seed))
    csv_path = config.OUTPUT_DIR / "exp1_balance.csv"
    png_path = config.OUTPUT_DIR / "exp1_balance.png"
    table.to_csv(csv_path, index=False)
    plot_balance(table, png_path)

    show = table[["Starters", "Bin", "Decisions", "Differ", "Share_Differ", "Share_Differ_CI_low",
                  "Share_Differ_CI_high", "Gain_When_Differ", "Gain_When_Differ_CI_low",
                  "Gain_When_Differ_CI_high"]]
    print(show.round(4).to_string(index=False))
    print(f"\nSaved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
