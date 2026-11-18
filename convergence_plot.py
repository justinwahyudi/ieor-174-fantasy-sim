"""Convergence check: CI half-width of the win-probability estimate vs n.

Expected: a straight line with slope about -1/2 on log-log axes (1/sqrt(n)).

Run:  python convergence_plot.py     (saves outputs/convergence.png)
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window needed; we only save a file
import matplotlib.pyplot as plt
import numpy as np

from simulation import (
    estimate_win_prob, simulate_all_players, simulate_team,
    team_theory, theory_win_prob,
)
from toy_data import PLAYERS, TEAM_X, TEAM_Y

SEED = 42
NS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
OUT = Path("outputs") / "convergence.png"


def main():
    mu_x, sd_x = team_theory(PLAYERS, TEAM_X)
    mu_y, sd_y = team_theory(PLAYERS, TEAM_Y)
    p_theory = theory_win_prob(mu_x, sd_x, mu_y, sd_y)

    half = []
    for n in NS:
        results = simulate_all_players(PLAYERS, n, np.random.default_rng(SEED))
        _, ci = estimate_win_prob(simulate_team(results, TEAM_X),
                                  simulate_team(results, TEAM_Y))
        half.append((ci[1] - ci[0]) / 2)

    ref = [1.96 * math.sqrt(p_theory * (1 - p_theory) / n) for n in NS]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(NS, ref, color="#52514e", lw=1.5, ls="--", label="Theory: 1.96*sqrt(p(1-p)/n)")
    ax.plot(NS, half, color="#2a78d6", lw=2, marker="o", label="Simulated CI half-width")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Replications (n)")
    ax.set_ylabel("95% CI half-width of P(X wins)")
    ax.set_title("Win-probability estimate converges at rate 1/sqrt(n)", loc="left")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()

    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=200)
    slope = np.polyfit(np.log(NS), np.log(half), 1)[0]
    print(f"Saved {OUT}. Fitted log-log slope: {slope:.3f} (expect about -0.5)")


if __name__ == "__main__":
    main()
