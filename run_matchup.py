"""Stage 4: simulate Team X vs Team Y and compare with the exact theory value.

Run:  python run_matchup.py
"""
import numpy as np

from simulation import (
    estimate_win_prob, simulate_all_players, simulate_team,
    team_theory, theory_win_prob,
)
from toy_data import PLAYERS, TEAM_X, TEAM_Y

N = 10_000
SEED = 42


def main():
    rng = np.random.default_rng(SEED)
    results = simulate_all_players(PLAYERS, N, rng)

    mu_x, sd_x = team_theory(PLAYERS, TEAM_X)
    mu_y, sd_y = team_theory(PLAYERS, TEAM_Y)
    print(f"Team X: mean {mu_x:.0f}, SD {sd_x:.2f}")
    print(f"Team Y: mean {mu_y:.0f}, SD {sd_y:.2f}")
    print(f"Theory P(X wins):    {theory_win_prob(mu_x, sd_x, mu_y, sd_y):.3f}")

    x = simulate_team(results, TEAM_X)
    y = simulate_team(results, TEAM_Y)
    p, ci = estimate_win_prob(x, y)
    print(f"Simulated P(X wins): {p:.3f}   95% CI: ({ci[0]:.3f}, {ci[1]:.3f})   n={N}")


if __name__ == "__main__":
    main()
