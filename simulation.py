"""Core simulation functions for the fantasy basketball Monte Carlo project.

Model (v1, toy): each player's game score is Normal(mean, std), games are
independent, every scheduled game is played, players are independent.
"""
import math

import numpy as np


def simulate_player(mean, std, games, n_simulations, rng):
    """Return weekly point totals for one player, shape (n_simulations,).

    rng is created by the caller (np.random.default_rng(seed)) so the caller
    controls randomness. Never create a fixed-seed generator in here: two
    players simulated with the same seed would be perfectly correlated.
    """
    # rows = simulated weeks, columns = games within a week
    game_points = rng.normal(loc=mean, scale=std, size=(n_simulations, games))
    return game_points.sum(axis=1)


def simulate_all_players(df, n_simulations, rng):
    """Simulate every player in df with one shared rng.

    Returns {player name: weekly totals}. Position i in every array is the
    same simulated week.
    """
    results = {}
    for row in df.itertuples():
        results[row.Player] = simulate_player(
            float(row.Mean_PPG), float(row.Std_PPG), int(row.Games),
            n_simulations, rng,
        )
    return results


def simulate_team(results, players):
    """Add the players' weekly totals, week by week."""
    return sum(results[p] for p in players)


def team_theory(df, players):
    """Theoretical weekly mean and SD of a team of independent players.

    Means add; variances add (not SDs).
    """
    sub = df[df["Player"].isin(players)]
    mean = (sub["Games"] * sub["Mean_PPG"]).sum()
    variance = ((sub["Std_PPG"] ** 2) * sub["Games"]).sum()
    return float(mean), float(np.sqrt(variance))


def norm_cdf(z):
    """Standard normal CDF (no scipy needed)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def theory_win_prob(mu_x, sd_x, mu_y, sd_y):
    """Exact P(X > Y) when both team totals are independent normals."""
    return norm_cdf((mu_x - mu_y) / math.sqrt(sd_x**2 + sd_y**2))


def estimate_win_prob(x, y):
    """Estimate P(X wins) from simulated totals; a tie counts as half a win.

    Returns (p, (ci_low, ci_high)) using a normal-approximation 95% CI.
    """
    n = len(x)
    wins = (x > y).sum() + 0.5 * (x == y).sum()
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    return float(p), (float(p - 1.96 * se), float(p + 1.96 * se))
