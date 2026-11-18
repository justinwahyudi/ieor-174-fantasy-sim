"""Weekly player simulation with availability.

For each scheduled game: the player plays with probability Avail; if he plays,
his fantasy score is Normal(Mean_PPG, Std_PPG), optionally clipped at 0.
Players are independent. Pass in ONE shared rng and simulate every rostered
player in a single call: the resulting scenario table is then reused for every
lineup, which gives common random numbers for free.
"""
import numpy as np

from .config import CLIP_AT_ZERO


def simulate_player_games(mean, std, avail, games, n, rng, clip_at_zero=CLIP_AT_ZERO):
    """Game-level scores, shape (n, games); 0 for games the player misses."""
    if games == 0:
        return np.zeros((n, 0))
    pts = rng.normal(mean, std, size=(n, games))
    if clip_at_zero:
        pts = np.maximum(pts, 0.0)
    played = rng.random(size=(n, games)) < avail
    return pts * played


def simulate_weekly_totals(players, n, rng, clip_at_zero=CLIP_AT_ZERO):
    """players: DataFrame with Player, Mean_PPG, Std_PPG, Avail, Games.

    Returns {player: weekly totals, shape (n,)}. Position i in every array is
    the same simulated week.
    """
    totals = {}
    for r in players.itertuples():
        games = simulate_player_games(
            float(r.Mean_PPG), float(r.Std_PPG), float(r.Avail), int(r.Games),
            n, rng, clip_at_zero,
        )
        totals[r.Player] = games.sum(axis=1)
    return totals
