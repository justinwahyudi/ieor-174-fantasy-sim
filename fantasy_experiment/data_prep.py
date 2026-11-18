"""Turn a player game log into per-player simulation inputs.

Inputs (from nba_api LeagueGameLog): PLAYER_NAME, TEAM_ID, GAME_ID, GAME_DATE
and the box-score columns used by the scoring formula.

Output columns used by the model: Player, Mean_PPG, Std_PPG, Avail, Games.
Only games BEFORE the cutoff date are used for estimates, to avoid leakage.

Known limitation: Avail = games played / team games is a proxy for availability,
not an injury forecast, and it is biased for players traded mid-season.
"""
import pandas as pd

from .scoring import add_fantasy_points


def load_gamelog(path):
    gl = pd.read_csv(path, parse_dates=["GAME_DATE"])
    return add_fantasy_points(gl)


def team_game_counts(gl, start=None, end=None):
    """Number of distinct games each team played in [start, end]."""
    sub = gl
    if start is not None:
        sub = sub[sub["GAME_DATE"] >= pd.Timestamp(start)]
    if end is not None:
        sub = sub[sub["GAME_DATE"] <= pd.Timestamp(end)]
    return sub.groupby("TEAM_ID")["GAME_ID"].nunique()


def player_estimates(gl, cutoff, min_games=5):
    """Per-player mean/std of fantasy points when playing, and availability rate."""
    hist = gl[gl["GAME_DATE"] < pd.Timestamp(cutoff)]
    team_games = team_game_counts(hist)
    last_team = hist.sort_values("GAME_DATE").groupby("PLAYER_NAME")["TEAM_ID"].last()

    est = hist.groupby("PLAYER_NAME")["FANTASY_PTS"].agg(
        Mean_PPG="mean", Std_PPG="std", GP="count"
    )
    est = est[est["GP"] >= min_games].copy()
    est["Std_PPG"] = est["Std_PPG"].fillna(0.0)
    est["TEAM_ID"] = last_team.reindex(est.index)
    est["Team_Games"] = est["TEAM_ID"].map(team_games)
    est["Avail"] = (est["GP"] / est["Team_Games"]).clip(upper=1.0)
    return est.reset_index().rename(columns={"PLAYER_NAME": "Player"})


def attach_schedule(est, gl, start, end):
    """Add Games = number of games the player's team plays in [start, end]."""
    week = team_game_counts(gl, start, end)
    out = est.copy()
    out["Games"] = out["TEAM_ID"].map(week).fillna(0).astype(int)
    return out
