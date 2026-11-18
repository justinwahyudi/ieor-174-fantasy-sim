"""Compare lineups using the same simulated weeks (common random numbers)."""
import math
from itertools import combinations

import pandas as pd

from simulation import estimate_win_prob


def lineup_total(totals, lineup):
    return sum(totals[p] for p in lineup)


def win_indicator(team_total, opp_total):
    """1 if win, 0.5 if tie, 0 if loss, for every simulated week."""
    return (team_total > opp_total) * 1.0 + (team_total == opp_total) * 0.5


def compare_lineups(totals, roster, k, opp_total):
    """Evaluate every k-player lineup from roster against a fixed opponent total."""
    rows = []
    for lineup in combinations(roster, k):
        t = lineup_total(totals, lineup)
        p, (lo, hi) = estimate_win_prob(t, opp_total)
        rows.append({
            "Lineup": ", ".join(lineup),
            "Expected_Pts": float(t.mean()),
            "SD": float(t.std(ddof=1)),
            "P_win": p, "CI_low": lo, "CI_high": hi,
        })
    df = pd.DataFrame(rows)
    df["Rank_Pts"] = df["Expected_Pts"].rank(ascending=False, method="min").astype(int)
    df["Rank_Pwin"] = df["P_win"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("P_win", ascending=False).reset_index(drop=True)


def paired_win_diff(totals, lineup_a, lineup_b, opp_total):
    """P(win | A) - P(win | B) with a 95% CI from paired per-week differences.

    Because both lineups are scored on the same weeks, this CI is much tighter
    than comparing two independent estimates.
    """
    d = (win_indicator(lineup_total(totals, lineup_a), opp_total)
         - win_indicator(lineup_total(totals, lineup_b), opp_total))
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1)) / math.sqrt(n)
    return mean, (mean - 1.96 * se, mean + 1.96 * se)


def top_k_by_expected(players, k):
    """Names of the k players with highest Mean_PPG * Avail * Games."""
    score = players["Mean_PPG"] * players["Avail"] * players["Games"]
    return list(players.assign(_s=score).nlargest(k, "_s")["Player"])


def expected_week_pts(players):
    """Theoretical expected weekly points per player: Mean_PPG * Avail * Games.

    Ignores the tiny effect of clipping scores at 0.
    """
    return (players["Mean_PPG"] * players["Avail"] * players["Games"]).set_axis(players["Player"])


def team_expected_total(players, roster, k):
    """Expected weekly total of a team's best-by-expected-points k-player lineup."""
    exp = expected_week_pts(players)
    return float(exp[list(roster)].nlargest(k).sum())


def closest_pair(players, rosters, k):
    """The two teams whose expected weekly totals are closest (no simulation used).

    Returns (team_x, team_y), ordered by team name; ties go to the first pair in that order.
    """
    teams = sorted(rosters["Team"].unique())
    totals = {t: team_expected_total(players, rosters.loc[rosters["Team"] == t, "Player"], k)
              for t in teams}
    return min(combinations(teams, 2), key=lambda ab: abs(totals[ab[0]] - totals[ab[1]])), totals


def opponent_lineup(players, rosters, team, k):
    """Players marked Starter=1 for `team` if exactly k are marked, else its top k by expected points."""
    ros = rosters[rosters["Team"] == team]
    if "Starter" in ros.columns and ros["Starter"].fillna(0).sum() == k:
        return list(ros.loc[ros["Starter"] == 1, "Player"])
    return top_k_by_expected(players[players["Player"].isin(ros["Player"])], k)


def evaluate_matchup(totals, players, x_roster, y_lineup, k):
    """Experiment 1 for one matchup: every Team X lineup against Team Y's fixed lineup.

    Returns (table, summary). The points pick is the lineup with the highest
    THEORETICAL expected points (what a points maximizer would choose, with no
    Monte Carlo noise). If the win-probability pick differs, the summary holds
    the paired gain over the points pick; if not, the paired margin over the
    runner-up by win probability.
    """
    opp = lineup_total(totals, y_lineup)
    exp = expected_week_pts(players)
    table = compare_lineups(totals, x_roster, k, opp)
    table["Theory_Pts"] = table["Lineup"].map(lambda s: float(exp[s.split(", ")].sum()))
    table["Rank_Pts"] = table["Theory_Pts"].rank(ascending=False, method="min").astype(int)

    best_pwin = table.iloc[0]
    best_pts = table.loc[table["Theory_Pts"].idxmax()]
    differ = best_pwin["Lineup"] != best_pts["Lineup"]
    other = best_pts if differ else table.iloc[1]
    diff, (lo, hi) = paired_win_diff(
        totals, best_pwin["Lineup"].split(", "), other["Lineup"].split(", "), opp)
    summary = {
        "Opp_Lineup": ", ".join(y_lineup),
        "Opp_Mean": float(opp.mean()), "Opp_SD": float(opp.std(ddof=1)),
        "Best_Pwin": best_pwin["Lineup"], "Best_Pts": best_pts["Lineup"],
        "Differ": bool(differ),
        "P_win_best": float(best_pwin["P_win"]), "P_win_pts_pick": float(best_pts["P_win"]),
        "Compared_To": "points pick" if differ else "runner-up",
        "Diff": diff, "Diff_CI_low": lo, "Diff_CI_high": hi,
    }
    return table, summary
