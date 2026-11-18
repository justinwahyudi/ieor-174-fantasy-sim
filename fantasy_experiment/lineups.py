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
