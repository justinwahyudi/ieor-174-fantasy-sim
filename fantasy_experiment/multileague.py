"""Experiment 1 over many independently drafted leagues.

One league = one seeded snake draft (draft.py) + a selection run and an
independent evaluation run of every drafted player. In each league every
ordered pair of teams is one decision (30 for 6 teams), evaluated for each
lineup size k on the SAME draft and the SAME simulated weeks (common random
numbers across k).

Decisions in one league share rosters and simulated weeks, so they are not
independent. Uncertainty is therefore measured by resampling whole leagues
(a cluster bootstrap), not individual decisions.
"""
import numpy as np
import pandas as pd

from .draft import snake_draft
from .model import simulate_weekly_totals
from .run_exp1_pairs import run_all_pairs


def run_league(est, rng, ks, n):
    """Draft one league with rng, simulate it twice, evaluate all pairs for each k.

    est: estimates with schedule (Player, Mean_PPG, Std_PPG, Avail, Games, GP).
    Returns one row per (k, ordered pair).
    """
    draft = snake_draft(est, rng)
    rosters = draft[["Team", "Player"]]
    players = est[est["Player"].isin(rosters["Player"])].reset_index(drop=True)
    totals = simulate_weekly_totals(players, n, rng)        # selection run
    totals_eval = simulate_weekly_totals(players, n, rng)   # evaluation run
    frames = [run_all_pairs(players, rosters, totals, k, totals_eval).assign(Starters=k) for k in ks]
    return pd.concat(frames, ignore_index=True)


def run_leagues(est, rng, n_leagues, ks, n, progress=None):
    """Run n_leagues leagues; each gets its own independent child generator."""
    frames = []
    for i, child in enumerate(rng.spawn(n_leagues), start=1):
        frames.append(run_league(est, child, ks, n).assign(League=i))
        if progress:
            progress(i)
    return pd.concat(frames, ignore_index=True)


def league_table(detail):
    """Per (Starters, League): decisions, number that differ, sum of held-out gains."""
    return (detail.groupby(["Starters", "League"])
            .agg(Decisions=("Differ", "size"), Differ=("Differ", "sum"),
                 Gain_Sum=("Holdout_Gain", "sum"),
                 Significant=("Holdout_Significant", "sum"))
            .reset_index())


def _stats(lt):
    d, n, g = lt["Differ"].sum(), lt["Decisions"].sum(), lt["Gain_Sum"].sum()
    return {"Share_Differ": d / n,
            "Gain_All": g / n,                              # per decision, 0 when picks agree
            "Gain_When_Differ": g / d if d else np.nan}


def bootstrap_summary(detail, rng, n_boot=10_000):
    """Point estimates and 95% percentile CIs, resampling leagues with replacement.

    The same resampled leagues are used for every k, so the k differences
    are paired.
    """
    lt = league_table(detail)
    ks = sorted(lt["Starters"].unique())
    leagues = np.sort(lt["League"].unique())
    by_k = {k: lt[lt["Starters"] == k].set_index("League").loc[leagues] for k in ks}
    idx = rng.integers(0, len(leagues), size=(n_boot, len(leagues)))

    rows = []
    for k in ks:
        t = by_k[k]
        point = _stats(t)
        arr = {key: [] for key in point}
        D, N, G = (t[c].to_numpy() for c in ("Differ", "Decisions", "Gain_Sum"))
        d, nn, g = D[idx].sum(1), N[idx].sum(1), G[idx].sum(1)
        boots = {"Share_Differ": d / nn, "Gain_All": g / nn,
                 "Gain_When_Differ": np.where(d > 0, g / np.maximum(d, 1), np.nan)}
        row = {"Starters": k, "Leagues": len(leagues), "Decisions": int(N.sum()),
               "Differ": int(D.sum()), "Holdout_Significant": int(t["Significant"].sum())}
        for key, val in point.items():
            b = boots[key][~np.isnan(boots[key])]
            row[key] = val
            row[key + "_CI_low"], row[key + "_CI_high"] = np.percentile(b, [2.5, 97.5])
        rows.append(row)
    return pd.DataFrame(rows), lt
