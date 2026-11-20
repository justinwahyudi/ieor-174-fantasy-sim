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


BALANCE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # DECISION: bins of matchup balance


def balance_breakdown(detail, rng, edges=BALANCE_EDGES, n_boot=10_000):
    """Disagreement and gain by matchup balance, with league-bootstrap 95% CIs.

    Balance = held-out P(win) of the points pick (Holdout_P_win_pts_pick), i.e.
    how likely Team X is to win if it simply starts its expected-points lineup.
    Returns one row per (Starters, bin): Decisions, Differ, Share_Differ,
    Gain_All (mean held-out gain per decision, 0 when picks agree) and
    Gain_When_Differ, each with a CI from resampling whole leagues. The same
    resampled leagues are used for every k and bin.
    """
    d = detail.copy()
    labels = [f"{a:.1f}-{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
    d["Bin"] = pd.cut(d["Holdout_P_win_pts_pick"], list(edges), labels=labels,
                      include_lowest=True)
    leagues = np.sort(d["League"].unique())
    idx = rng.integers(0, len(leagues), size=(n_boot, len(leagues)))

    rows = []
    for k in sorted(d["Starters"].unique()):
        g = (d[d["Starters"] == k]
             .groupby(["League", "Bin"], observed=False)
             .agg(N=("Differ", "size"), D=("Differ", "sum"), G=("Holdout_Gain", "sum")))
        for b in labels:
            t = g.xs(b, level="Bin").reindex(leagues, fill_value=0)
            N, D, G = (t[c].to_numpy(dtype=float) for c in ("N", "D", "G"))
            bn, bd, bg = N[idx].sum(1), D[idx].sum(1), G[idx].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                boots = {"Share_Differ": bd / bn, "Gain_All": bg / bn, "Gain_When_Differ": bg / bd}
            point = {"Share_Differ": D.sum() / N.sum() if N.sum() else np.nan,
                     "Gain_All": G.sum() / N.sum() if N.sum() else np.nan,
                     "Gain_When_Differ": G.sum() / D.sum() if D.sum() else np.nan}
            row = {"Starters": k, "Bin": b, "Decisions": int(N.sum()), "Differ": int(D.sum())}
            for key, val in point.items():
                bb = boots[key][np.isfinite(boots[key])]
                row[key] = val
                row[key + "_CI_low"], row[key + "_CI_high"] = (
                    np.percentile(bb, [2.5, 97.5]) if len(bb) else (np.nan, np.nan))
            rows.append(row)
    return pd.DataFrame(rows)
