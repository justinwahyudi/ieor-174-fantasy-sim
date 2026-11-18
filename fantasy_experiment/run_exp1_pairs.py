"""Experiment 1 for every ordered pair of teams in the league (both directions).

Run:  python -m fantasy_experiment.run_exp1_pairs --week-start 2026-01-12 --week-end 2026-01-18

For each of the 6 x 5 = 30 ordered pairs, Team X chooses its lineup and Team
Y's lineup is fixed (top 5 by expected points). Each unordered pair is run in
both directions, since the favourite and the underdog face different
decisions. All pairs use ONE simulation of every drafted player (common random numbers), so
the main matchup's numbers here equal run_exp1's.

Picks are made on a selection run and re-scored on an independent evaluation
run (see run_exp1), so Holdout_Gain is an unbiased estimate of the gain.

Saves outputs/exp1_pairs.csv and prints how often the win-probability pick
differs from the expected-points pick, and the held-out gain when it does.
"""
import argparse
from itertools import permutations

import pandas as pd

from . import config
from .lineups import closest_pair, evaluate_matchup, opponent_lineup, team_expected_total
from .run_exp1 import add_common_args, build_inputs, check_args, roster_of, simulate_two_runs


def run_all_pairs(players, rosters, totals, k, totals_eval=None):
    rows = []
    for tx, ty in permutations(sorted(rosters["Team"].unique()), 2):
        y_lineup = opponent_lineup(players, rosters, ty, k)
        _, s = evaluate_matchup(totals, players, roster_of(rosters, tx), y_lineup, k, totals_eval)
        rows.append({"Team_X": tx, "Team_Y": ty,
                     "Pair": "-".join(sorted((tx, ty))),
                     "Exp_X": team_expected_total(players, roster_of(rosters, tx), k),
                     "Exp_Y": team_expected_total(players, roster_of(rosters, ty), k), **s})
    df = pd.DataFrame(rows)
    df["Gain_Significant"] = df["Differ"] & (df["Diff_CI_low"] > 0)
    if totals_eval is not None:
        df["Holdout_Significant"] = df["Differ"] & (df["Holdout_CI_low"] > 0)
    return df


def summarize(df):
    """One-row summary of a run_all_pairs table."""
    d = df[df["Differ"]]
    out = {"Decisions": len(df), "Differ": len(d),
           "InSample_Gain_Mean": d["Diff"].mean() if len(d) else 0.0,
           "InSample_Significant": int(d["Gain_Significant"].sum())}
    if "Holdout_Gain" in df:
        out.update({"Holdout_Gain_Mean": d["Holdout_Gain"].mean() if len(d) else 0.0,
                    "Holdout_Gain_Min": d["Holdout_Gain"].min() if len(d) else 0.0,
                    "Holdout_Gain_Max": d["Holdout_Gain"].max() if len(d) else 0.0,
                    "Holdout_Significant": int(d["Holdout_Significant"].sum()),
                    "Holdout_Negative": int((d["Holdout_Gain"] < 0).sum())})
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = check_args(p, p.parse_args(argv))

    players, rosters = build_inputs(args)
    k = args.starters
    totals, totals_eval = simulate_two_runs(players, args.n, args.seed)
    df = run_all_pairs(players, rosters, totals, k, totals_eval)
    (mx, my), _ = closest_pair(players, rosters, k)
    df["Main"] = (df["Team_X"] == mx) & (df["Team_Y"] == my)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "exp1_pairs.csv"
    df.to_csv(out, index=False)

    show = df[["Team_X", "Team_Y", "Exp_X", "Exp_Y", "Holdout_P_win_pts_pick", "Differ",
               "Diff", "Holdout_Gain", "Holdout_CI_low", "Holdout_CI_high", "Main"]]
    print(show.round(4).to_string(index=False))
    sm = summarize(df)
    print(f"\nOrdered pairs (decisions): {sm['Decisions']}   (n={args.n} per run, seed={args.seed})")
    print(f"Win-probability pick differs from points pick: {sm['Differ']} of {sm['Decisions']}")
    if sm["Differ"]:
        print(f"  In-sample gain (selection run, biased up): mean {sm['InSample_Gain_Mean']:+.4f}; "
              f"CI above 0 in {sm['InSample_Significant']}")
        print(f"  Held-out gain (independent run): mean {sm['Holdout_Gain_Mean']:+.4f}, "
              f"min {sm['Holdout_Gain_Min']:+.4f}, max {sm['Holdout_Gain_Max']:+.4f}; "
              f"CI above 0 in {sm['Holdout_Significant']}, negative in {sm['Holdout_Negative']}")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
