"""Experiment 1 for every ordered pair of teams in the league (both directions).

Run:  python -m fantasy_experiment.run_exp1_pairs --week-start 2026-01-12 --week-end 2026-01-18

For each of the 6 x 5 = 30 ordered pairs, Team X chooses its lineup and Team
Y's lineup is fixed (top 5 by expected points). Each unordered pair is run in
both directions, since the favourite and the underdog face different
decisions. All pairs use ONE simulation of every drafted player (common random numbers), so
the main matchup's numbers here equal run_exp1's.

Saves outputs/exp1_pairs.csv and prints how often the win-probability pick
differs from the expected-points pick, and the paired gain when it does.
"""
import argparse
from itertools import permutations

import numpy as np
import pandas as pd

from . import config
from .lineups import closest_pair, evaluate_matchup, opponent_lineup, team_expected_total
from .model import simulate_weekly_totals
from .run_exp1 import add_common_args, build_inputs, check_args, roster_of


def run_all_pairs(players, rosters, totals, k):
    rows = []
    for tx, ty in permutations(sorted(rosters["Team"].unique()), 2):
        y_lineup = opponent_lineup(players, rosters, ty, k)
        _, s = evaluate_matchup(totals, players, roster_of(rosters, tx), y_lineup, k)
        rows.append({"Team_X": tx, "Team_Y": ty,
                     "Pair": "-".join(sorted((tx, ty))),
                     "Exp_X": team_expected_total(players, roster_of(rosters, tx), k),
                     "Exp_Y": team_expected_total(players, roster_of(rosters, ty), k), **s})
    df = pd.DataFrame(rows)
    df["Gain_Significant"] = df["Differ"] & (df["Diff_CI_low"] > 0)
    return df


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = check_args(p, p.parse_args(argv))

    players, rosters = build_inputs(args)
    k = args.starters
    rng = np.random.default_rng(args.seed)
    totals = simulate_weekly_totals(players, args.n, rng)
    df = run_all_pairs(players, rosters, totals, k)
    (mx, my), _ = closest_pair(players, rosters, k)
    df["Main"] = (df["Team_X"] == mx) & (df["Team_Y"] == my)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "exp1_pairs.csv"
    df.to_csv(out, index=False)

    show = df[["Team_X", "Team_Y", "Exp_X", "Exp_Y", "P_win_pts_pick", "P_win_best",
               "Differ", "Diff", "Diff_CI_low", "Diff_CI_high", "Main"]]
    print(show.round(4).to_string(index=False))
    d = df[df["Differ"]]
    print(f"\nOrdered pairs (decisions): {len(df)}   (n={args.n}, seed={args.seed})")
    print(f"Win-probability pick differs from points pick: {len(d)} of {len(df)}")
    if len(d):
        print(f"  Gain when they differ: mean {d['Diff'].mean():+.4f}, "
              f"min {d['Diff'].min():+.4f}, max {d['Diff'].max():+.4f}")
        print(f"  Gain with 95% CI above 0: {int(d['Gain_Significant'].sum())} of {len(d)}")
    print(f"\nSaved {out}")
    return df


if __name__ == "__main__":
    main()
