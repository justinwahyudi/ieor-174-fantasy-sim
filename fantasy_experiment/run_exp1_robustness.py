"""Robustness of Experiment 1 to the random seed and the number of replications.

Run:  python -m fantasy_experiment.run_exp1_robustness --week-start 2026-01-12 --week-end 2026-01-18

Repeats the all-pairs analysis (30 ordered pairs, selection run + independent
evaluation run) for every seed in config.ROBUSTNESS_SEEDS and every n in
config.ROBUSTNESS_NS. A disagreement between the two picks that is real should
show up for most seeds at large n, with a held-out gain above 0; one caused by
Monte Carlo noise comes and goes with the seed.

Saves:
  outputs/exp1_robustness.csv          one row per (n, seed, ordered pair)
  outputs/exp1_robustness_summary.csv  one row per (n, seed)
  outputs/exp1_robustness_stability.csv  one row per (n, ordered pair): in how many
                                        seeds the picks differ, and the mean held-out gain
"""
import argparse

import pandas as pd

from . import config
from .run_exp1 import add_common_args, build_inputs, check_args, simulate_two_runs
from .run_exp1_pairs import run_all_pairs, summarize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    args = check_args(p, p.parse_args(argv))
    players, rosters = build_inputs(args)

    frames, summaries = [], []
    for n in config.ROBUSTNESS_NS:
        for seed in config.ROBUSTNESS_SEEDS:
            totals, totals_eval = simulate_two_runs(players, n, seed)
            df = run_all_pairs(players, rosters, totals, args.starters, totals_eval)
            frames.append(df.assign(N=n, Seed=seed))
            summaries.append({"N": n, "Seed": seed, **summarize(df)})
            print(f"done n={n} seed={seed}")
    detail = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    stability = (detail.groupby(["N", "Team_X", "Team_Y"])
                 .agg(Seeds=("Seed", "nunique"), Seeds_Differ=("Differ", "sum"),
                      Holdout_Gain_Mean=("Holdout_Gain", "mean"),
                      Seeds_Holdout_Significant=("Holdout_Significant", "sum"),
                      P_win_pts_pick=("Holdout_P_win_pts_pick", "mean"))
                 .reset_index())

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    detail.to_csv(config.OUTPUT_DIR / "exp1_robustness.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "exp1_robustness_summary.csv", index=False)
    stability.to_csv(config.OUTPUT_DIR / "exp1_robustness_stability.csv", index=False)

    print("\nPer (n, seed):")
    print(summary.round(4).to_string(index=False))
    print("\nOrdered pairs where the picks differ in at least one seed:")
    print(stability[stability["Seeds_Differ"] > 0].round(4).to_string(index=False))
    print(f"\nSaved outputs/exp1_robustness*.csv")


if __name__ == "__main__":
    main()
