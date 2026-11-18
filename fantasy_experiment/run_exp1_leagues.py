"""Experiment 1 across many drafted leagues, for several lineup sizes.

Run:  python -m fantasy_experiment.run_exp1_leagues --week-start 2026-01-12 --week-end 2026-01-18
      (40-90 s per league at n = 200,000; 50 leagues took 35-75 min)

One rng(seed) is created here. Each league gets an independent child generator
(rng.spawn) that drives its draft and its two simulation runs; the bootstrap
uses the parent rng afterwards.

Saves:
  outputs/exp1_leagues_detail.csv   one row per (league, k, ordered pair)
  outputs/exp1_leagues_by_league.csv  one row per (league, k)
  outputs/exp1_leagues_summary.csv  one row per k, with league-bootstrap 95% CIs
"""
import argparse
from math import comb

import numpy as np

from . import config
from .data_prep import attach_schedule, load_gamelog, player_estimates
from .draft import N_ROUNDS
from .multileague import bootstrap_summary, run_leagues


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamelog", default=str(config.RAW_GAMELOG))
    p.add_argument("--week-start", required=True)
    p.add_argument("--week-end", required=True)
    p.add_argument("--cutoff", help="use games before this date (default: week start)")
    p.add_argument("--leagues", type=int, default=config.N_LEAGUES)
    p.add_argument("--starters", type=int, nargs="+", default=list(config.STARTERS_COMPARE))
    p.add_argument("--n", type=int, default=config.N_SIM)
    p.add_argument("--seed", type=int, default=config.SEED)
    args = p.parse_args(argv)

    gl = load_gamelog(args.gamelog)
    est = attach_schedule(player_estimates(gl, args.cutoff or args.week_start), gl,
                          args.week_start, args.week_end)
    rng = np.random.default_rng(args.seed)
    detail = run_leagues(est, rng, args.leagues, args.starters, args.n,
                         progress=lambda i: print(f"league {i}/{args.leagues} done", flush=True))
    summary, by_league = bootstrap_summary(detail, rng)
    summary.insert(1, "Lineups_per_decision", [comb(N_ROUNDS, k) for k in summary["Starters"]])

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    detail.to_csv(config.OUTPUT_DIR / "exp1_leagues_detail.csv", index=False)
    by_league.to_csv(config.OUTPUT_DIR / "exp1_leagues_by_league.csv", index=False)
    summary.to_csv(config.OUTPUT_DIR / "exp1_leagues_summary.csv", index=False)

    print(f"\n{args.leagues} leagues x 30 ordered pairs, n={args.n} per run, seed={args.seed}")
    print(summary.T.to_string(header=False))
    print("\nSaved outputs/exp1_leagues_*.csv")


if __name__ == "__main__":
    main()
