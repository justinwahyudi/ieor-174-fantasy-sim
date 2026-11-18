"""Experiment 1: which starting lineup is best, by expected points vs win probability?

Demo (no data needed):
    python -m fantasy_experiment.run_exp1 --toy

Real week (after download_data and creating data/rosters.csv):
    python -m fantasy_experiment.run_exp1 --week-start 2026-01-12 --week-end 2026-01-18

Saves outputs/exp1_lineups.csv (all lineups ranked by win probability).
"""
import argparse

import numpy as np
import pandas as pd

from . import config
from .data_prep import attach_schedule, load_gamelog, player_estimates
from .demo import toy_players
from .lineups import (
    compare_lineups, lineup_total, paired_win_diff, top_k_by_expected,
)
from .model import simulate_weekly_totals


def build_inputs(args):
    """Return (players DataFrame, rosters DataFrame)."""
    if args.toy:
        return toy_players()
    gl = load_gamelog(args.gamelog)
    cutoff = args.cutoff or args.week_start
    est = attach_schedule(player_estimates(gl, cutoff), gl, args.week_start, args.week_end)
    rosters = pd.read_csv(args.rosters)
    missing = sorted(set(rosters["Player"]) - set(est["Player"]))
    if missing:
        print(f"WARNING: no estimates for (dropped): {missing}")
    players = est[est["Player"].isin(rosters["Player"])].reset_index(drop=True)
    return players, rosters[rosters["Player"].isin(players["Player"])]


def run(args):
    players, rosters = build_inputs(args)
    k = args.starters
    x_roster = list(rosters.loc[rosters["Team"] == "X", "Player"])
    y_roster = list(rosters.loc[rosters["Team"] == "Y", "Player"])

    rng = np.random.default_rng(args.seed)
    totals = simulate_weekly_totals(players, args.n, rng)   # one shared scenario table

    # Opponent lineup: marked Starter=1 in rosters.csv, else the top-k by expected points
    y_players = players[players["Player"].isin(y_roster)]
    if "Starter" in rosters.columns and rosters.loc[rosters["Team"] == "Y", "Starter"].sum() == k:
        y_lineup = list(rosters.loc[(rosters["Team"] == "Y") & (rosters["Starter"] == 1), "Player"])
    else:
        y_lineup = top_k_by_expected(y_players, k)
    opp_total = lineup_total(totals, y_lineup)

    table = compare_lineups(totals, x_roster, k, opp_total)
    best_pwin = table.iloc[0]
    best_pts = table.loc[table["Expected_Pts"].idxmax()]

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "exp1_lineups.csv"
    table.to_csv(out, index=False)

    print(f"Opponent lineup: {', '.join(y_lineup)}")
    print(f"Opponent mean {opp_total.mean():.1f}, SD {opp_total.std(ddof=1):.1f}   (n={args.n}, seed={args.seed})")
    print(f"{len(table)} candidate lineups of {k} from {len(x_roster)} players\n")
    show = table.head(5)[["Lineup", "Expected_Pts", "SD", "P_win", "CI_low", "CI_high", "Rank_Pts", "Rank_Pwin"]]
    print(show.round(3).to_string(index=False))

    print(f"\nBest by expected points: {best_pts['Lineup']}  (P_win {best_pts['P_win']:.3f})")
    print(f"Best by win probability: {best_pwin['Lineup']}  (P_win {best_pwin['P_win']:.3f})")
    if best_pts["Lineup"] == best_pwin["Lineup"]:
        print("Same lineup: simulation did not change the decision here.")
    else:
        diff, (lo, hi) = paired_win_diff(
            totals, best_pwin["Lineup"].split(", "), best_pts["Lineup"].split(", "), opp_total)
        print(f"Win-prob gain of the simulation pick: {diff:+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})  (paired, same weeks)")
    print(f"\nSaved {out}")
    return table


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--toy", action="store_true", help="use hypothetical demo players")
    p.add_argument("--gamelog", default=str(config.RAW_GAMELOG))
    p.add_argument("--rosters", default=str(config.ROSTERS_CSV))
    p.add_argument("--week-start")
    p.add_argument("--week-end")
    p.add_argument("--cutoff", help="use games before this date for estimates (default: week start)")
    p.add_argument("--n", type=int, default=config.N_SIM)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--starters", type=int, default=config.N_STARTERS)
    args = p.parse_args(argv)
    if not args.toy and not (args.week_start and args.week_end):
        p.error("give --week-start and --week-end (or use --toy)")
    return args


if __name__ == "__main__":
    run(parse_args())
