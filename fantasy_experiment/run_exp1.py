"""Experiment 1: which starting lineup is best, by expected points vs win probability?

Demo (no data needed):
    python -m fantasy_experiment.run_exp1 --toy

Real week (after download_data and draft):
    python -m fantasy_experiment.run_exp1 --week-start 2026-01-12 --week-end 2026-01-18

Matchup: --team-x/--team-y, or by default the two teams whose EXPECTED weekly
totals (best k by expected points, from the formula, no simulation) are
closest; Team X is the one earlier in name order. Team X chooses its lineup;
Team Y's lineup is fixed (Starter=1 in rosters.csv, else its top k by expected points).

Two independent runs of every drafted player are drawn from one rng: the
SELECTION run ranks lineups and picks the best by win probability; the
EVALUATION run re-scores the two picks, giving an unbiased gain estimate
(choosing the max of many noisy estimates and reporting that same estimate is
biased upward). All matchups here and in run_exp1_pairs share both runs.

Saves outputs/exp1_lineups.csv (all Team X lineups ranked by win probability).
"""
import argparse

import numpy as np
import pandas as pd

from . import config
from .data_prep import attach_schedule, load_gamelog, player_estimates
from .demo import toy_players
from .lineups import closest_pair, evaluate_matchup, opponent_lineup
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


def simulate_two_runs(players, n, seed):
    """Selection run and an independent evaluation run, both from one rng(seed)."""
    rng = np.random.default_rng(seed)
    return simulate_weekly_totals(players, n, rng), simulate_weekly_totals(players, n, rng)


def roster_of(rosters, team):
    return list(rosters.loc[rosters["Team"] == team, "Player"])


def run(args):
    players, rosters = build_inputs(args)
    k = args.starters
    (auto_x, auto_y), team_totals = closest_pair(players, rosters, k)
    tx, ty = args.team_x or auto_x, args.team_y or auto_y
    rule = "given on command line" if args.team_x else "closest expected weekly totals"

    totals, totals_eval = simulate_two_runs(players, args.n, args.seed)

    y_lineup = opponent_lineup(players, rosters, ty, k)
    table, s = evaluate_matchup(totals, players, roster_of(rosters, tx), y_lineup, k, totals_eval)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "exp1_lineups.csv"
    table.to_csv(out, index=False)

    print(f"Expected weekly totals (best {k} by expected points): "
          + ", ".join(f"{t} {v:.1f}" for t, v in team_totals.items()))
    print(f"Matchup: {tx} (chooses) vs {ty} (fixed)   [{rule}]")
    print(f"Opponent lineup: {s['Opp_Lineup']}")
    print(f"Opponent mean {s['Opp_Mean']:.1f}, SD {s['Opp_SD']:.1f}   (n={args.n}, seed={args.seed})")
    print(f"{len(table)} candidate lineups of {k} from {len(roster_of(rosters, tx))} players\n")
    show = table.head(5)[["Lineup", "Theory_Pts", "Expected_Pts", "SD", "P_win",
                          "CI_low", "CI_high", "Rank_Pts", "Rank_Pwin"]]
    print(show.round(3).to_string(index=False))

    print(f"\nBest by expected points: {s['Best_Pts']}  (P_win {s['P_win_pts_pick']:.3f})")
    print(f"Best by win probability: {s['Best_Pwin']}  (P_win {s['P_win_best']:.3f})")
    if not s["Differ"]:
        print("Same lineup: simulation did not change the decision here.")
    print(f"In-sample paired P_win difference, best minus {s['Compared_To']}: {s['Diff']:+.4f}  "
          f"95% CI ({s['Diff_CI_low']:+.4f}, {s['Diff_CI_high']:+.4f})  (selection run; biased up)")
    print(f"Held-out (independent run, n={args.n}): P_win of win-prob pick {s['Holdout_P_win_best']:.4f}, "
          f"of points pick {s['Holdout_P_win_pts_pick']:.4f}")
    print(f"Held-out gain of win-prob pick over points pick: {s['Holdout_Gain']:+.4f}  "
          f"95% CI ({s['Holdout_CI_low']:+.4f}, {s['Holdout_CI_high']:+.4f})"
          + ("  (picks agree: gain is 0 by definition)" if not s["Differ"] else ""))
    print(f"\nSaved {out}")
    return table


def add_common_args(p):
    p.add_argument("--toy", action="store_true", help="use hypothetical demo players")
    p.add_argument("--gamelog", default=str(config.RAW_GAMELOG))
    p.add_argument("--rosters", default=str(config.ROSTERS_CSV))
    p.add_argument("--week-start")
    p.add_argument("--week-end")
    p.add_argument("--cutoff", help="use games before this date for estimates (default: week start)")
    p.add_argument("--n", type=int, default=config.N_SIM)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--starters", type=int, default=config.N_STARTERS)


def check_args(p, args):
    if not args.toy and not (args.week_start and args.week_end):
        p.error("give --week-start and --week-end (or use --toy)")
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--team-x", help="team that chooses its lineup (default: closest pair)")
    p.add_argument("--team-y", help="opponent with a fixed lineup")
    args = p.parse_args(argv)
    if bool(args.team_x) != bool(args.team_y):
        p.error("give both --team-x and --team-y, or neither")
    return check_args(p, args)


if __name__ == "__main__":
    run(parse_args())
