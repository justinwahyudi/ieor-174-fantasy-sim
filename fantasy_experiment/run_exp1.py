"""Experiment 1: which starting lineup is best, by expected points vs win probability?

Demo (no data needed):
    python -m fantasy_experiment.run_exp1 --toy

Real week (after download_data and draft):
    python -m fantasy_experiment.run_exp1 --week-start 2026-01-12 --week-end 2026-01-18

Matchup: --team-x/--team-y, or by default the two teams whose EXPECTED weekly
totals (best 5 by expected points, from the formula, no simulation) are
closest; Team X is the one earlier in name order. Team X chooses its lineup;
Team Y's lineup is fixed (Starter=1 in rosters.csv, else its top 5 by expected points).

Every drafted player is simulated once from one rng, so all matchups (here and
in run_exp1_pairs) share the same simulated weeks.

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


def roster_of(rosters, team):
    return list(rosters.loc[rosters["Team"] == team, "Player"])


def run(args):
    players, rosters = build_inputs(args)
    k = args.starters
    (auto_x, auto_y), team_totals = closest_pair(players, rosters, k)
    tx, ty = args.team_x or auto_x, args.team_y or auto_y
    rule = "given on command line" if args.team_x else "closest expected weekly totals"

    rng = np.random.default_rng(args.seed)
    totals = simulate_weekly_totals(players, args.n, rng)   # one shared scenario table

    y_lineup = opponent_lineup(players, rosters, ty, k)
    table, s = evaluate_matchup(totals, players, roster_of(rosters, tx), y_lineup, k)

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    out = config.OUTPUT_DIR / "exp1_lineups.csv"
    table.to_csv(out, index=False)

    print("Expected weekly totals (best 5 by expected points): "
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
    print(f"Paired P_win difference, best minus {s['Compared_To']}: {s['Diff']:+.4f}  "
          f"95% CI ({s['Diff_CI_low']:+.4f}, {s['Diff_CI_high']:+.4f})  (same simulated weeks)")
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
