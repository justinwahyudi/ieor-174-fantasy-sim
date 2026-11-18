"""Build league rosters with a seeded, noisy snake draft from the game log.

Run:  python -m fantasy_experiment.draft --cutoff 2026-01-12

Pool: every player with at least MIN_GP games before the cutoff.
Value: each player's average fantasy points per game played before the cutoff
(our placeholder scoring, config.SCORING).
Noise: each team gets its own draft board, value + Normal(0, NOISE_SD), drawn
once for every (team, player) from the shared rng. So teams disagree a little
about players of similar value, but nobody passes on a clearly better player.
Order: snake (T1..T6, T6..T1, ...) for N_ROUNDS rounds; each pick takes the
available player highest on that team's board.

Writes data/rosters.csv (Team, Player, Starter blank) and
outputs/draft_order.csv (every pick with its round and both values).
"""
import argparse

import numpy as np
import pandas as pd

from . import config
from .data_prep import load_gamelog, player_estimates

N_TEAMS = 6
N_ROUNDS = 13
MIN_GP = 25
NOISE_SD = 2.0   # DECISION: fantasy points of board noise per team and player


def team_names(n_teams):
    """T1..Tn, numbered by draft slot (T1 picks first)."""
    return [f"T{i}" for i in range(1, n_teams + 1)]


def snake_order(n_teams, n_rounds):
    """Team name for every pick: T1..Tn, then Tn..T1, and so on."""
    teams = team_names(n_teams)
    order = []
    for rnd in range(n_rounds):
        order.extend(teams if rnd % 2 == 0 else teams[::-1])
    return order


def snake_draft(est, rng, n_teams=N_TEAMS, n_rounds=N_ROUNDS, min_gp=MIN_GP, noise_sd=NOISE_SD):
    """Return one row per pick: Pick, Round, Team, Player, Mean_PPG, Board_Value.

    est: player estimates with Player, Mean_PPG, GP. rng: a numpy Generator.
    """
    pool = (est[est["GP"] >= min_gp]
            .sort_values(["Mean_PPG", "Player"], ascending=[False, True])
            .reset_index(drop=True))
    n_picks = n_teams * n_rounds
    if len(pool) < n_picks:
        raise ValueError(f"pool has {len(pool)} players, need {n_picks}")

    teams = team_names(n_teams)
    noise = rng.normal(0.0, noise_sd, size=(n_teams, len(pool)))   # one call: all boards
    boards = pool["Mean_PPG"].to_numpy()[None, :] + noise
    available = np.ones(len(pool), dtype=bool)

    rows = []
    for pick, team in enumerate(snake_order(n_teams, n_rounds), start=1):
        t = teams.index(team)
        i = int(np.argmax(np.where(available, boards[t], -np.inf)))
        available[i] = False
        rows.append({
            "Pick": pick, "Round": (pick - 1) // n_teams + 1, "Team": team,
            "Player": pool.at[i, "Player"], "Mean_PPG": pool.at[i, "Mean_PPG"],
            "Board_Value": boards[t, i],
        })
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gamelog", default=str(config.RAW_GAMELOG))
    p.add_argument("--cutoff", required=True, help="only games before this date are used")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--out", default=str(config.ROSTERS_CSV))
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    est = player_estimates(load_gamelog(args.gamelog), args.cutoff)
    draft = snake_draft(est, rng)

    draft[["Team", "Player"]].assign(Starter="").to_csv(args.out, index=False)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    order_path = config.OUTPUT_DIR / "draft_order.csv"
    draft.to_csv(order_path, index=False)

    print(f"Pool: {int((est['GP'] >= MIN_GP).sum())} players with >= {MIN_GP} games; "
          f"noise SD {NOISE_SD}; seed {args.seed}")
    print(draft.groupby("Team")["Mean_PPG"].agg(["count", "mean", "sum"]).round(1).to_string())
    print(f"\nSaved {args.out} and {order_path}")


if __name__ == "__main__":
    main()
