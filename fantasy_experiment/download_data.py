"""Download the season's player game logs with nba_api (run on YOUR machine).

Run:  python -m fantasy_experiment.download_data

Needs internet and `pip install nba_api`. Saves data/raw/gamelog_<season>.csv
(git-ignored). Not verified against the live API from the build environment,
so if a column name differs, fix it in config.SCORING.
"""
from .config import RAW_GAMELOG, SEASON


def main():
    from nba_api.stats.endpoints import leaguegamelog  # imported here so tests need no nba_api

    log = leaguegamelog.LeagueGameLog(
        season=SEASON, player_or_team_abbreviation="P"
    ).get_data_frames()[0]
    RAW_GAMELOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(RAW_GAMELOG, index=False)
    print(f"Saved {len(log)} player-games to {RAW_GAMELOG}")


if __name__ == "__main__":
    main()
