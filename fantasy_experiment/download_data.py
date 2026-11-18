"""Download the season's player game logs with nba_api.

Run:  python -m fantasy_experiment.download_data

Needs internet and `pip install nba_api`. Saves data/raw/gamelog_<season>.csv
(git-ignored). stats.nba.com often times out, so each request is retried with
a growing wait and a longer timeout.
"""
import time

from .config import RAW_GAMELOG, SEASON

MAX_TRIES = 5
BASE_TIMEOUT = 60   # seconds; grows with each retry
BASE_WAIT = 5       # seconds between tries; doubles each retry


def fetch_gamelog(season=SEASON, max_tries=MAX_TRIES):
    from nba_api.stats.endpoints import leaguegamelog  # imported here so tests need no nba_api

    for attempt in range(1, max_tries + 1):
        timeout = BASE_TIMEOUT * attempt
        try:
            return leaguegamelog.LeagueGameLog(
                season=season, player_or_team_abbreviation="P", timeout=timeout
            ).get_data_frames()[0]
        except Exception as exc:  # requests timeouts / connection resets
            if attempt == max_tries:
                raise
            wait = BASE_WAIT * 2 ** (attempt - 1)
            print(f"Attempt {attempt}/{max_tries} failed ({type(exc).__name__}: {exc}); "
                  f"retrying in {wait}s with timeout {BASE_TIMEOUT * (attempt + 1)}s")
            time.sleep(wait)


def main():
    log = fetch_gamelog()
    RAW_GAMELOG.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(RAW_GAMELOG, index=False)
    print(f"Saved {len(log)} player-games to {RAW_GAMELOG}")


if __name__ == "__main__":
    main()
