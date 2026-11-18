"""Settings for the fantasy basketball experiment.

Everything marked DECISION is a modeling choice that is not final. Change it
here and the rest of the code follows. Document the final choice in the report.
"""
from pathlib import Path

SEED = 42
N_SIM = 20_000          # replications per experiment (about +/-0.7 pts of win prob)
N_STARTERS = 5          # DECISION: players in a lineup (positions are ignored for now)

SEASON = "2025-26"
RAW_GAMELOG = Path("data/raw") / f"gamelog_{SEASON}.csv"
ROSTERS_CSV = Path("data/rosters.csv")   # columns: Team (X or Y), Player [, Starter]
OUTPUT_DIR = Path("outputs")

# DECISION (placeholder): fantasy points per stat. Replace with your league's
# real scoring. Keys must be column names in the nba_api LeagueGameLog output.
SCORING = {
    "PTS": 1.0,
    "REB": 1.2,
    "AST": 1.5,
    "STL": 3.0,
    "BLK": 3.0,
    "TOV": -1.0,
    "FG3M": 0.5,
}

# DECISION: clip simulated single-game scores at 0 (normal scores can go negative).
CLIP_AT_ZERO = True
