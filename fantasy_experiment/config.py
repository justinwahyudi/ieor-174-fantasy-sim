"""Settings for the fantasy basketball experiment.

Everything marked DECISION is a modeling choice that is not final. Change it
here and the rest of the code follows. Document the final choice in the report.
"""
from pathlib import Path

SEED = 42
N_SIM = 200_000         # replications per run (about +/-0.22 pts of win prob at p=0.5)
ROBUSTNESS_SEEDS = (42, 1, 2, 3)        # seeds for the Experiment 1 robustness check
ROBUSTNESS_NS = (20_000, 200_000)       # replication counts for the robustness check
N_STARTERS = 10         # DECISION: starters per team; 13-man roster, 3 on the bench.
                        # Positions and daily lineup changes are ignored (limitation).
STARTERS_COMPARE = (10, 5)              # lineup sizes compared in the multi-league run
N_LEAGUES = 50          # DECISION: independently drafted leagues in the multi-league run

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

# DECISION: drop game-log rows with 0 minutes (logged but did not really play)
# from games played, availability, and the mean/SD of fantasy points.
EXCLUDE_ZERO_MIN = True

# DECISION: clip simulated single-game scores at 0 (normal scores can go negative).
CLIP_AT_ZERO = True
