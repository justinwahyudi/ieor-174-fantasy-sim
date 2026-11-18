"""Turn raw box-score columns into fantasy points."""
from .config import SCORING


def add_fantasy_points(df, weights=None):
    """Return a copy of df with a FANTASY_PTS column (sum of weight * stat)."""
    weights = weights or SCORING
    missing = [c for c in weights if c not in df.columns]
    if missing:
        raise KeyError(f"Game log is missing scoring columns: {missing}")
    out = df.copy()
    out["FANTASY_PTS"] = sum(out[col] * w for col, w in weights.items())
    return out
