"""Fixed hypothetical rosters so the experiment runs without real data (--toy)."""
import numpy as np
import pandas as pd


def toy_players():
    """Return (players DataFrame, rosters DataFrame) for two 8-player teams."""
    rng = np.random.default_rng(0)   # fixed: the demo data never changes
    rows, ros = [], []
    for team in ("X", "Y"):
        for i in range(1, 9):
            name = f"{team}{i}"
            mean = float(rng.uniform(22, 48))
            rows.append({
                "Player": name,
                "Mean_PPG": mean,
                "Std_PPG": mean * float(rng.uniform(0.2, 0.45)),
                "Avail": float(rng.uniform(0.75, 1.0)),
                "Games": int(rng.choice([3, 4])),
            })
            ros.append({"Team": team, "Player": name})
    return pd.DataFrame(rows), pd.DataFrame(ros)
