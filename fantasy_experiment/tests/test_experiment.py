"""Sanity tests. Run from the repo root:
    python -m unittest discover -s fantasy_experiment/tests -t .
"""
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_experiment import config
from fantasy_experiment.data_prep import attach_schedule, load_gamelog, player_estimates
from fantasy_experiment.demo import toy_players
from fantasy_experiment.lineups import compare_lineups, lineup_total, paired_win_diff
from fantasy_experiment.model import simulate_weekly_totals
from fantasy_experiment.run_exp1 import parse_args, run


def synthetic_gamelog(path):
    """Two teams, 3 players each, 40 game days. Player 'Ghost' misses half his games."""
    rng = np.random.default_rng(1)
    rows = []
    for day in range(40):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)
        for team_id, names in ((1, ["A1", "A2", "Ghost"]), (2, ["B1", "B2", "B3"])):
            for name in names:
                if name == "Ghost" and day % 2 == 0:
                    continue
                rows.append({
                    "PLAYER_NAME": name, "TEAM_ID": team_id,
                    "GAME_ID": f"{team_id}-{day}", "GAME_DATE": date,
                    "PTS": rng.normal(20, 5), "REB": 5, "AST": 4, "STL": 1,
                    "BLK": 1, "TOV": 2, "FG3M": 2,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


class TestModel(unittest.TestCase):
    def test_single_player_matches_theory(self):
        df = pd.DataFrame({"Player": ["P"], "Mean_PPG": [25.0], "Std_PPG": [6.0],
                           "Avail": [1.0], "Games": [4]})
        t = simulate_weekly_totals(df, 200_000, np.random.default_rng(0), clip_at_zero=False)["P"]
        self.assertAlmostEqual(t.mean(), 100.0, delta=0.2)
        self.assertAlmostEqual(t.std(ddof=1), 12.0, delta=0.15)

    def test_availability_scales_mean(self):
        df = pd.DataFrame({"Player": ["P"], "Mean_PPG": [25.0], "Std_PPG": [0.0],
                           "Avail": [0.5], "Games": [4]})
        t = simulate_weekly_totals(df, 100_000, np.random.default_rng(0), clip_at_zero=False)["P"]
        self.assertAlmostEqual(t.mean(), 50.0, delta=0.3)

    def test_zero_games(self):
        df = pd.DataFrame({"Player": ["P"], "Mean_PPG": [25.0], "Std_PPG": [6.0],
                           "Avail": [1.0], "Games": [0]})
        t = simulate_weekly_totals(df, 10, np.random.default_rng(0))["P"]
        self.assertTrue((t == 0).all())


class TestLineups(unittest.TestCase):
    def setUp(self):
        self.players, self.rosters = toy_players()
        self.totals = simulate_weekly_totals(self.players, 5_000, np.random.default_rng(0))
        self.x = list(self.rosters.loc[self.rosters.Team == "X", "Player"])
        self.y = list(self.rosters.loc[self.rosters.Team == "Y", "Player"])

    def test_number_of_lineups(self):
        opp = lineup_total(self.totals, self.y[:5])
        table = compare_lineups(self.totals, self.x, 5, opp)
        self.assertEqual(len(table), math.comb(8, 5))

    def test_paired_diff_of_same_lineup_is_zero(self):
        opp = lineup_total(self.totals, self.y[:5])
        diff, (lo, hi) = paired_win_diff(self.totals, self.x[:5], self.x[:5], opp)
        self.assertEqual(diff, 0.0)
        self.assertEqual((lo, hi), (0.0, 0.0))

    def test_same_scenarios_for_every_lineup(self):
        # Two lineups sharing 4 of 5 players must have highly correlated totals (CRN)
        a = lineup_total(self.totals, self.x[:5])
        b = lineup_total(self.totals, self.x[:4] + [self.x[5]])
        self.assertGreater(np.corrcoef(a, b)[0, 1], 0.5)


class TestDataPrep(unittest.TestCase):
    def test_estimates_and_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "log.csv"
            synthetic_gamelog(path)
            gl = load_gamelog(path)
            est = player_estimates(gl, cutoff="2026-01-31")
            ghost = est.loc[est.Player == "Ghost"].iloc[0]
            self.assertAlmostEqual(ghost["Avail"], 0.5, delta=0.05)
            self.assertAlmostEqual(est.loc[est.Player == "A1"].iloc[0]["Avail"], 1.0)
            est = attach_schedule(est, gl, "2026-02-02", "2026-02-08")
            self.assertTrue((est["Games"] == 7).all())     # 7 days, a game each day

    def test_end_to_end_real_mode(self):
        with tempfile.TemporaryDirectory() as d:
            log, ros = Path(d) / "log.csv", Path(d) / "rosters.csv"
            synthetic_gamelog(log)
            pd.DataFrame({"Team": ["X"] * 3 + ["Y"] * 3,
                          "Player": ["A1", "A2", "Ghost", "B1", "B2", "B3"]}).to_csv(ros, index=False)
            old = config.OUTPUT_DIR
            config.OUTPUT_DIR = Path(d) / "out"
            try:
                args = parse_args(["--gamelog", str(log), "--rosters", str(ros),
                                   "--week-start", "2026-02-02", "--week-end", "2026-02-08",
                                   "--n", "2000", "--starters", "2"])
                table = run(args)
            finally:
                config.OUTPUT_DIR = old
            self.assertEqual(len(table), 3)     # C(3,2) lineups


if __name__ == "__main__":
    unittest.main()
