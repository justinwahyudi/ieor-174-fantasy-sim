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


class TestDraft(unittest.TestCase):
    def setUp(self):
        from fantasy_experiment.draft import snake_draft
        rng = np.random.default_rng(42)
        n = 120
        self.est = pd.DataFrame({"Player": [f"P{i:03d}" for i in range(n)],
                                 "Mean_PPG": np.linspace(60, 20, n),
                                 "GP": [30] * (n - 10) + [3] * 10})   # last 10 ineligible
        self.draft = snake_draft(self.est, rng, n_teams=6, n_rounds=13)

    def test_snake_order(self):
        from fantasy_experiment.draft import snake_order
        order = snake_order(6, 3)
        self.assertEqual(order[:6], ["T1", "T2", "T3", "T4", "T5", "T6"])
        self.assertEqual(order[6:12], ["T6", "T5", "T4", "T3", "T2", "T1"])
        self.assertEqual(order[12:], ["T1", "T2", "T3", "T4", "T5", "T6"])
        self.assertEqual(list(self.draft["Team"]), snake_order(6, 13))

    def test_no_player_drafted_twice(self):
        self.assertEqual(len(self.draft), 78)
        self.assertEqual(self.draft["Player"].nunique(), 78)

    def test_each_roster_has_13(self):
        self.assertEqual(self.draft.groupby("Team").size().to_dict(),
                         {f"T{i}": 13 for i in range(1, 7)})

    def test_min_games_enforced(self):
        ineligible = set(self.est.loc[self.est["GP"] < 25, "Player"])
        self.assertFalse(ineligible & set(self.draft["Player"]))

    def test_seeded_and_noise_free_case(self):
        from fantasy_experiment.draft import snake_draft
        again = snake_draft(self.est, np.random.default_rng(42))
        pd.testing.assert_frame_equal(self.draft, again)        # same seed, same draft
        exact = snake_draft(self.est, np.random.default_rng(0), noise_sd=0.0)
        self.assertEqual(list(exact["Player"]), list(self.est["Player"][:78]))  # strict rank order


class TestMatchupSelection(unittest.TestCase):
    def test_closest_pair_uses_expected_totals(self):
        from fantasy_experiment.lineups import closest_pair
        players = pd.DataFrame({"Player": ["a", "b", "c", "d", "e", "f"],
                                "Mean_PPG": [50, 10, 40, 10, 41, 10],
                                "Avail": [1.0] * 6, "Games": [3] * 6})
        rosters = pd.DataFrame({"Team": ["T1", "T1", "T2", "T2", "T3", "T3"],
                                "Player": list("abcdef")})
        (x, y), totals = closest_pair(players, rosters, k=1)
        self.assertEqual((x, y), ("T2", "T3"))
        self.assertAlmostEqual(totals["T1"], 150.0)


class TestScoring(unittest.TestCase):
    def test_load_gamelog_overwrites_api_fantasy_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gl.csv"
            synthetic_gamelog(path)
            df = pd.read_csv(path)
            df["FANTASY_PTS"] = -999.0          # nba_api's own column
            df.to_csv(path, index=False)
            gl = load_gamelog(path)
            self.assertTrue((gl["NBA_FANTASY_PTS"] == -999.0).all())
            expected = sum(gl[c] * w for c, w in config.SCORING.items())
            np.testing.assert_allclose(gl["FANTASY_PTS"], expected)


class TestAllPairs(unittest.TestCase):
    def test_both_directions(self):
        from fantasy_experiment.run_exp1_pairs import run_all_pairs
        players, rosters = toy_players()
        totals = simulate_weekly_totals(players, 2000, np.random.default_rng(0))
        df = run_all_pairs(players, rosters, totals, 3)
        self.assertEqual(sorted(zip(df["Team_X"], df["Team_Y"])), [("X", "Y"), ("Y", "X")])


class TestHoldout(unittest.TestCase):
    def test_winners_curse_removed_by_holdout(self):
        """Identical players: every lineup is truly equal, so the true gain is 0.
        The in-sample gain of the best-looking lineup is positive by construction;
        the held-out gain should not be."""
        from fantasy_experiment.lineups import evaluate_matchup
        names = [f"P{i}" for i in range(10)]
        players = pd.DataFrame({"Player": names, "Mean_PPG": 40.0, "Std_PPG": 12.0,
                                "Avail": 0.9, "Games": 3})
        players.loc[:2, "Mean_PPG"] = 40.001   # points pick = P0, P1, P2 (true gain ~ 0)
        rng = np.random.default_rng(7)
        sel = simulate_weekly_totals(players, 20000, rng)
        ev = simulate_weekly_totals(players, 20000, rng)
        _, s = evaluate_matchup(sel, players, names[:8], names[8:], 3, ev)
        self.assertTrue(s["Differ"])
        self.assertGreater(s["Diff"], 0)
        self.assertLessEqual(s["Holdout_CI_low"], 0.0)
        self.assertGreaterEqual(s["Holdout_CI_high"], 0.0)

    def test_holdout_zero_when_picks_agree_and_matches_in_sample_on_same_run(self):
        from fantasy_experiment.lineups import evaluate_matchup
        players, rosters = toy_players()
        x = list(rosters.loc[rosters.Team == "X", "Player"])
        y = list(rosters.loc[rosters.Team == "Y", "Player"])[:3]
        totals = simulate_weekly_totals(players, 5000, np.random.default_rng(3))
        _, s = evaluate_matchup(totals, players, x, y, 3, totals_eval=totals)
        if s["Differ"]:
            self.assertAlmostEqual(s["Holdout_Gain"], s["Diff"])
        else:
            self.assertEqual(s["Holdout_Gain"], 0.0)
        self.assertAlmostEqual(s["Holdout_P_win_best"], s["P_win_best"])


class TestZeroMinutes(unittest.TestCase):
    def test_zero_minute_rows_not_games_played(self):
        from fantasy_experiment.data_prep import player_estimates
        dates = pd.date_range("2026-01-01", periods=10)
        gl = pd.DataFrame({"PLAYER_NAME": ["A"] * 10 + ["B"] * 10, "TEAM_ID": 1,
                           "GAME_ID": list(range(10)) * 2, "GAME_DATE": list(dates) * 2,
                           "MIN": [30] * 10 + [30] * 6 + [0] * 4,
                           "FANTASY_PTS": [40.0] * 10 + [40.0] * 6 + [0.0] * 4})
        est = player_estimates(gl, "2026-02-01").set_index("Player")
        self.assertEqual(est.loc["B", "GP"], 6)
        self.assertAlmostEqual(est.loc["B", "Avail"], 0.6)       # team still played 10
        self.assertAlmostEqual(est.loc["B", "Mean_PPG"], 40.0)
        kept = player_estimates(gl, "2026-02-01", exclude_zero_min=False).set_index("Player")
        self.assertEqual(kept.loc["B", "GP"], 10)
        self.assertAlmostEqual(kept.loc["B", "Mean_PPG"], 24.0)


class TestMultiLeague(unittest.TestCase):
    @staticmethod
    def fake_est(n=100):
        rng = np.random.default_rng(5)
        return pd.DataFrame({"Player": [f"P{i:03d}" for i in range(n)],
                             "Mean_PPG": np.linspace(55, 20, n), "Std_PPG": 10.0,
                             "GP": 30, "Avail": rng.uniform(0.7, 1.0, n),
                             "Games": rng.integers(2, 5, n)})

    def test_leagues_differ_and_are_reproducible(self):
        from fantasy_experiment.multileague import run_leagues
        est = self.fake_est()
        a = run_leagues(est, np.random.default_rng(1), 2, (10, 5), 500)
        b = run_leagues(est, np.random.default_rng(1), 2, (10, 5), 500)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(len(a), 2 * 2 * 30)                      # leagues x k x ordered pairs
        l1 = a[(a.League == 1) & (a.Starters == 5)].Best_Pts.tolist()
        l2 = a[(a.League == 2) & (a.Starters == 5)].Best_Pts.tolist()
        self.assertNotEqual(l1, l2)                               # different drafts

    def test_bootstrap_summary_consistent(self):
        from fantasy_experiment.multileague import bootstrap_summary, run_leagues
        detail = run_leagues(self.fake_est(), np.random.default_rng(2), 3, (10, 5), 500)
        summ, lt = bootstrap_summary(detail, np.random.default_rng(0), n_boot=500)
        for r in summ.itertuples():
            sub = detail[detail.Starters == r.Starters]
            self.assertAlmostEqual(r.Share_Differ, sub.Differ.mean())
            self.assertAlmostEqual(r.Gain_All, sub.Holdout_Gain.mean())
            self.assertLessEqual(r.Share_Differ_CI_low, r.Share_Differ)
            self.assertGreaterEqual(r.Share_Differ_CI_high, r.Share_Differ)
