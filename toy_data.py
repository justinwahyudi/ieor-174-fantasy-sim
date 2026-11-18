"""Hypothetical players used while developing the simulator (Stage 1).

Replace with real NBA game-log estimates later. Weekly theory:
Team X = 232 pts, SD 20.00. Team Y = 240 pts, SD about 8.4 (a close matchup
where X is a slight underdog, so variance matters).
"""
import pandas as pd

PLAYERS = pd.DataFrame({
    "Player":   ["Player A", "Player B", "Player C", "Player D", "Player E", "Player F"],
    "Mean_PPG": [25, 20, 18, 22, 21, 22.25],
    "Std_PPG":  [6, 8, 4, 3, 3, 1.3],
    "Games":    [4, 3, 4, 4, 3, 4],
})

TEAM_X = ["Player A", "Player B", "Player C"]
TEAM_Y = ["Player D", "Player E", "Player F"]
