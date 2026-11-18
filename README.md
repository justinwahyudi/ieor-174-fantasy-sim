# Fantasy Basketball Roster Decisions with Monte Carlo Simulation

INDENG 174 course project. Question: does choosing a weekly lineup by simulated
win probability give different decisions than choosing by expected points?

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Toy-data stages (validated against theory):

```
python run_matchup.py            # Team X vs Team Y: theory vs simulation
python convergence_plot.py       # saves outputs/convergence.png
```

Fantasy experiment (`fantasy_experiment/`):

```
python -m fantasy_experiment.run_exp1 --toy      # demo, no data needed

# real data (needs internet once; retries if stats.nba.com times out):
python -m fantasy_experiment.download_data       # saves data/raw/gamelog_2025-26.csv
python -m fantasy_experiment.draft --cutoff 2026-01-12     # data/rosters.csv, outputs/draft_order.csv
python -m fantasy_experiment.run_exp1 --week-start 2026-01-12 --week-end 2026-01-18        # main matchup
python -m fantasy_experiment.run_exp1_pairs --week-start 2026-01-12 --week-end 2026-01-18  # all 30 ordered pairs
python -m fantasy_experiment.run_exp1_robustness --week-start 2026-01-12 --week-end 2026-01-18  # seeds x n (about 2 min)

# tests
python -m unittest discover -s fantasy_experiment/tests -t .
```

Expected output of `run_matchup.py` (seed 42): Team X mean 232 / SD 20.00,
Team Y mean 240 / SD about 8.4, theory P(X wins) about 0.356, with the
simulated estimate inside its 95% CI of that value.

## Layout

| Path | Purpose |
|---|---|
| `simulation.py` | Toy-stage functions: player and team simulation, theory, win probability |
| `toy_data.py`, `run_matchup.py`, `convergence_plot.py` | Toy Stages 1-4 and the convergence check |
| `fantasy_experiment/` | Real experiment: data prep, availability model, lineups, Experiment 1 (see its files' docstrings) |
| `notebooks/` | Exploration only; keep real logic in `.py` files |
| `data/` | Small cleaned data; large raw data is git-ignored |
| `outputs/` | Figures and tables for the report |
| `CLAUDE.md` | Project context and working rules for Claude Code |

## Model assumptions

- **Scoring is a placeholder points formula, not a real league's.** Fantasy
  points per game = 1.0 PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - 1 TOV
  + 0.5 3PM (`config.SCORING`). The author's real Yahoo league is 9-category, which
  this project does not model (9-cat is a stretch goal). The NBA's own
  `FANTASY_PTS` column from nba_api is ignored (kept as `NBA_FANTASY_PTS`).
- **Rosters come from a simulated 6-team league draft** (`fantasy_experiment/draft.py`).
  The pool is every player with at least 25 games before the cutoff (Jan 12, 2026):
  302 players. A player's value is his average fantasy points per game played
  before the cutoff. Each team gets its own draft board, equal to value plus
  Normal(0, 2.0) noise, drawn once per (team, player) from `default_rng(42)`.
  The draft is a snake (T1..T6, T6..T1, ...) for 13 rounds: 78 players, 13 per
  team. Each pick takes the available player ranked highest on that team's
  board. Team names are draft slots (T1 picks first). Every pick is recorded in
  `outputs/draft_order.csv`. Names come straight from the game log.
  `Starter` is left blank, so an opponent's lineup is its top 5 by expected weekly points.
- **Main matchup rule (fixed before any simulation was run):** the two teams whose
  expected weekly totals for Jan 12-18 are closest. A team's expected total is the
  sum of Mean x Avail x Games over its best 5 by that same quantity, computed from
  the formula, not from simulation. The team earlier in name order chooses its
  lineup; the other team's lineup is fixed. The rule picked T1 vs T3 (761.0 vs
  761.3; the next closest pair differs by 19.7).
- **All-pairs check:** Experiment 1 is also run for all 30 ordered pairs. Each of
  the 15 pairs is run in both directions (each team takes a turn choosing its lineup
  against the other's fixed lineup). All 78 players are simulated once, so every pair uses the same
  simulated weeks.
- **Points pick** = the lineup with the highest theoretical expected points (no
  Monte Carlo noise). **Win-probability pick** = the lineup with the highest
  simulated P(win). If the two match, the reported paired CI compares the pick
  with the runner-up.
- **Selection run vs evaluation run.** Each experiment draws two independent
  runs of n = 200,000 simulated weeks from one `default_rng(seed)`. The
  selection run ranks all lineups and picks the one with the highest P(win).
  The evaluation run re-scores the two picks; its paired difference
  (`Holdout_Gain`) is the reported gain. Taking the maximum of 1,287 noisy
  estimates and reporting that same estimate is biased upward (winner's curse);
  re-scoring on independent data removes the bias. The in-sample difference is
  kept in the output only to show the bias.
- **Robustness:** the 30-pair analysis is repeated for seeds 42, 1, 2, 3 at
  n = 20,000 and n = 200,000 (`outputs/exp1_robustness*.csv`). A
  disagreement is treated as real if it appears for most seeds at n = 200,000
  with a held-out CI above 0.
- **Experiment 1 week:** Mon Jan 12 to Sun Jan 18, 2026 (2025-26 season).
  Estimates use only games before Jan 12. Games in the week = games the
  player's team actually played that week (from the log).

- A scheduled game is played with probability Avail (games played / team games);
  if played, the fantasy score is Normal(mean, std), clipped at 0.
- Games and players are independent; no teammate correlation.
- Lineups are fixed for the week; positions and daily slots are ignored.
- Opponent lineup is fixed.

## Reproducibility

Every script creates `np.random.default_rng(seed)` (default 42) and passes it
into the functions. Results are identical on every run.

## Status

- [x] Player simulation, validated against theory (mean, SD)
- [x] Team totals and matchup win probability (checked against exact formula)
- [x] Convergence check (half-width shrinks like 1/sqrt(n))
- [x] Lineup comparison with common random numbers (demo and synthetic data only)
- [x] Availability model (tested on synthetic data)
- [x] Real NBA game logs (2025-26, 26,651 player-games, 1,230 games)
- [x] 6-team seeded snake draft from the game log; placeholder points scoring (documented)
- [x] Experiment 1 on a real week (Jan 12-18, 2026): main matchup and all 30 ordered pairs
- [ ] Experiments 2 and 3, sensitivity analysis
- [ ] Backtest, limitations, draft report (Nov 30), final report + video (Dec 16)
