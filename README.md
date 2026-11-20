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
python -m fantasy_experiment.run_exp1_leagues --week-start 2026-01-12 --week-end 2026-01-18    # 50 leagues, k=10 and 5 (35-75 min, depends on machine load)
python -m fantasy_experiment.exp1_balance       # breakdown by matchup balance: outputs/exp1_balance.{csv,png} (seconds; reads the 50-league output)
# any script takes --starters K (default 10); run_exp1_leagues takes several, e.g. --starters 10 5

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

## Model and experiment design

### Data and scoring

- **Data:** 2025-26 regular-season player game logs from nba_api (26,651
  player-games, 1,230 games). Estimates use only games before the cutoff
  (Jan 12, 2026), so nothing from the experiment week leaks in.
- **Scoring is a placeholder points formula, not a real league's.** Fantasy
  points per game = 1.0 PTS + 1.2 REB + 1.5 AST + 3 STL + 3 BLK - 1 TOV
  + 0.5 3PM (`config.SCORING`). The author's real Yahoo league is 9-category,
  which this project does not model (a stretch goal). The NBA's own
  `FANTASY_PTS` column from nba_api is ignored (kept as `NBA_FANTASY_PTS`).
- **Zero-minute rows are not games played.** The game log has 83 rows with
  0 minutes (48 before the cutoff). They are dropped from games played,
  availability, and the mean/SD of fantasy points (`config.EXCLUDE_ZERO_MIN`),
  but still count toward the team's schedule.

### Simulation model

- A scheduled game is played with probability Avail (games played / team
  games); if played, the fantasy score is Normal(mean, SD), clipped at 0.
- Games and players are independent; no teammate correlation.
- **Experiment 1 week:** Mon Jan 12 to Sun Jan 18, 2026. Games in the week =
  games the player's team actually played that week (from the log).

### Rosters and lineups

- **Rosters come from a simulated 6-team league draft** (`draft.py`). The pool
  is every player with at least 25 games before the cutoff: 298 players. A
  player's value is his average fantasy points per game played before the
  cutoff. The draft noise is **one 6 x pool-size array of Normal(0, 2.0)
  draws** from the league's generator (`default_rng(42)` in `draft.py`; a
  spawned child generator per league in the multi-league run). Row t is team
  t's noise, so each team's board is value plus its row. The draft is a snake
  (T1..T6, T6..T1, ...) for 13 rounds: 78 players, 13 per team. Each pick takes
  the available player ranked highest on that team's board. Team names are
  draft slots (T1 picks first). Every pick of the single league is recorded in
  `outputs/draft_order.csv`. Names come straight from the game log.
- **Starters:** 10 of 13 rostered players start and 3 sit on the bench
  (`config.N_STARTERS`, `--starters`). That gives C(13,10) = 286 possible
  lineups per decision at k = 10, and C(13,5) = 1,287 at k = 5.
- **Opponent lineup:** fixed for the week. `Starter` is left blank in
  `data/rosters.csv`, so the opponent starts its top k players by expected
  weekly points (Mean x Avail x Games).
- **Points pick** = the lineup with the highest theoretical expected points
  (no Monte Carlo noise). **Win-probability pick** = the lineup with the
  highest simulated P(win).

### Experiment 1

- **Main matchup rule (fixed before any simulation was run):** the two teams
  whose expected weekly totals for Jan 12-18 are closest. A team's expected
  total is the sum of Mean x Avail x Games over its best k players by that
  quantity, computed from the formula, not from simulation. The team earlier
  in name order chooses its lineup; the other team's lineup is fixed. At
  k = 10 the rule picks T2 vs T4 (1,329.1 vs 1,331.4). The next closest pair,
  T5 vs T6, differs by 2.9, so the choice is close.
- **All-pairs check:** each of the 15 pairs is run in both directions (30
  ordered pairs, i.e. 30 decisions). Every pair uses the same simulated weeks.
- **Selection run vs evaluation run.** Each experiment draws two independent
  runs of n = 200,000 simulated weeks. The selection run ranks all lineups and
  picks the one with the highest P(win). The evaluation run re-scores the two
  picks; its paired difference (`Holdout_Gain`) is the reported gain, and it is
  exactly 0 when the picks agree. Taking the maximum of 286 (or 1,287) noisy
  estimates and reporting that same estimate is biased upward (winner's
  curse); re-scoring on independent data removes the bias. The in-sample
  difference is kept in the output only to show the bias.
- **Robustness (single league):** the 30-pair analysis is repeated for seeds
  42, 1, 2, 3 at n = 20,000 and n = 200,000 (`outputs/exp1_robustness*.csv`).
  A disagreement is treated as real if it appears for most seeds at
  n = 200,000 with a held-out CI above 0.
- **Multi-league Experiment 1 (headline result).** A single league's result
  depends on its rosters, so Experiment 1 is repeated over 50 independently
  drafted leagues (`run_exp1_leagues.py`). One `default_rng(42)` spawns an
  independent child generator per league. The child drives that league's draft
  and its selection and evaluation runs (n = 200,000 each). Every league is
  evaluated for k = 10 and k = 5 on the same draft and the same simulated
  weeks. Reported per k: the share of decisions where the picks differ, the
  mean held-out gain per decision (0 when the picks agree), and the mean
  held-out gain when they differ. The 95% CIs come from a bootstrap that
  resamples whole leagues (10,000 resamples), because the 30 decisions in a
  league share rosters and simulated weeks and are not independent.

### Experiment 1 results (50 leagues, Jan 12-18, 2026)

1,500 decisions per lineup size (50 leagues x 30 ordered pairs), n = 200,000
simulated weeks per run, seed 42. 95% CIs from the league bootstrap.
Source: `outputs/exp1_leagues_summary.csv`.

| Starters (k) | Lineups per decision | Share of decisions where picks differ | Mean held-out gain per decision | Mean held-out gain when picks differ |
|---|---|---|---|---|
| 10 | 286 | 9.5% (7.2%, 11.9%) | +0.017 pts (+0.012, +0.022) | +0.18 pts (+0.14, +0.22) |
| 5 | 1,287 | 9.0% (6.7%, 11.6%) | +0.031 pts (+0.019, +0.044) | +0.34 pts (+0.25, +0.43) |

"pts" = percentage points of win probability. Using simulated win probability
instead of expected points changes about 1 decision in 10. When it does, the
win probability rises by a fraction of a percentage point (largest single gain:
0.9 pts at k = 10, 2.1 pts at k = 5). Disagreements are rarest in even
matchups (2-3% of decisions when the points pick's P(win) is 0.4-0.6) and
most common in lopsided ones (10-19% below 0.2 or above 0.8), as expected:
variance matters most to clear underdogs and clear favourites. Breakdown and
chart: `outputs/exp1_balance.csv`, `outputs/exp1_balance.png`
(`exp1_balance.py`; bins of width 0.2 on the held-out P(win) of the points
pick, league-bootstrap CIs).

### Limitations

- **Positions and daily lineup changes are ignored.** Any 10 of the 13 players
  can start, regardless of position (real leagues require
  PG/SG/G/SF/PF/F/C slots). The lineup is set once for the whole week, while
  real leagues let managers change starters daily (for example, to start a
  bench player on a day a starter has no game). Both simplifications make the
  decision a pure "which 10 players" choice.
- Availability = games played / team games is a proxy, biased for traded
  players, and is not an injury forecast. Missed games are independent coin
  flips, while real absences come in streaks.
- Normal scores ignore skew and correlation between teammates.
- The opponent's lineup is fixed.
- A given league is reproducible only for a fixed pool of players: the draft
  noise array is sized by the pool, so adding or removing any eligible player
  reshuffles every team's board.

## Reproducibility

Every script creates one `np.random.default_rng(seed)` (default 42) and passes
it into the functions; the multi-league run spawns independent child
generators from it, one per league. Results are identical on every run.

## Status

- [x] Player simulation, validated against theory (mean, SD)
- [x] Team totals and matchup win probability (checked against exact formula)
- [x] Convergence check (half-width shrinks like 1/sqrt(n))
- [x] Lineup comparison with common random numbers (validated on toy data, used on real data)
- [x] Availability model (validated on synthetic data, used on real data)
- [x] Real NBA game logs (2025-26, 26,651 player-games, 1,230 games)
- [x] 6-team seeded snake draft from the game log; placeholder points scoring (documented)
- [x] Experiment 1 on one drafted league (Jan 12-18, 2026): main matchup, 30 ordered pairs, seed/n robustness
- [x] Experiment 1 across 50 drafted leagues, 10 and 5 starters, held-out gain with league-bootstrap CIs
- [ ] Experiments 2 and 3, sensitivity analysis
- [ ] Backtest on weeks not used for estimation
- [ ] Draft report (Nov 30), final report + video (Dec 16)
