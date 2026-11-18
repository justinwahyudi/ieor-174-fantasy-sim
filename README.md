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

# real data (needs internet once):
python -m fantasy_experiment.download_data       # saves data/raw/gamelog_<season>.csv
# create data/rosters.csv with columns Team (X or Y), Player [, Starter]
python -m fantasy_experiment.run_exp1 --week-start 2026-01-12 --week-end 2026-01-18

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
- [ ] Real NBA game logs, real rosters, league scoring formula
- [ ] Experiment 1 on a real week; Experiments 2 and 3, sensitivity analysis
- [ ] Backtest, limitations, draft report (Nov 30), final report + video (Dec 16)
