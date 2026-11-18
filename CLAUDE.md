# Project context for Claude

INDENG 174 (UC Berkeley, Simulation for Enterprise-Scale Systems) solo course
project: **Fantasy Basketball Roster Decisions Using Monte Carlo Simulation.**
Question: does choosing a weekly lineup by simulated win probability differ from
choosing by expected fantasy points?

Deadlines (11 PM PT): draft report and presentation Nov 30, 2026; second
presentation day Dec 2; final report + 10-minute video + code Dec 16, 2026.
The project is 50% of the course grade, and the final report and video are 75%
of that. Graders look for: reproducible code with run instructions, validation
against theory or data, confidence intervals, convergence evidence, common
random numbers, sensitivity analysis, and honest limitations.

## How to work with me (build mode)

I am no longer learning to write this code myself. You build it; I direct and
review. Work like a capable teammate:

1. Implement what I ask end to end, run it, and test it. Do not hand back untested code.
2. Keep each change small and reviewable. Show me the result (numbers, a table, a plot) and a 2-3 line summary of what changed and why.
3. Still explain the statistical idea behind any new method in a few sentences, because I must explain it in the report and the presentation.
4. Flag every modeling decision you make instead of silently choosing. List open decisions at the end of your reply. Defaults are marked `DECISION` in `fantasy_experiment/config.py`.
5. Never invent data or results. If something needs the internet or my data (NBA game logs, my rosters, my league's scoring), say so and tell me what to run.
6. Do not add features outside the scope below without asking.
7. Commit when I say so, with a clear message. Do not push unless asked.

## Scope

Simulation and decision analysis, not machine learning, not Yahoo API
integration, not a full fantasy app. Minimum viable product: fixed lineups in a
points league with an availability model and two fixed rosters. Stretch (only
after the core is validated): daily lineup slots and positions, waiver
optimization, correlated teammates, 9-category scoring.

## Code layout

- `simulation.py`, `toy_data.py`, `run_matchup.py`, `convergence_plot.py`: toy-data Stages 1-4 (validated against theory). Keep working.
- `fantasy_experiment/`: the real experiment.
  - `config.py` settings and DECISIONs; `scoring.py` fantasy points
  - `download_data.py` nba_api game logs (run locally; needs internet)
  - `data_prep.py` per-player mean/std/availability and games in a week
  - `model.py` weekly simulation with availability (Bernoulli games, normal scores)
  - `lineups.py` enumerate lineups, win probability, paired difference CIs
  - `draft.py` seeded 6-team snake draft -> `data/rosters.csv`, `outputs/draft_order.csv`
  - `run_exp1.py` Experiment 1 (main matchup = closest expected totals); `run_exp1_pairs.py` all 15 pairs
  - `demo.py` toy rosters (`--toy`)
  - `tests/` run: `python -m unittest discover -s fantasy_experiment/tests -t .`
- `data/` (raw data git-ignored), `outputs/` (figures and tables for the report), `notebooks/` (exploration only).

## Conventions

- Randomness: create one `np.random.default_rng(seed)` in the script and pass it in. Never hardcode a seed inside a function. Simulate all players in one call from one generator so they are independent. Default seed 42.
- Common random numbers: simulate the whole roster once, then score every lineup from the same scenario table. Compare lineups with `paired_win_diff`, not two independent estimates.
- Report every win probability with a 95% CI and the number of replications.
- Validate every model change against theory or data before building on it, and add a test.
- Avoid scipy unless needed (`math.erf` is used for the normal CDF).
- Keep reusable logic in modules; scripts only orchestrate. Update the README status checklist when a stage finishes. Update the Status section below too.

## Status (update as you go)

Done: toy Stages 1-4 validated; convergence plot; `fantasy_experiment` scaffold with
availability model, lineup enumeration, paired CIs, Experiment 1 script, tests (8 pass),
verified on demo data and a synthetic game log. 2025-26 game log downloaded (nba_api,
with retry). Rosters come from a 6-team seeded snake draft (`draft.py`, noise SD 2.0, seed 42).
Experiment 1 was run on Jan 12-18, 2026 for the main matchup (T1 vs T3, chosen before
simulating by the closest-expected-totals rule) and for all 15 pairs. Scoring is the
placeholder points formula (the user's real league is 9-cat; documented as an assumption).
15 tests pass.

Next, in order:
1. Experiment 2 (waiver add), Experiment 3 (sensitivity to availability and variance), convergence plots on real data.
2. Backtest on weeks not used for estimation; write limitations; draft report by Nov 30.

## Known limitations to document

Availability = games played / team games (a proxy, biased for traded players, not an injury forecast). Normal scores ignore skew and correlation between teammates. Positions and daily lineup slots are ignored. Lineups are fixed for the week. Opponent lineup is fixed.
