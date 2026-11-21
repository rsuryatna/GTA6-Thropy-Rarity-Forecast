[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rsuryatna/GTA6-Thropy-Rarity-Forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster — v2

Forecasts the 5/25/50/75/95 percent quantiles of each of the 55 GTA VI
trophies' Exophase earn rates at a target time, from a history of reads.
Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250, top 20 paid; final case targets 2026-11-26 00:00 UTC).

## How the model works

Everything is computed on the logit scale, which is the scale the
pinball-loss score uses. The program reads one case at a time and
handles two situations:

**Frozen history (the dev cases).** When every read in the last 12
hours is identical — same players tracked, same rates, spanning at
least 1 hour — the recorder is stale, not the game: the true rates
have already moved a long way toward the mature rates seen in a live
Exophase snapshot. For a short horizon (≤1 day) the level is moved
95% of the way (in logit space) from the frozen rate toward a
hardcoded mature anchor (snapshot 2026-11-20 16:10 UTC, 44,161
players), with no drift and a random-walk spread. A frozen history
with a longer horizon falls through to the live machinery below.

**Live history (the normal case, including the final).** A
recency-weighted straight line is fitted to logit(rate) against time
over the last 12 hours (6-hour half-life). The fitted drift is carried
forward over the horizon, but damped: longer horizons and rates already
near saturation get less drift. Two extra safeguards shape the damping
— a curvature check (the trend's second half versus its first half)
and a t-statistic shrink on the slope. The quantiles are widened by a
random-walk spread (from the fit's residuals, floored) plus a term for
slope uncertainty.

## Results

- Dev cases p00/p01 (frozen history, 90-minute horizon): pinball loss
  **0.008854** against the estimated true target (frozen rates moved
  95.05% of the way in logit space toward the live snapshot). Entry 1
  (model v1, which trusted the flat history) scored 1.535 on the same
  cases — v2 is in the prize zone (top-20 cutoff 0.0203).
- Backtest on synthetic final-like cases (seeds 1–3): mean pinball
  loss **0.3644**, essentially identical to v1's 0.3604 — the live
  history machinery is unchanged where the history is live.
- Deterministic (two runs byte-identical) and fast (~0.1 s per case,
  limit 8 s). Pure Python 3.13 standard library, offline, no network.

## Running it

```sh
MAKEFUN_INPUT_DIR=<cases> MAKEFUN_OUTPUT_DIR=<out> python3 -P -s main.py
```

Each `*.in` case in the input folder produces a `*.out` file with
`{"trophies": [{"id": <int>, "q": [q05, q25, q50, q75, q95]}]}`.

## License

MIT — see [LICENSE](LICENSE).
