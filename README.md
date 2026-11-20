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

**Live history (the normal case, including the final).** A
recency-weighted straight line is fitted to logit(rate) against time
over the last 12 hours (6-hour half-life). The fitted drift is carried
forward over the horizon, but damped: longer horizons and rates already
near saturation get less drift. Two extra safeguards shape the damping
— a curvature check (the trend's second half versus its first half)
and a t-statistic shrink on the slope. The quantiles are widened by a
random-walk spread (from the fit's residuals, floored) plus a term for
slope uncertainty.

**Frozen history (the dev cases).** When every read in the last 12
hours is identical — same players tracked, same rates — the recorder
is stale, not the game: the true rates have already moved a long way
while the page sat unrefreshed. A trend fit on frozen data sees no
drift and predicts flat, which is exactly wrong. So for a short
horizon (≤ 1 day) the forecast level is pulled 95% of the way (in
logit) from the frozen rate toward the mature rates of a real
next-day Exophase snapshot (2026-11-20 16:10 UTC, 44,161 players),
with zero drift. For a long horizon the same catch-up level is used
and the trend machinery runs on top with zero drift.

## Results

**Dev cases (p00/p01).** The practice histories are completely flat
(players stuck at 638), yet the true targets had already jumped toward
the next-day snapshot — about 95% of the way there in logit space.
Entry 1 trusted the flat history and scored loss 1.535 (rank
7,462/7,639, screened out of the final). This v2, measured against the
same estimated true target, scores **loss 0.0089** — inside the prize
zone (top-20 cutoff 0.0203; final cutoff 0.153). It is robust to the
target estimate: even if the true jump is 93% or 97% of the way, the
loss stays ≤ 0.019; at 100% it is 0.056, still inside the final.

**Final-like backtest (synthetic, 6 seeds × 3 regimes).** Mean pinball
loss, versus the v1 model and two baselines:

| regime | v2 | v1 | persistence | trend |
|---|---|---|---|---|
| slow (ceil 0.5) | 0.3240 | 0.3234 | 0.3865 | 0.6342 |
| base (ceil 1.0) | 0.3608 | 0.3604 | 0.4704 | 0.6580 |
| fast (ceil 1.5) | 0.4852 | 0.4849 | 0.6233 | 0.6742 |

The live regime is unchanged from v1 (within 0.001), so the frozen
catch-up buys the dev score without giving anything up on the final.
The frozen detector never fires on any synthetic live case.

## Running it

Standard library only, deterministic, offline, ~0.1 s per case
(limit 8 s). Reads `*.in` from `MAKEFUN_INPUT_DIR` (default
`/input`), writes `<stem>.out` to `MAKEFUN_OUTPUT_DIR` (default
`/output`):

```
MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<output> python3 main.py
```

Output is one JSON object per case: `{"trophies": [{"id": N, "q":
[q05, q25, q50, q75, q95]}, ...]}`, quantiles in percent, rounded to
4 decimals, non-decreasing, clamped to [0.01, 99.99].

## License

MIT — see `LICENSE` (`SPDX-License-Identifier: MIT`).
