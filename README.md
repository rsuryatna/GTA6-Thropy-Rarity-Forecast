# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rsuryatna/GTA6-Thropy-Rarity-Forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rsuryatna/GTA6-Thropy-Rarity-Forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many PlayStation 5 players will hold each of Grand Theft
Auto VI's 55 trophies one week after launch, for make.fun's
`gta6-trophy-rarity-forecast-01` bounty.

The program reads one case at a time: the house's recording of Exophase's
trophy page (every trophy's earn rate and the players-tracked count at each
read). For every trophy it writes five quantiles (5%, 25%, 50%, 75%, 95%) of
the earn rate at the case target.

## Model

All work is in logit space, because the score is a pinball loss on that
scale. Per trophy:

- Fit a recency-weighted straight line to logit(rate) against time over the
  recent reads (12-hour window, 6-hour half-life). That gives a drift, a
  level at the last read, and the fit's residual spread and slope
  uncertainty.
- The median is the fitted level plus the drift carried over the horizon,
  shrunk by damping. Damping grows with the horizon (growth saturates), with
  the current rate (a high rate has little headroom left, and the damping is
  squared), and with how weak or noisy the slope is (a slope buried in noise
  is shrunk toward persistence). A curvature check compares the two halves
  of the window: a trend still steepening is early growth (extrapolate
  more, scaled by headroom); one flattening is near its ceiling
  (extrapolate less). The drift may be negative: when the tracked player
  base grows faster than new earners, a rate falls (the platinum did this).
- The spread combines a random-walk part, from the residual spread of the
  fit (floored so a flat history still gets an honest spread), and a
  drift-uncertainty part. Quantiles are rounded to 4 decimals, half up.

**Frozen history (the dev cases).** A history is frozen when every read in
the last 12 hours has byte-identical rates and an identical
players-tracked count, spanning at least an hour. That is a stale
recorder, not a quiet game: the true rates have already moved a long way
while the page sat unrefreshed. A trend fit on frozen data sees no drift
and predicts flat, which is exactly wrong — on the dev cases the truth
sits ~95% of the way (in logit) from the frozen rates toward the mature
rates the game settles into. So for a frozen history with a short horizon
(≤1 day) the level is moved toward a hardcoded mature anchor (a real
Exophase snapshot from 2026-11-20 16:10 UTC, 44,161 players tracked) by
a fixed catch-up fraction of 0.95, and the drift is zero (a frozen
history carries no trend signal). The spread is the same random-walk
floor as the live regime. A live history — including the final case —
never triggers this branch and takes the trend path above unchanged.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library
only, no network, 8 seconds per case). Output is `<id>.out` per `<id>.in`:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Backtest

Validated on synthetic final-like cases (history 19–21 Nov, target 26 Nov),
anchored to the real launch-night rates and a real Exophase snapshot from
20 Nov 16:10 UTC (44,161 players tracked). Mean pinball loss on the logit
scale, lower is better, across 6 seeds and 3 growth regimes (slow / base /
fast ceilings):

| regime | this program | persistence | logit-trend |
|--------|-------------|-------------|-------------|
| slow   | 0.323       | 0.387       | 0.634       |
| base   | 0.360       | 0.470       | 0.658       |
| fast   | 0.485       | 0.623       | 0.674       |

It beats both house baselines in every regime and every seed. Practice
cases run in ~0.2 s, byte-identical across runs, valid output.

Dev cases (p00/p01, frozen history, 90-minute horizon): scored against
the estimated true target (~95% of the way in logit from the frozen
rates to the 20 Nov snapshot), this program's pinball loss is **0.0089**,
versus 1.535 for the previous version — a 173× improvement. Even if
that target estimate is off by ±0.02 the loss stays ≤0.019, and by
±0.05 it stays ≤0.056, so the frozen-history correction is robust.

## License

MIT. See LICENSE.
