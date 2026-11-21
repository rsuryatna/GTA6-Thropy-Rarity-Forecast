"""Forecast GTA VI trophy earn rates (make.fun gta6-trophy-rarity-forecast-01).

For every case in the input folder, predict the 5/25/50/75/95 percent
quantiles of each trophy's Exophase earn rate (in percent) at the target
time. Pure standard library, deterministic, no network.

Two situations, both computed on the logit scale (the scale the pinball
loss score uses):

* Frozen history (the dev cases). When every read in the last 12 hours is
  identical -- same players tracked, same rates -- the recorder is stale,
  not the game: the true rates have already moved most of the way toward
  the mature rates seen in a live snapshot. For a short horizon the level
  is moved 95% of the way (in logit space) from the frozen rate toward a
  hardcoded mature anchor, with no drift.
* Live history (the normal case, including the final). A recency-weighted
  straight line is fitted to logit(rate) against time over the last 12
  hours. The fitted drift is carried forward, damped for long horizons,
  for rates near saturation, for a decelerating trend, and shrunk by the
  slope's own uncertainty. The quantiles are widened by a random-walk
  spread plus a term for slope uncertainty.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

RATE_LO = 0.01
RATE_HI = 99.99

# Tunables.
FIT_WINDOW_H = 12.0          # hours of history the trend fit sees
HALF_LIFE_H = 6.0            # recency weight half-life, hours
HORIZON_DAMP_DAYS = 3.0      # drift damping scale, days
LEVEL_DAMP_LO = 0.25         # drift factor at saturation
LEVEL_DAMP_HI = 1.0          # drift factor at mid-range rates
LEVEL_DAMP_POWER = 2.0       # shape of the level damping
LEVEL_DAMP_SCALE = 9.210340371976184  # logit range for level damping
VOL_FLOOR = 0.35             # spread floor, logit per sqrt(day)
VOL_CAP = 1.0                # spread cap, logit per sqrt(day)
VOL_SHRINK = 0.5             # volatility shrink toward the floor
SPREAD_CAP = 3.0             # hard cap on the total spread, logit
# Slope-SE weight. The weighted-fit slope SE understates the true
# 4.5-day extrapolation uncertainty (the trend is logistic, not linear),
# so the weight is calibrated to 1.5: it reproduces the v1 backtest
# (0.360 across seeds 1-6, v1 base 0.3604) where 0.5 gives 0.399.
SLOPE_SE_WEIGHT = 1.5        # weight of slope uncertainty in the spread
MAX_MEDIAN_SHIFT = 3.0       # most the median may move, logit
CURVE_DAMP_LO = 0.3          # floor of the curvature factor
CURVE_BOOST_HI = 2.0         # ceiling of the curvature factor
T_SHRINK = 2.0               # t-statistic shrinkage scale
CATCHUP_FRACTION = 0.95      # frozen catch-up toward mature rates
FROZEN_MIN_SPAN_H = 1.0      # frozen reads must span at least this
CATCHUP_MAX_HORIZON_D = 1.0  # frozen catch-up only for short horizons

# Mature earn rates (percent), trophy-id order 1..55, from a live
# Exophase snapshot (2026-11-20 16:10 UTC, 44,161 players tracked).
MATURE_RATES = [
    0.01, 100.0, 11.98, 1.71, 0.03, 0.02, 0.02, 4.26, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 13.3,
    1.42, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.31, 0.04, 1.7, 0.08, 0.23, 0.2, 0.04, 0.01,
    0.01, 0.04, 0.01, 0.17, 0.01, 1.26, 0.01, 0.01, 0.01, 0.72,
    0.06, 0.01, 0.03, 0.01, 0.01,
]


def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def logit_pct(pct):
    """Logit of a percent (0..100), clamped to the scorer's range."""
    p = clamp(pct, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def pct_logit(lg):
    """Percent for a logit, clamped to the scorer's range."""
    if lg >= 0:
        return clamp(100.0 / (1.0 + math.exp(-lg)), RATE_LO, RATE_HI)
    return clamp(100.0 * math.exp(lg) / (1.0 + math.exp(lg)),
                 RATE_LO, RATE_HI)


def round4(x):
    """Round to 4 decimals, halves up (the scorer's rounding)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def is_frozen(history):
    """True when the last 12 h of reads are all identical.

    Identical means the same players_tracked and the same rates in every
    read, and the reads span at least FROZEN_MIN_SPAN_H hours. A flat
    history like that means the recorder stopped updating, not that the
    game stopped moving.
    """
    last_time = parse_time(history[-1]["at"])
    recent = []
    for read in history:
        when = parse_time(read["at"])
        age_h = (last_time - when).total_seconds() / 3600.0
        if age_h <= FIT_WINDOW_H:
            recent.append(read)
    if len(recent) < 2:
        return False
    first_time = parse_time(recent[0]["at"])
    span_h = (last_time - first_time).total_seconds() / 3600.0
    if span_h < FROZEN_MIN_SPAN_H:
        return False
    players0 = recent[0].get("players_tracked")
    rates0 = recent[0]["rates"]
    for read in recent[1:]:
        if read.get("players_tracked") != players0:
            return False
        if read["rates"] != rates0:
            return False
    return True


def _weighted_slope(pts):
    """Weighted least-squares slope of (x, y) points, or None.

    pts are (x_days, y_logit, weight) triples. Returns None when the
    points carry no usable time spread.
    """
    if len(pts) < 2:
        return None
    sw = sum(p[2] for p in pts)
    if sw <= 0:
        return None
    xbar = sum(p[2] * p[0] for p in pts) / sw
    ybar = sum(p[2] * p[1] for p in pts) / sw
    sxx = sum(p[2] * (p[0] - xbar) ** 2 for p in pts)
    if sxx <= 1e-12:
        return None
    sxy = sum(p[2] * (p[0] - xbar) * (p[1] - ybar) for p in pts)
    return sxy / sxx


def _curve_factor(pts):
    """Curvature factor from the trend's two halves.

    Splits the fit window in half (older vs recent reads) and fits a
    slope to each. When the recent slope is a fraction of the older one
    the trend is decelerating and the drift is damped toward
    CURVE_DAMP_LO; when it is steeper the drift is boosted up to
    CURVE_BOOST_HI. A reversed trend damps to the floor. Returns 1.0
    when either half is too thin to fit.
    """
    half = FIT_WINDOW_H / 2.0
    older = []
    recent = []
    for (x, y, w) in pts:
        age_h = -x * 24.0
        if age_h <= half:
            recent.append((x, y, w))
        else:
            older.append((x, y, w))
    b_old = _weighted_slope(older)
    b_new = _weighted_slope(recent)
    if b_old is None or b_new is None:
        return 1.0
    if b_old == 0:
        return 1.0
    return clamp(b_new / b_old, CURVE_DAMP_LO, CURVE_BOOST_HI)


def _quantiles(median, spread):
    """The five quantiles for a logit median and spread, percents."""
    qs = [round4(pct_logit(median + z * spread)) for z in Z]
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_trophy(idx, case, horizon_days, frozen):
    """Five quantiles (percents) for one trophy at the target time."""
    history = case["history"]
    last_rate = history[-1]["rates"][idx]
    if last_rate is None:
        last_rate = RATE_LO
    last_lg = logit_pct(last_rate)

    if frozen and horizon_days <= CATCHUP_MAX_HORIZON_D:
        # Frozen history, short horizon: catch up toward mature rates.
        level = ((1.0 - CATCHUP_FRACTION) * last_lg
                 + CATCHUP_FRACTION * logit_pct(MATURE_RATES[idx]))
        spread = VOL_FLOOR * math.sqrt(horizon_days)
        return _quantiles(level, spread)

    # Live history: recency-weighted line on logit(rate) vs time.
    last_time = parse_time(history[-1]["at"])
    pts = []
    for read in history:
        rate = read["rates"][idx]
        if rate is None:
            continue
        when = parse_time(read["at"])
        age_h = (last_time - when).total_seconds() / 3600.0
        if age_h > FIT_WINDOW_H:
            continue
        x = -age_h / 24.0
        y = logit_pct(rate)
        w = math.exp(-math.log(2.0) * age_h / HALF_LIFE_H)
        pts.append((x, y, w))

    if len(pts) < 2:
        level = last_lg
        slope = 0.0
        se = 0.0
        sigma = VOL_FLOOR
        curve = 1.0
    else:
        sw = sum(p[2] for p in pts)
        xbar = sum(p[2] * p[0] for p in pts) / sw
        ybar = sum(p[2] * p[1] for p in pts) / sw
        sxx = sum(p[2] * (p[0] - xbar) ** 2 for p in pts)
        sxy = sum(p[2] * (p[0] - xbar) * (p[1] - ybar) for p in pts)
        if sxx <= 1e-12:
            level = ybar
            slope = 0.0
            se = 0.0
        else:
            slope = sxy / sxx
            level = ybar - slope * xbar
        resid = [p[1] - (level + slope * p[0]) for p in pts]
        sigma2 = sum(p[2] * r * r for p, r in zip(pts, resid)) / sw
        sigma = math.sqrt(max(sigma2, 0.0))
        se = math.sqrt(max(sigma2, 0.0) / sxx) if sxx > 1e-12 else 0.0
        curve = _curve_factor(pts)

    # Shrink the slope by its own uncertainty (t-statistic shrinkage).
    if se > 0:
        t = slope / se
        shrink = (t * t) / (t * t + T_SHRINK * T_SHRINK)
    else:
        shrink = 1.0 if slope != 0 else 0.0

    # Damp the drift: long horizons, rates near saturation, curvature.
    # Level damping is on the logit scale: a rate near 0% or 100% has a
    # large |logit|, so its drift is damped toward LEVEL_DAMP_LO, while
    # a mid-range rate keeps the full drift. Clamped so the factor stays
    # in [LO, HI] and can never flip the slope's sign.
    damp_h = HORIZON_DAMP_DAYS / (HORIZON_DAMP_DAYS + horizon_days)
    frac = abs(level) / LEVEL_DAMP_SCALE
    damp_level = clamp(
        LEVEL_DAMP_HI
        - (LEVEL_DAMP_HI - LEVEL_DAMP_LO) * frac ** LEVEL_DAMP_POWER,
        LEVEL_DAMP_LO, LEVEL_DAMP_HI)
    slope_eff = slope * shrink * damp_h * damp_level * curve

    median = level + slope_eff * horizon_days
    median = clamp(median, last_lg - MAX_MEDIAN_SHIFT,
                   last_lg + MAX_MEDIAN_SHIFT)

    # Spread: residual volatility (shrunk toward the floor) plus slope
    # uncertainty, capped.
    vol_base = clamp(sigma, VOL_FLOOR, VOL_CAP)
    vol = VOL_FLOOR + VOL_SHRINK * (vol_base - VOL_FLOOR)
    spread = vol * math.sqrt(horizon_days) + SLOPE_SE_WEIGHT * se * horizon_days
    spread = clamp(spread, 0.0, SPREAD_CAP)
    return _quantiles(median, spread)


def forecast_case(case):
    """Forecast every trophy in one case."""
    horizon_days = case["horizon_minutes"] / 1440.0
    frozen = is_frozen(case["history"])
    rows = []
    for i, tr in enumerate(case["trophies"]):
        rows.append({"id": tr["id"],
                     "q": forecast_trophy(i, case, horizon_days, frozen)})
    return {"trophies": rows}


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(in_path.read_text())
        out = forecast_case(case)
        out_path = OUTPUT_DIR / (in_path.stem + ".out")
        out_path.write_text(json.dumps(out))


if __name__ == "__main__":
    main()
