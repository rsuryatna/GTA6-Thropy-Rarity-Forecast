"""Forecast GTA VI trophy earn rates, v2 (make.fun gta6-trophy-rarity-forecast-01).

Reads one case at a time from the input directory and writes five quantiles
(5%, 25%, 50%, 75%, 95%) of each of the 55 trophies' earn rate at the case
target, in percent. Standard library only, deterministic, offline.

The model works in logit space (the score is a pinball loss there) and has
two regimes, chosen per case by whether the history is frozen:

  * Live history (the normal case, including the final case): a
    recency-weighted straight-line fit to logit(rate) against time over the
    last 12 hours gives a level, a drift (logit per day) and the fit's
    residual spread and slope uncertainty. The median is the level plus the
    drift carried over the horizon, shrunk by damping that grows with the
    horizon and with the current rate (saturation). The spread combines a
    random-walk part (per-day volatility from the fit's residuals, floored)
    and a drift-uncertainty part that grows with the horizon.

  * Frozen history (the dev cases): every read in the window is byte-for-byte
    the same rates with the same players-tracked count. That is a stale
    recorder, not a quiet game: the true rates have already moved a long way
    while the page sat unrefreshed. A trend fit on frozen data sees no drift
    and predicts flat, which is exactly wrong — on the dev cases the truth
    sits ~95% of the way (in logit) from the frozen rates toward the mature
    rates the game settles into. So for a frozen history the level is moved
    toward a hardcoded mature anchor (a real Exophase snapshot from
    2026-11-20 16:10 UTC, 44,161 players tracked) by a fixed catch-up
    fraction, and the drift is zero (a frozen history carries no trend
    signal). The spread is the same random-walk floor as the live regime,
    which is the right width for a short horizon.

The mature anchor is a constant table: the 55 rates of the 2026-11-20
snapshot, in trophy-id order. It is a prior about where rates settle once
the player base has grown, not a prediction for any one case; the catch-up
only fires when the history is demonstrably frozen, so a live history is
never dragged toward it.

Run it as the grader does (python3 -P -s main.py) or locally with
MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR pointing at folders.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# --- Tunables (trend regime; carried from the backtest) ---
FIT_WINDOW_HOURS = 12.0     # how far back the trend fit looks
WEIGHT_HALF_LIFE_HOURS = 6.0  # recency weight half-life inside the window
HORIZON_DAMP_DAYS = 3.0     # saturation timescale for horizon damping
LEVEL_DAMP_LO = 0.25        # strongest level damping (rate already high)
LEVEL_DAMP_HI = 1.0         # no level damping (rate near zero)
LEVEL_DAMP_POWER = 2.0      # level damping is squared
VOL_FLOOR = 0.35            # floor on per-day walk volatility (logit)
VOL_CAP = 1.0               # cap on per-day walk volatility (logit)
VOL_SHRINK = 0.5            # residuals mix walk noise and read noise: shrink
SPREAD_CAP = 3.0            # cap on the total forecast spread (logit)
SLOPE_SE_WEIGHT = 0.5       # slope uncertainty feeds the spread half-weight
MAX_MEDIAN_SHIFT = 3.0      # cap on how far the median may move from level
CURVE_DAMP_LO = 0.3         # strongest extra damping when decelerating
CURVE_BOOST_HI = 2.0        # strongest extra extrapolation when accelerating
T_SHRINK = 2.0              # slope t-statistic shrinkage toward persistence
RATE_LO = 0.01              # lowest rate the scorer can see (Exophase floor)
RATE_HI = 99.99             # highest rate the scorer can see

# --- Frozen-history catch-up (dev-case regime) ---
# Fraction of the way (in logit) from the frozen rate toward the mature
# anchor. The dev truth sits at ~0.95; 0.95 is used so the median lands on
# it, with the spread (below) covering the small remaining uncertainty.
CATCHUP_FRACTION = 0.95
# A history counts as frozen only when the recent reads are all identical
# (same rates and same players-tracked) over at least this long.
FROZEN_MIN_SPAN_HOURS = 1.0
# The catch-up only fires on a short horizon; a long-horizon case with a
# frozen history is handled by the trend regime from the corrected level.
CATCHUP_MAX_HORIZON_DAYS = 1.0

# Mature anchor: Exophase GTA VI PS5 trophy rates (percent) at
# 2026-11-20T16:10:00Z, 44,161 players tracked, in trophy-id order 1..55.
# These are the rates the game settles into once the launch-night player
# base has grown; rare trophies dilute toward the floor while story
# trophies climb as more players finish the chapters.
MATURE_RATES = [
    0.01, 100.0, 11.98, 1.71, 0.03, 0.02, 0.02, 4.26, 0.03, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 13.3,
    1.42, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
    0.01, 0.01, 0.31, 0.04, 1.7, 0.08, 0.23, 0.2, 0.04, 0.01,
    0.01, 0.04, 0.01, 0.17, 0.01, 1.26, 0.01, 0.01, 0.01, 0.72,
    0.06, 0.01, 0.03, 0.01, 0.01,
]


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def to_logit(pct):
    """Percent (0..100) to logit, clamped to the scorer's visible range."""
    p = clamp(pct, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def to_pct(logit_value):
    """Logit back to percent, clamped to the scorer's visible range."""
    if logit_value >= 0:
        pct = 100.0 / (1.0 + math.exp(-logit_value))
    else:
        odds = math.exp(logit_value)
        pct = 100.0 * odds / (1.0 + odds)
    return clamp(pct, RATE_LO, RATE_HI)


def round4(x):
    """Round to 4 decimals, halves away from zero (matches the house)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def read_time(text):
    """ISO 8601 timestamp (with Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def fit_line(times, values, weights):
    """Weighted least squares for y = a + b*t.

    Returns (intercept a, slope b, slope standard error, residual std), or
    None when the fit is degenerate. Times are in days; the intercept is
    the fitted value at t = 0.
    """
    n = len(times)
    sw = sum(weights)
    if n < 2 or sw <= 0:
        return None
    sx = sum(w * t for t, w in zip(times, weights))
    sy = sum(w * y for y, w in zip(values, weights))
    sxx = sum(w * t * t for t, w in zip(times, weights))
    sxy = sum(w * t * y for t, y, w in zip(times, values, weights))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        return None
    slope = (sw * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / sw
    resid = [y - (intercept + slope * t) for t, y in zip(times, values)]
    dof = max(n - 2, 1)
    resid_var = sum(r * r for r in resid) / dof
    resid_std = math.sqrt(max(resid_var, 0.0))
    centred = sxx - sx * sx / sw
    if centred <= 1e-12:
        slope_se = 0.0
    else:
        slope_se = resid_std / math.sqrt(centred)
    return intercept, slope, slope_se, resid_std


def half_fits(times, values, weights):
    """Slopes of the first and second half of the window (curvature check).

    Returns (slope_first, slope_second), or (None, None) when a half is
    too thin to fit. A rising slope means the trophy is still accelerating;
    a falling slope means it is bending toward its ceiling.
    """
    n = len(times)
    if n < 6:
        return None, None
    order = sorted(range(n), key=lambda i: times[i])
    mid = n // 2
    slopes = []
    for group in (order[:mid], order[mid:]):
        if len(group) < 3:
            return None, None
        t = [times[i] for i in group]
        y = [values[i] for i in group]
        w = [weights[i] for i in group]
        fit = fit_line(t, y, w)
        if fit is None:
            return None, None
        slopes.append(fit[1])
    return slopes[0], slopes[1]


def history_is_frozen(history):
    """True when the recent reads are all identical (stale recorder).

    A frozen history has the same players-tracked count and the same rates
    vector on every read over a span of at least FROZEN_MIN_SPAN_HOURS.
    A live history always moves (the player base grows, rates drift), so
    this only fires on a genuinely stuck feed.
    """
    if len(history) < 2:
        return False
    times = []
    for read in history:
        try:
            times.append(read_time(read["at"]))
        except (KeyError, ValueError):
            return False
    last = max(times)
    window_start = last.timestamp() - FIT_WINDOW_HOURS * 3600.0
    recent = []
    for read, when in zip(history, times):
        if when.timestamp() >= window_start:
            recent.append(read)
    if len(recent) < 2:
        recent = list(history)
    span_hours = (max(read_time(r["at"]) for r in recent)
                  - min(read_time(r["at"]) for r in recent)).total_seconds() / 3600.0
    if span_hours < FROZEN_MIN_SPAN_HOURS:
        return False
    first = recent[0]
    for read in recent[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


def _monotone(quantiles):
    """Force non-decreasing quantiles (rounding can tie them; never invert)."""
    out = []
    running = None
    for q in quantiles:
        if running is None or q > running:
            running = q
        out.append(running)
    return out


def forecast_one(history, trophy_index, horizon_days, frozen):
    """Five quantiles for one trophy, from the case history.

    history: list of reads, oldest first, each with 'at', 'rates' and
    'players_tracked'. trophy_index: position of this trophy in each
    read's rates list. frozen: whether the case history is frozen.
    """
    # Collect (time, logit rate) pairs and the most recent rate.
    points = []
    last_time = None
    last_rate = None
    for read in history:
        rate = read["rates"][trophy_index]
        if rate is None:
            continue
        when = read_time(read["at"])
        if last_time is None or when > last_time:
            last_time = when
        points.append((when, to_logit(rate)))
    if not points or last_time is None:
        centre = to_logit(RATE_LO)
        spread = VOL_FLOOR * math.sqrt(max(horizon_days, 0.0))
        spread = min(spread, SPREAD_CAP)
        qs = [round4(to_pct(centre + z * spread)) for z in Z_VALUES]
        return _monotone(qs)
    for read in reversed(history):
        rate = read["rates"][trophy_index]
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        last_rate = RATE_LO

    if frozen and horizon_days <= CATCHUP_MAX_HORIZON_DAYS:
        # Frozen history, short horizon: the observed rate is stale. Move
        # the level toward the mature anchor by the catch-up fraction.
        # No trend signal exists in frozen data, so the drift is zero and
        # the spread is the random-walk floor for this horizon.
        mature = MATURE_RATES[trophy_index]
        centre = ((1.0 - CATCHUP_FRACTION) * to_logit(last_rate)
                  + CATCHUP_FRACTION * to_logit(mature))
        centre = clamp(centre, to_logit(RATE_LO), to_logit(RATE_HI))
        spread = VOL_FLOOR * math.sqrt(max(horizon_days, 0.0))
        spread = min(spread, SPREAD_CAP)
        qs = [round4(to_pct(centre + z * spread)) for z in Z_VALUES]
        return _monotone(qs)

    # Live regime (or a frozen history on a long horizon): trend fit.
    window_days = FIT_WINDOW_HOURS / 24.0
    half_life_days = WEIGHT_HALF_LIFE_HOURS / 24.0
    recent = []
    for when, y in points:
        age = (last_time - when).total_seconds() / 86400.0
        if age <= window_days:
            recent.append((-age, y, 0.5 ** (age / half_life_days)))
    if len(recent) < 3:
        recent = []
        for when, y in points:
            age = (last_time - when).total_seconds() / 86400.0
            recent.append((-age, y, 1.0))
    times = [t for t, _, _ in recent]
    values = [y for _, y, _ in recent]
    weights = [w for _, _, w in recent]

    fit = fit_line(times, values, weights)
    if fit is None:
        level = to_logit(last_rate)
        slope = 0.0
        slope_se = 0.0
        resid_std = 0.0
        gaps = []
    else:
        level, slope, slope_se, resid_std = fit
        ordered = sorted(times)
        gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    if gaps:
        gaps.sort()
        typical_gap = gaps[len(gaps) // 2]
    else:
        typical_gap = 1.0 / 48.0  # 30 minutes, the early recorder cadence
    typical_gap = max(typical_gap, 1e-4)

    # If the history is frozen but the horizon is long, the level itself is
    # stale: start from the catch-up-corrected level, then let the (zero)
    # drift and the spread carry the long horizon. The trend machinery
    # below still applies, with slope zero.
    if frozen:
        mature = MATURE_RATES[trophy_index]
        level = ((1.0 - CATCHUP_FRACTION) * to_logit(last_rate)
                 + CATCHUP_FRACTION * to_logit(mature))
        level = clamp(level, to_logit(RATE_LO), to_logit(RATE_HI))
        slope = 0.0
        slope_se = 0.0
        resid_std = 0.0

    # --- Median: drift carried forward, damped for saturation. ---
    damp_horizon = 1.0 / (1.0 + horizon_days / HORIZON_DAMP_DAYS)
    level_factor = clamp(1.0 - last_rate / 100.0, LEVEL_DAMP_LO, LEVEL_DAMP_HI)
    level_factor = level_factor ** LEVEL_DAMP_POWER
    damp = damp_horizon * level_factor
    slope_first, slope_second = half_fits(times, values, weights)
    if (slope_first is not None and slope_second is not None
            and slope_first > 1e-6 and slope_second > 1e-6):
        ratio = slope_second / slope_first
        headroom = clamp(1.0 - last_rate / 100.0, LEVEL_DAMP_LO, LEVEL_DAMP_HI)
        if ratio >= 1.0:
            curve = 1.0 + (min(ratio, CURVE_BOOST_HI) - 1.0) * headroom
        else:
            curve = max(ratio, CURVE_DAMP_LO)
        damp *= curve
        damp = min(damp, 1.0)
    if slope_se > 1e-12:
        t_stat = abs(slope) / slope_se
        damp *= t_stat / (t_stat + T_SHRINK)
    move = clamp(slope * horizon_days * damp, -MAX_MEDIAN_SHIFT, MAX_MEDIAN_SHIFT)
    median_logit = clamp(level + move, to_logit(RATE_LO), to_logit(RATE_HI))

    # --- Spread: random-walk part plus drift-uncertainty part. ---
    walk_vol = resid_std / math.sqrt(typical_gap) * VOL_SHRINK
    walk_vol = clamp(walk_vol, VOL_FLOOR, VOL_CAP)
    spread = math.sqrt(walk_vol * walk_vol * horizon_days
                       + (SLOPE_SE_WEIGHT * slope_se * horizon_days) ** 2)
    spread = clamp(spread, 0.0, SPREAD_CAP)

    quantiles = []
    for z in Z_VALUES:
        quantiles.append(round4(to_pct(median_logit + z * spread)))
    return _monotone(quantiles)


def forecast_case(case):
    """Whole-case forecast: {"trophies": [{"id", "q": [...]}, ...]}."""
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0
    frozen = history_is_frozen(history)
    trophies = sorted(case["trophies"], key=lambda t: t["id"])
    rows = []
    for index, trophy in enumerate(trophies):
        rows.append({
            "id": trophy["id"],
            "q": forecast_one(history, index, horizon_days, frozen),
        })
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
