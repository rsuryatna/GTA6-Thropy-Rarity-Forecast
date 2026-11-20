"""Forecast GTA VI trophy earn rates for make.fun gta6-trophy-rarity-forecast-01.

The program reads one case at a time from the input directory. A case is the
house's recording of Exophase's GTA VI PlayStation 5 trophy page: the 55
trophies and, for each read before the case checkpoint, every trophy's earn
rate (percent of tracked players who hold it) and the players-tracked count.

For every trophy it writes five quantiles (5%, 25%, 50%, 75%, 95%) of the
earn rate at the case target. The model works in logit space, because the
score is a pinball loss on that scale:

  - Per trophy, fit a recency-weighted straight line to logit(rate) against
    time over the recent reads. That gives a drift (logit per day), a level
    at the last read, and the fit's residual spread and slope uncertainty.
  - The median forecast is the fitted level plus the drift carried forward
    over the horizon, shrunk by a damping factor. The damping grows with the
    horizon (growth saturates: a trophy cannot keep its launch-night pace
    for four days) and with the current rate (a trophy already held by most
    players has little headroom left). The drift is allowed to be negative:
    when the tracked player base grows faster than new earners, a rate
    falls, and the recent data shows that (the platinum did exactly that).
  - The spread of the five quantiles combines a random-walk part, whose
    per-day size comes from the residual spread of the trend fit (floored so
    a flat history still gets an honest spread), and a drift-uncertainty
    part, which grows linearly with the horizon.

Everything is deterministic: no clock, no randomness, trophies in id order,
case files in name order, so two runs write identical bytes.

Standard library only. Run it as the grader does (python3 -P -s main.py) or
locally with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR pointing at folders.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels the contest scores, and the standard normal's value at each.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# --- Tunables (chosen from the synthetic backtest; see the dev folder) ---
WINDOW_HOURS = 12.0       # how far back the trend fit looks
HALF_LIFE_HOURS = 6.0     # recency weight half-life inside that window
DAMP_HORIZON_DAYS = 3.0   # saturation timescale for the horizon damping
DAMP_LEVEL_MIN = 0.25     # strongest level damping (rate already high)
DAMP_LEVEL_MAX = 1.0      # no level damping (rate near zero)
LEVEL_POWER = 2.0         # level damping is squared: mid rates saturate hard
SIGMA_FLOOR = 0.35        # floor on per-day random-walk volatility (logit)
SIGMA_CAP = 1.0           # cap on per-day random-walk volatility (logit)
SIGMA_SHRINK = 0.5        # residuals also carry measurement noise: shrink
SIGMA_TOTAL_CAP = 3.0     # cap on the total forecast spread (logit)
SPREAD_SE_FACTOR = 0.5    # slope uncertainty feeds the spread only half-weight
MAX_LOGIT_MOVE = 3.0      # cap on how far the median may move from the level
CURVE_MIN = 0.3           # strongest extra damping when decelerating
CURVE_MAX = 2.0           # strongest extra extrapolation when accelerating
T_SHRINK_K = 2.0          # slope t-statistic shrinkage: a weak/noisy trend
                          # is shrunk toward persistence, a strong one kept
RATE_MIN = 0.01           # lowest rate the scorer can see (Exophase floor)
RATE_MAX = 99.99          # highest rate the scorer can see


def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def logit_of(pct):
    """Percent (0..100) to logit, clamped to the scorer's visible range."""
    p = clamp(pct, RATE_MIN, RATE_MAX) / 100.0
    return math.log(p / (1.0 - p))


def pct_of(logit_value):
    """Logit back to percent, clamped to the scorer's visible range."""
    if logit_value >= 0:
        odds = math.exp(-logit_value)
        pct = 100.0 / (1.0 + odds)
    else:
        odds = math.exp(logit_value)
        pct = 100.0 * odds / (1.0 + odds)
    return clamp(pct, RATE_MIN, RATE_MAX)


def round4_half_up(x):
    """Round to 4 decimals, halves away from zero (matches the house)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    """ISO 8601 timestamp (with Z) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_line(times, values, weights):
    """Weighted least squares for y = a + b*t.

    Returns (intercept a, slope b, slope standard error, residual std).
    Times are in days; the intercept is the fitted value at t = 0.
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
    residuals = [y - (intercept + slope * t) for t, y in zip(times, values)]
    dof = max(n - 2, 1)
    resid_var = sum(r * r for r in residuals) / dof
    resid_std = math.sqrt(max(resid_var, 0.0))
    # Variance of the slope: sigma^2 / sum(w (t - tbar_w)^2).
    sxx_centred = sxx - sx * sx / sw
    if sxx_centred <= 1e-12:
        slope_se = 0.0
    else:
        slope_se = resid_std / math.sqrt(sxx_centred)
    return intercept, slope, slope_se, resid_std


def half_slopes(times, values, weights):
    """Slopes of the first and second half of the window (curvature check).

    A rising slope means the trophy is still accelerating into its growth;
    a falling slope means it is bending toward its ceiling. Returns
    (slope_first, slope_second) or (None, None) when a half is too thin.
    """
    n = len(times)
    if n < 6:
        return None, None
    order = sorted(range(n), key=lambda i: times[i])
    mid = n // 2
    first = order[:mid]
    second = order[mid:]
    out = []
    for idx in (first, second):
        if len(idx) < 3:
            return None, None
        t = [times[i] for i in idx]
        y = [values[i] for i in idx]
        w = [weights[i] for i in idx]
        fit = weighted_line(t, y, w)
        if fit is None:
            return None, None
        out.append(fit[1])
    return out[0], out[1]


def forecast_trophy(history, trophy_index, horizon_days):
    """Five quantiles for one trophy, from the case history.

    history: list of reads, oldest first, each with 'at' and 'rates'.
    trophy_index: position of this trophy in each read's rates list.
    """
    # Collect (age in days before the last read, logit rate) pairs.
    points = []
    last_time = None
    last_rate = None
    for read in history:
        rate = read["rates"][trophy_index]
        if rate is None:
            continue
        when = parse_time(read["at"])
        if last_time is None or when > last_time:
            last_time = when
        points.append((when, logit_of(rate)))
    if not points or last_time is None:
        # No usable read at all: a flat, wide, low forecast (still valid).
        centre = logit_of(0.01)
        spread = SIGMA_FLOOR * math.sqrt(max(horizon_days, 0.0))
        spread = min(spread, SIGMA_TOTAL_CAP)
        qs = [round4_half_up(pct_of(centre + z * spread)) for z in Z_VALUES]
        return _monotone(qs)
    last_rate = None
    for read in reversed(history):
        rate = read["rates"][trophy_index]
        if rate is not None:
            last_rate = rate
            break
    if last_rate is None:
        last_rate = 0.01

    # Recent window, recency-weighted, times in days before the last read.
    window_days = WINDOW_HOURS / 24.0
    half_life_days = HALF_LIFE_HOURS / 24.0
    recent = []
    for when, y in points:
        age_days = (last_time - when).total_seconds() / 86400.0
        if age_days <= window_days:
            weight = 0.5 ** (age_days / half_life_days)
            recent.append((-age_days, y, weight))
    if len(recent) < 3:
        # Too few recent reads: use everything, equal weights.
        recent = []
        for when, y in points:
            age_days = (last_time - when).total_seconds() / 86400.0
            recent.append((-age_days, y, 1.0))
    times = [t for t, _, _ in recent]
    values = [y for _, y, _ in recent]
    weights = [w for _, _, w in recent]

    fit = weighted_line(times, values, weights)
    if fit is None:
        level = logit_of(last_rate)
        slope = 0.0
        slope_se = 0.0
        resid_std = 0.0
        gaps = []
    else:
        level, slope, slope_se, resid_std = fit
        gaps = []
        ordered = sorted(t for t in times)
        for a, b in zip(ordered, ordered[1:]):
            if b > a:
                gaps.append(b - a)
    if gaps:
        gaps.sort()
        typical_gap = gaps[len(gaps) // 2]
    else:
        typical_gap = 1.0 / 48.0  # 30 minutes, the early recorder cadence
    typical_gap = max(typical_gap, 1e-4)

    # --- Median: drift carried forward, damped for saturation. ---
    # Horizon damping: the trend cannot hold at full strength for days.
    damp_horizon = 1.0 / (1.0 + horizon_days / DAMP_HORIZON_DAYS)
    # Level damping: a rate already high has little headroom left. The
    # damping is squared, because saturation bites hard once a trophy is
    # a third of the way up: a 33% rate has far less headroom than a 3% one.
    level_factor = clamp(1.0 - last_rate / 100.0, DAMP_LEVEL_MIN, DAMP_LEVEL_MAX)
    level_factor = level_factor ** LEVEL_POWER
    damp = damp_horizon * level_factor
    # Curvature: a trend that is still steepening is early in its growth
    # (extrapolate more); one that is flattening is near its ceiling
    # (extrapolate less). Compare the two halves of the window. The boost
    # from acceleration is scaled by headroom: a low rate that is still
    # steepening has room to run, while a mid rate that is steepening is
    # about to bend over. Deceleration always damps fully.
    slope_first, slope_second = half_slopes(times, values, weights)
    if (slope_first is not None and slope_second is not None
            and slope_first > 1e-6 and slope_second > 1e-6):
        ratio = slope_second / slope_first
        headroom = clamp(1.0 - last_rate / 100.0, DAMP_LEVEL_MIN, DAMP_LEVEL_MAX)
        if ratio >= 1.0:
            curve_factor = 1.0 + (min(ratio, CURVE_MAX) - 1.0) * headroom
        else:
            curve_factor = max(ratio, CURVE_MIN)
        damp *= curve_factor
        damp = min(damp, 1.0)
    # Trend-strength shrinkage: a slope that is large compared with its own
    # standard error is a real signal (keep it); a slope buried in noise
    # is mostly measurement wobble (shrink it toward persistence). This is
    # what keeps the forecast honest when growth is slow and the recent
    # wiggles are noise, not signal.
    if slope_se > 1e-12:
        t_stat = abs(slope) / slope_se
        damp *= t_stat / (t_stat + T_SHRINK_K)
    move = clamp(slope * horizon_days * damp, -MAX_LOGIT_MOVE, MAX_LOGIT_MOVE)
    median_logit = clamp(level + move, logit_of(RATE_MIN), logit_of(RATE_MAX))

    # --- Spread: random-walk part plus drift-uncertainty part. ---
    # Residuals mix true walk noise with per-read measurement noise, so the
    # implied per-day walk volatility is shrunk, then floored and capped.
    walk_vol = resid_std / math.sqrt(typical_gap) * SIGMA_SHRINK
    walk_vol = clamp(walk_vol, SIGMA_FLOOR, SIGMA_CAP)
    spread = math.sqrt(walk_vol * walk_vol * horizon_days
                       + (SPREAD_SE_FACTOR * slope_se * horizon_days) ** 2)
    spread = clamp(spread, 0.0, SIGMA_TOTAL_CAP)

    quantiles = []
    for z in Z_VALUES:
        quantiles.append(round4_half_up(pct_of(median_logit + z * spread)))
    return _monotone(quantiles)


def _monotone(quantiles):
    """Force non-decreasing quantiles (rounding can tie them; never invert)."""
    out = []
    running = None
    for q in quantiles:
        if running is None or q > running:
            running = q
        out.append(running)
    return out


def forecast_case(case):
    """Whole-case forecast: {"trophies": [{"id", "q": [...]}, ...]}."""
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0
    trophies = sorted(case["trophies"], key=lambda t: t["id"])
    rows = []
    for index, trophy in enumerate(trophies):
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(history, index, horizon_days),
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
