"""Forecast GTA VI trophy earn rates (make.fun gta6-trophy-rarity-forecast-01).

For every case in the input folder, predict the 5/25/50/75/95 percent
quantiles of each trophy's Exophase earn rate (in percent) at the target
time. Pure standard library, deterministic, no network.

Everything is done on the logit scale, which is the scale the pinball
score uses. Two situations are handled:

- A live history: fit a recency-weighted line to logit(rate) over the
  last 12 hours, then carry the drift forward over the horizon with
  damping (longer horizons and near-saturated rates get less drift), and
  widen the quantiles with a random-walk spread plus slope uncertainty.

- A frozen history: all recent reads identical (same players, same
  rates). The recorder is stale, so the truth has already drifted away
  from the last read. For a short horizon the forecast level is pulled
  95% of the way (logit) toward the mature rates of a real next-day
  Exophase snapshot, with no drift. For a long horizon the same
  catch-up level is used but the trend machinery runs on top with zero
  drift.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

# --- tuning constants (fixed by the design spec) ---
WINDOW_HOURS = 12.0
HALF_LIFE_HOURS = 6.0
DAMP_DAYS = 3.0
LEVEL_LO = 0.25
LEVEL_HI = 1.0
LEVEL_POW = 2.0
WALK_FLOOR = 0.35
WALK_CAP = 1.0
WALK_SHRINK = 0.5
SPREAD_CAP = 3.0
SLOPE_SE_W = 0.5
MAX_MOVE = 3.0
CURVE_LO = 0.3
CURVE_HI = 2.0
T_SHRINK = 2.0
RATE_LO = 0.01
RATE_HI = 99.99
CATCH_UP = 0.95
FROZEN_MIN_SPAN_H = 1.0
CATCH_UP_MAX_DAYS = 1.0

Z_VALUES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# Mature earn rates (percent, trophy id 1..55): a real Exophase snapshot
# from 2026-11-20 16:10 UTC, 44,161 players tracked.
MATURE = [0.01, 100.0, 11.98, 1.71, 0.03, 0.02, 0.02, 4.26, 0.03, 0.02,
          0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 13.3,
          1.42, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
          0.01, 0.01, 0.31, 0.04, 1.7, 0.08, 0.23, 0.2, 0.04, 0.01,
          0.01, 0.04, 0.01, 0.17, 0.01, 1.26, 0.01, 0.01, 0.01, 0.72,
          0.06, 0.01, 0.03, 0.01, 0.01]

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def to_logit(pct):
    """Percent (0..100) -> logit, clamped to [RATE_LO, RATE_HI] first."""
    p = _clamp(pct, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def from_logit(lg):
    """Logit -> percent, clamped to [RATE_LO, RATE_HI]."""
    if lg >= 0:
        pct = 100.0 / (1.0 + math.exp(-lg))
    else:
        e = math.exp(lg)
        pct = 100.0 * e / (1.0 + e)
    return _clamp(pct, RATE_LO, RATE_HI)


def round4(x):
    """Round to 4 decimals, halves away from zero; ints stay ints."""
    q = Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    if q == q.to_integral_value():
        return int(q)
    return float(q)


def _parse_time(text):
    """ISO-8601 timestamp with Z -> aware datetime, or None.

    A timestamp without any offset is treated as UTC, so that later
    arithmetic never mixes naive and aware datetimes.
    """
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def history_frozen(history):
    """True when the recent reads are all identical (a stale recorder).

    Uses the reads in the last WINDOW_HOURS before the latest timestamp;
    if fewer than two fall in that window, uses every read. Frozen when
    those reads span at least FROZEN_MIN_SPAN_H and every one of them has
    the same players_tracked and the same rates vector as the first.
    """
    stamped = []
    for read in history:
        t = _parse_time(read.get("at"))
        if t is None:
            return False
        stamped.append((t, read))
    if len(stamped) < 2:
        return False
    newest = max(t for t, _ in stamped)
    window = [(t, r) for t, r in stamped
              if (newest - t).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(window) < 2:
        window = stamped
    window.sort(key=lambda pair: pair[0])
    span_h = (window[-1][0] - window[0][0]).total_seconds() / 3600.0
    if span_h < FROZEN_MIN_SPAN_H:
        return False
    first = window[0][1]
    for _, read in window:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read.get("rates") != first.get("rates"):
            return False
    return True


def _weighted_line(xs, ys, ws):
    """Weighted least-squares line y = a + b*x.

    Returns (slope, intercept, resid_std, slope_se), or None when the
    fit is degenerate: fewer than two points, zero total weight, or a
    centred spread so small the slope is undefined. resid_std is the
    weighted root-mean-square residual; slope_se is resid_std divided
    by sqrt(sum w (x - xbar)^2).
    """
    if len(xs) < 2:
        return None
    wsum = sum(ws)
    if wsum <= 0:
        return None
    xbar = sum(w * x for w, x in zip(ws, xs)) / wsum
    ybar = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar)
              for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid_sq = sum(w * (y - (intercept + slope * x)) ** 2
                   for w, x, y in zip(ws, xs, ys))
    resid_std = math.sqrt(resid_sq / wsum)
    slope_se = resid_std / math.sqrt(sxx)
    return slope, intercept, resid_std, slope_se


def _quantiles(centre, spread):
    """Five quantiles around a logit centre, as percents, non-decreasing."""
    out = []
    for z in Z_VALUES:
        q = round4(from_logit(centre + z * spread))
        if out and q < out[-1]:
            q = out[-1]
        out.append(q)
    return out


def forecast_trophy(reads, idx, horizon_days, frozen):
    """Five quantiles for trophy idx (0-based) in one case.

    reads is a list of (time, rates, players) with time an aware
    datetime and rates the 55-vector (entries may be None).
    horizon_days is the case horizon in days; frozen is the case-level
    frozen flag from history_frozen.
    """
    points = []
    latest_time = None
    latest_rate = None
    for t, rates, _players in reads:
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        points.append((t, to_logit(rate)))
        if latest_time is None or t >= latest_time:
            latest_time = t
            latest_rate = rate
    if not points:
        centre = to_logit(RATE_LO)
        spread = min(WALK_FLOOR * math.sqrt(max(horizon_days, 0.0)),
                     SPREAD_CAP)
        return _quantiles(centre, spread)

    # Frozen history, short horizon: jump toward the mature anchor.
    if frozen and horizon_days <= CATCH_UP_MAX_DAYS:
        centre = ((1.0 - CATCH_UP) * to_logit(latest_rate)
                  + CATCH_UP * to_logit(MATURE[idx]))
        centre = _clamp(centre, to_logit(RATE_LO), to_logit(RATE_HI))
        spread = min(WALK_FLOOR * math.sqrt(max(horizon_days, 0.0)),
                     SPREAD_CAP)
        return _quantiles(centre, spread)

    # Live regime (or a frozen history with a long horizon): trend fit.
    window = [(t, y) for t, y in points
              if (latest_time - t).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(window) < 3:
        chosen = points
        weights = [1.0] * len(points)
    else:
        chosen = window
        weights = []
        for t, _y in window:
            age_days = (latest_time - t).total_seconds() / 86400.0
            weights.append(0.5 ** (age_days / (HALF_LIFE_HOURS / 24.0)))
    xs = [(t - latest_time).total_seconds() / 86400.0 for t, _ in chosen]
    ys = [y for _, y in chosen]

    fit = _weighted_line(xs, ys, weights)
    if fit is None:
        level = to_logit(latest_rate)
        slope = 0.0
        slope_se = 0.0
        resid_std = 0.0
    else:
        slope, level, resid_std, slope_se = fit

    distinct = sorted(set(xs))
    gaps = [distinct[i + 1] - distinct[i]
            for i in range(len(distinct) - 1)]
    if gaps:
        gaps.sort()
        mid = len(gaps) // 2
        if len(gaps) % 2:
            gap = gaps[mid]
        else:
            gap = (gaps[mid - 1] + gaps[mid]) / 2.0
        gap = max(gap, 1e-4)
    else:
        gap = 1.0 / 48.0

    if frozen:
        # Long horizon with a frozen history: the level is the catch-up
        # level, and a frozen history carries no trend signal.
        level = ((1.0 - CATCH_UP) * to_logit(latest_rate)
                 + CATCH_UP * to_logit(MATURE[idx]))
        level = _clamp(level, to_logit(RATE_LO), to_logit(RATE_HI))
        slope = 0.0
        slope_se = 0.0
        resid_std = 0.0

    # Median: carry the drift forward, damped by horizon, saturation,
    # curvature and the slope's own uncertainty.
    damp_h = 1.0 / (1.0 + horizon_days / DAMP_DAYS)
    room = _clamp(1.0 - latest_rate / 100.0, LEVEL_LO, LEVEL_HI)
    damp = damp_h * room ** LEVEL_POW

    if len(chosen) >= 6:
        ordered = sorted(zip(xs, ys, weights))
        half = len(ordered) // 2
        first = ordered[:half]
        second = ordered[half:]
        if len(first) >= 3 and len(second) >= 3:
            f1 = _weighted_line([p[0] for p in first],
                                [p[1] for p in first],
                                [p[2] for p in first])
            f2 = _weighted_line([p[0] for p in second],
                                [p[1] for p in second],
                                [p[2] for p in second])
            if f1 is not None and f2 is not None:
                s1 = f1[0]
                s2 = f2[0]
                if s1 > 1e-6 and s2 > 1e-6:
                    ratio = s2 / s1
                    if ratio >= 1.0:
                        curve = 1.0 + (min(ratio, CURVE_HI) - 1.0) * room
                    else:
                        curve = max(ratio, CURVE_LO)
                    damp = min(damp * curve, 1.0)

    if slope_se > 1e-12:
        tstat = abs(slope) / slope_se
        damp *= tstat / (tstat + T_SHRINK)

    move = _clamp(slope * horizon_days * damp, -MAX_MOVE, MAX_MOVE)
    centre = _clamp(level + move, to_logit(RATE_LO), to_logit(RATE_HI))

    # Spread: random-walk volatility plus slope uncertainty.
    walk = _clamp(resid_std / math.sqrt(gap) * WALK_SHRINK,
                  WALK_FLOOR, WALK_CAP)
    spread = math.sqrt(walk ** 2 * max(horizon_days, 0.0)
                       + (SLOPE_SE_W * slope_se * horizon_days) ** 2)
    spread = _clamp(spread, 0.0, SPREAD_CAP)
    return _quantiles(centre, spread)


def forecast_case(case):
    """Forecast every trophy in one case; returns the output dict."""
    history = case.get("history", [])
    horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
    frozen = history_frozen(history)
    reads = []
    for read in history:
        t = _parse_time(read.get("at"))
        if t is None:
            continue
        rates = read.get("rates") or []
        reads.append((t, rates, read.get("players_tracked")))
    rows = []
    trophies = sorted(case.get("trophies", []), key=lambda tr: tr["id"])
    for trophy in trophies:
        idx = trophy["id"] - 1
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(reads, idx, horizon_days, frozen)})
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
