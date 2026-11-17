"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Model v2: scale-aware empirical launch-day quantiles.
  A big console launch's Twitch day follows a diurnal shape driven by
  the midnight-local unlock rolling around the world: hour 1 (00:00-01:00
  UTC) already holds the UK midnight crowd plus New Zealand and Europe,
  the early UTC morning sags, and the count climbs into the US evening
  peak. But the SHAPE depends strongly on the launch's size: the
  biggest launches (hour 1 ~300-875k) peak at only ~1.0-1.3x hour 1,
  while small launches (hour 1 < 1k) explode to 5-8x by hour 19. The
  submitted v1 model pooled all 38 comparables equally, so its median
  sat at ~2.4x by hour 19 and its 95th quantile at ~350x -- fine for a
  mid-size launch, but it LOSES to the house baseline on big launches
  (leave-one-out 0.361 vs baseline 0.251) and is badly miscalibrated
  for a mega launch like GTA VI, whose hour-1 print will dwarf every
  comparable (max 875k).

  v2 fixes that by making the forecast conditional on hour-1 size,
  with two local estimators in log(hour-1) space, both centred on the
  case's own hour-1 print:
    * centre: a kernel-weighted local-linear fit (Gaussian kernel,
      sigma 1.5 log units) of each hour's log-ratio on log(hour-1),
      evaluated at the case's log(hour-1) CLAMPED to the observed
      range -- no extrapolation past the biggest comparable. The
      scale trend is real but must saturate: the biggest observed
      launches peak at 0.95-1.31x, and extrapolating the fitted slope
      past them would push the median below every big launch seen.
    * spread: Gaussian-kernel weighted quantiles (sigma 1.0) of the
      same hour's log-ratios, re-centred on the local-linear centre.
      The kernel gives ~zero weight to launches far in scale, so a
      mega launch is forecast from the biggest launches only, while a
      small launch is forecast from the small ones.
  The result is lightly smoothed across neighbouring hours, rounded
  half-up to 2 decimals (the starter's rounding), clamped to (0, 1e9]
  with monotonicity repaired per row.

  Measured (local harness, test_local.py + work/exp_v2..v5.py):
    practice mean (4 SullyGnome cases): 0.206 vs v1 0.325 vs baseline
      0.387 -- wins on all four cases individually (0.221 / 0.260 /
      0.148 / 0.197 vs v1's 0.405 / 0.289 / 0.288 / 0.317).
    leave-one-out over the 38 comparables: 0.386 vs v1 0.713 vs
      baseline 0.930; on big launches (hour-1 >= 50k): 0.204 vs v1
      0.361 vs baseline 0.251; on mega launches (>= 150k): 0.200 vs
      v1 0.408 vs baseline 0.253. The scale-aware model is the first
      version that beats the baseline on big launches at all.
    mega-launch sanity (hour-1 = 2,000,000, p00's comparables):
      median ratio 0.93x at hour 19, 0.89x at hour 21; 95th 1.35x /
      1.50x; 5th 0.83x / 0.84x -- inside the observed big-launch
      range (median 1.31x, 5th 0.84x, 95th 2.60x at hour 19), where
      v1 forecast 2.23x median and a 352x 95th.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0)
in python:3.13-slim, standard library only: no network, no pip packages.
-P keeps this file's directory OFF sys.path; -s drops user site-packages.
This file is self-contained (no sibling modules), so neither matters.

What it must do:
  - read every <id>.in in the input directory (the grader runs one case
    at a time, with only that case's file there): JSON
    {"schema": "makefun/twitch-ccu-case/v1", "id", "game", "day_start",
    "checkpoint", "levels", "known", "forecast", "history",
    "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until,
    viewers, channels, reads} and `forecast` lists the hours to
    forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2,
    "q": [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours
    2..24 each once, no other key, each q five finite numbers above 0
    and at most 1e9 (viewers; fractions allowed), non-decreasing; at
    most 65,536 bytes. Anything else is an invalid case, scored as if
    every quantile equalled hour 1's print, and an entry with an
    invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Fallback levels if a case does not carry its own.
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# A comparable only counts if its release day is this complete: hour 1
# present and at least this many hourly points on the day.
MIN_DAY_POINTS = 20
# Below this many usable comparables the empirical quantiles are too
# noisy to trust; fall back to the house baseline shape.
MIN_COMPARABLES = 5
# House baseline spread (log units per sqrt hour), used only as fallback.
BASELINE_SIGMA = 0.15
# Standard normal quantile at each scored level.
Z_BY_LEVEL = {0.05: -1.645, 0.25: -0.674, 0.5: 0.0, 0.75: 0.674, 0.95: 1.645}
# Gaussian kernel bandwidths in log(hour-1) units. The centre uses a
# wider kernel (a smooth trend needs more data); the spread uses a
# narrower one (the conditional distribution changes fast with scale).
# Both optima are flat (sigma 0.75-1.25 for the spread and 1.0-1.5 for
# the centre score within 0.005 of each other on leave-one-out), so
# these are not knife-edge tunings.
CENTRE_SIGMA = 1.5
SPREAD_SIGMA = 1.0


def round2(x):
    """x rounded to 2 decimals, half up (the starter's rounding).

    A whole number comes back as an int, so json.dumps writes 1240
    where JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def empirical_quantile(sorted_values, tau):
    """Linear-interpolated quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else lo
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def weighted_quantile(pairs, tau):
    """Weighted type-7 quantile of (value, weight) pairs.

    The pairs are sorted by value; the target is tau * total weight,
    linearly interpolated between the bracketing order statistics.
    Deterministic: sorting is stable and the input order is fixed.
    """
    if not pairs:
        return 0.0
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return empirical_quantile([v for v, _ in pairs], tau)
    target = tau * total
    acc = 0.0
    prev_v, prev_cum = pairs[0][0], 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            if w <= 0:
                return v
            frac = (target - prev_cum) / w
            return prev_v + frac * (v - prev_v)
        prev_v, prev_cum = v, acc
    return pairs[-1][0]


def kernel_weight(log_h1, log_p1, sigma):
    """Gaussian kernel weight of a comparable at log(hour-1)."""
    d = (log_h1 - log_p1) / sigma
    return math.exp(-0.5 * d * d)


def local_linear_centre(points, log_p1, sigma):
    """Kernel-weighted local-linear fit of log-ratio on log(hour-1),
    evaluated at log_p1 clamped to the observed log(hour-1) range.

    points: list of (log-ratio, log-hour-1). The clamp is the whole
    point for a mega launch: the scale trend is real inside the data
    but must saturate -- extrapolating the fitted slope past the
    biggest comparable would push the forecast below every big launch
    ever seen. With the clamp, a launch bigger than all comparables
    is forecast like the biggest ones, which is the honest prior.
    """
    if not points:
        return 0.0
    weights = [kernel_weight(x, log_p1, sigma) for _, x in points]
    sw = sum(weights)
    if sw <= 0:
        return sum(y for y, _ in points) / len(points)
    mx = sum(w * x for w, (_, x) in zip(weights, points)) / sw
    my = sum(w * y for w, (y, _) in zip(weights, points)) / sw
    var = sum(w * (x - mx) ** 2 for w, (_, x) in zip(weights, points))
    if var <= 1e-12:
        return my
    cov = sum(w * (x - mx) * (y - my)
              for w, (y, x) in zip(weights, points))
    slope = cov / var
    intercept = my - slope * mx
    xs = [x for _, x in points]
    x_pred = min(max(log_p1, min(xs)), max(xs))
    return intercept + slope * x_pred


def smooth_across_hours(values_by_hour, hours):
    """3-point moving average of one quantile level across hours.

    At the ends the missing neighbour is replaced by the hour's own
    value, so the ends move two-thirds of the way toward their single
    neighbour instead of staying put.
    """
    out = {}
    for h in hours:
        neighbours = [
            values_by_hour.get(h - 1, values_by_hour[h]),
            values_by_hour[h],
            values_by_hour.get(h + 1, values_by_hour[h]),
        ]
        out[h] = sum(neighbours) / 3.0
    return out


def release_day_curve(comparable):
    """{hour: viewers} for the comparable's release day (UTC day of
    day_start), from its hourly points. Empty when the day is missing."""
    day = comparable.get("day_start", "")[:10]
    if not day:
        return {}
    curve = {}
    for point in comparable.get("hourly") or []:
        stamp = point.get("from", "")
        if stamp[:10] != day:
            continue
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        hour = int(stamp[11:13]) + 1  # 00:00 -> hour 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    return curve


def usable_comparables(case):
    """Release-day curves usable as shape donors, by name order."""
    curves = {}
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        curve = release_day_curve(comp)
        if 1 in curve and len(curve) >= MIN_DAY_POINTS:
            curves[name] = curve
    return curves


def hour_quantiles(case, p1, hours, levels):
    """Log-quantile offsets per hour, conditional on hour-1 size.

    Returns {hour: [log offset at each level]} so the caller anchors
    them on hour 1's print. Falls back to the house baseline shape
    when too few comparables are usable.
    """
    curves = usable_comparables(case)
    if len(curves) < MIN_COMPARABLES:
        offsets = {}
        for h in hours:
            offsets[h] = [
                Z_BY_LEVEL.get(tau, 0.0) * BASELINE_SIGMA * math.sqrt(h - 1)
                for tau in levels
            ]
        return offsets
    log_p1 = math.log(p1)
    raw = {}
    for h in hours:
        # (log-ratio to hour 1, log hour-1) for every comparable that
        # has this hour.
        points = []
        for curve in curves.values():
            if h in curve:
                points.append((math.log(curve[h] / curve[1]),
                               math.log(curve[1])))
        if len(points) < MIN_COMPARABLES:
            # Too thin for the local fit: plain empirical quantiles.
            ordered = sorted(y for y, _ in points)
            raw[h] = [empirical_quantile(ordered, tau) for tau in levels]
            continue
        # Centre: local-linear trend at this case's scale (clamped).
        centre = local_linear_centre(points, log_p1, CENTRE_SIGMA)
        # Spread: kernel-weighted quantiles of the same log-ratios,
        # re-centred on the local-linear centre.
        pairs = [(y, kernel_weight(x, log_p1, SPREAD_SIGMA))
                 for y, x in points]
        spread = [weighted_quantile(pairs, tau) for tau in levels]
        median = weighted_quantile(pairs, 0.5)
        raw[h] = [centre + (q - median) for q in spread]
    # Smooth each level's curve across neighbouring hours.
    for i in range(len(levels)):
        smoothed = smooth_across_hours(
            {h: raw[h][i] for h in hours}, hours
        )
        for h in hours:
            raw[h][i] = smoothed[h]
    # A row's quantiles must not decrease.
    for h in hours:
        row = raw[h]
        for i in range(1, len(row)):
            if row[i] < row[i - 1]:
                row[i] = row[i - 1]
    return raw


def forecast_case(case, name):
    levels = list(case.get("levels") or DEFAULT_LEVELS)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = [int(row["hour"]) for row in rows]
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is None:
        if known:
            first = known[0]
        else:
            raise ValueError("%s has no hour-1 print" % name)
    p1 = max(1.0, float(first.get("viewers") or 1))
    offsets = hour_quantiles(case, p1, hours, levels)
    out_hours = []
    for h in hours:
        qs = []
        for offset in offsets[h]:
            q = p1 * math.exp(offset)
            if q <= 0 or not math.isfinite(q):
                q = p1
            if q > 1e9:
                q = 1e9
            qs.append(round2(q))
        # Rounding and the 1e9 clamp can break monotonicity; repair it.
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def baseline_forecast(case, name):
    """House baseline shape, used only if the model itself fails."""
    levels = list(case.get("levels") or DEFAULT_LEVELS)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    hours = [int(row["hour"]) for row in rows]
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = max(1.0, float((first or {}).get("viewers") or 1))
    z_by_level = Z_BY_LEVEL
    out_hours = []
    for h in hours:
        qs = [
            round2(p1 * math.exp(z_by_level.get(tau, 0.0) * BASELINE_SIGMA * math.sqrt(h - 1)))
            for tau in levels
        ]
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = baseline_forecast(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
