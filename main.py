"""gta6-twitch-ccu-forecast-01 entry in Python 3.13 (standard library only).

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2..24),
as five quantiles an hour, from everything recorded up to the end of hour 1.

Model (fit at run time from the case file's own comparables, so it adapts
to the 38 practice comparables and the 42 final ones alike):
  For each hour h = 2..24, a log-log regression over comparable launches
      log(viewers_h) = a_h + b1_h * log(p1) + b2_h * log(pre7) + b3_h * log(ch1)
  where p1 is the launch's hour-1 viewers, pre7 its mean hourly viewers over
  the 7 days before release day, and ch1 its live channels in hour 1.
  Hour 1 alone is a noisy anchor (its share of the day's unlock varies by
  region), so the pre-release baseline and the channel count add real signal:
  on the four practice cases the three-predictor fit beats hour-1 alone.
  The fit uses only the K=30 comparables nearest the target in log(p1):
  the full set's smallest launches (hour-1 in the hundreds) have ratios that
  explode and flatten the fit, and a target as big as GTA VI sits past the
  largest comparable, where the global fit collapses its later hours to a
  fifth of hour 1; the 30 nearest keep the fit local, so a huge hour-1
  still forecasts a sensible curve (about 0.6x at hour 19 for a 1.5M
  hour-1, not 0.2x).
  The forecast median is the regression prediction at the target's own
  (p1, pre7, ch1); the spread is the regression's residual std, scaled by
  0.5 (the comparables' residuals overstate the spread for big launches:
  unscaled, the 90% band covers ~99% of practice hours; scaled to 0.5 it
  covers ~0.90, the honest level, and the practice loss is lowest there).
  Quantiles are normal: q_tau = exp(mu + z_tau * 0.5 * sigma_h).

  Fallbacks, in order, if a fit fails: fewer predictors (p1, pre7), then
  (p1) alone, then the house persistence baseline; with fewer than 10
  usable comparables it goes straight to the baseline. Every quantile is
  clamped to (0, 1e9] and rounded to 2 decimals, half up, non-decreasing.

Practice scores (local scorer, pinball loss on log scale; lower is better):
  baseline (starter persistence)  0.387097 overall
  this model                     0.150995 overall
    p00 Wolverine 0.1386, p01 RDR2 0.1383, p02 GoW 0.1362, p03 Spider-Man 0.1908

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
Reads every <id>.in in the input directory and writes <id>.out per case.
Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Residual-spread scale (see module docstring): the comparables' residual std
# overstates the spread for big launches; 0.5 is the practice optimum and
# puts 90% coverage at ~0.90, the honest level.
SPREAD_SCALE = 0.5
# Pre-release window (days before release day) for the baseline predictor.
PRE_DAYS = 7
# How many comparables nearest the target (in log hour-1) the fit uses.
K_NEAREST = 30
# Minimum usable comparables before falling back to the baseline.
MIN_ROWS = 10
# Quantile cap required by the rules.
Q_MAX = 1e9


def parse_ts(t):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def ols(Xs, Y):
    """Least squares Y ~ 1 + Xs[0] + Xs[1] + ... (centered normal equations).

    Xs is a list of predictor lists (each length n); Y is the target list.
    Returns (intercept, [slopes], residuals) or None if singular.
    Deterministic: fixed order, partial pivoting, no randomness.
    """
    n = len(Y)
    k = len(Xs)
    means = [statistics.mean(x) for x in Xs]
    my = statistics.mean(Y)
    # Normal equations on centered predictors, with an intercept row/col.
    A = [[0.0] * (k + 1) for _ in range(k + 1)]
    rhs = [0.0] * (k + 1)
    for i in range(n):
        row = [1.0] + [Xs[j][i] - means[j] for j in range(k)]
        t = Y[i] - my
        for a in range(k + 1):
            rhs[a] += row[a] * t
            for b in range(k + 1):
                A[a][b] += row[a] * row[b]
    # Gaussian elimination with partial pivoting.
    M = [row[:] + [rhs[i]] for i, row in enumerate(A)]
    for col in range(k + 1):
        piv = max(range(col, k + 1), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(k + 1):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / M[col][col]
                for c in range(col, k + 2):
                    M[r][c] -= f * M[col][c]
    sol = [M[i][k + 1] / M[i][i] for i in range(k + 1)]
    slopes = sol[1:]
    intercept = my - sum(slopes[j] * means[j] for j in range(k))
    resid = [
        Y[i] - (intercept + sum(slopes[j] * Xs[j][i] for j in range(k)))
        for i in range(n)
    ]
    return intercept, slopes, resid


def comparable_rows(case):
    """One row per comparable with a full release day: predictors + log curve.

    A row is (log p1, log pre7, log channels-in-hour-1, {hour: log viewers}).
    Comparables missing hour 1, any release-day hour, or the pre-release
    window are skipped; with too few rows the caller falls back.
    """
    rows = []
    for comp in case.get("comparables", []):
        try:
            ds = parse_ts(comp["day_start"])
        except (KeyError, ValueError):
            continue
        hourly = {}
        channels = {}
        for pt in comp.get("hourly", []):
            try:
                t = parse_ts(pt["from"])
                v = pt["viewers"]
            except (KeyError, ValueError):
                continue
            if v is not None and v > 0:
                hourly[t] = v
                ch = pt.get("channels")
                if ch is not None and ch > 0:
                    channels[t] = ch
        p1 = hourly.get(ds)
        if not p1 or p1 <= 0:
            continue
        pre = [
            v for t, v in hourly.items()
            if ds - timedelta(days=PRE_DAYS) <= t < ds
        ]
        if len(pre) < 12:
            continue
        day = {}
        ok = True
        for h in range(1, 25):
            v = hourly.get(ds + timedelta(hours=h - 1))
            if not v or v <= 0:
                ok = False
                break
            day[h] = math.log(v)
        if not ok:
            continue
        rows.append((
            math.log(p1),
            math.log(statistics.mean(pre)),
            math.log(max(1, channels.get(ds, 0))),
            day,
        ))
    return rows


def target_features(case):
    """The target case's (log p1, log pre7, log channels-in-hour-1)."""
    known = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if known is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, known["viewers"])
    try:
        ds = parse_ts(case["day_start"])
    except (KeyError, ValueError):
        ds = None
    lpre = math.log(p1)
    if ds is not None:
        hist = {}
        for pt in case.get("history_hourly", []):
            try:
                t = parse_ts(pt["from"])
                v = pt["viewers"]
            except (KeyError, ValueError):
                continue
            if v is not None and v > 0:
                hist[t] = v
        pre = [
            v for t, v in hist.items()
            if ds - timedelta(days=PRE_DAYS) <= t < ds
        ]
        if len(pre) >= 12:
            lpre = math.log(statistics.mean(pre))
    ch1 = known.get("channels") or 0
    return math.log(p1), lpre, math.log(max(1, ch1))


def baseline_quantiles(p1, h):
    """House persistence baseline: every hour at hour 1, spread widening."""
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    lp1, lpre, lch1 = target_features(case)
    p1 = math.exp(lp1)
    target = (lp1, lpre, lch1)
    rows = comparable_rows(case)
    # Keep only the K comparables nearest the target in log hour-1; the
    # smallest launches distort the fit and a huge target sits past the
    # largest comparable, where a global fit collapses (see docstring).
    if len(rows) > K_NEAREST:
        rows = sorted(rows, key=lambda r: abs(r[0] - lp1))[:K_NEAREST]
    # Predictor sets to try, most informative first.
    pred_sets = [(0, 1, 2), (0, 1), (0,)]
    hours = []
    for row in case.get("forecast", []):
        h = row["hour"]
        quantiles = None
        if len(rows) >= MIN_ROWS:
            for pred_idx in pred_sets:
                Xs = [[r[i] for r in rows] for i in pred_idx]
                Y = [r[3][h] for r in rows]
                fit = ols(Xs, Y)
                if fit is None:
                    continue
                a, slopes, resid = fit
                mu = a + sum(
                    slopes[j] * target[pred_idx[j]] for j in range(len(pred_idx))
                )
                sd = statistics.pstdev(resid) if len(resid) > 1 else 0.0
                quantiles = [mu + z * sd * SPREAD_SCALE for z in Z]
                break
        if quantiles is None:
            quantiles = [math.log(max(1.0, q)) for q in baseline_quantiles(p1, h)]
        # Clamp to (0, 1e9], round to 2 decimals half up, keep non-decreasing.
        qs = []
        prev = 0.0
        for q in quantiles:
            v = math.exp(q)
            if not math.isfinite(v) or v <= 0:
                v = 1.0
            v = min(v, Q_MAX)
            v = round2(v)
            if v < prev:
                v = prev
            qs.append(v)
            prev = float(v)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
