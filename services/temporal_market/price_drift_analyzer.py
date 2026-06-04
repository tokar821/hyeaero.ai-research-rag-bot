"""Deterministic price drift from historical listing medians (OLS slope, no ML)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

from services.temporal_market.price_history import PriceHistorySeries

SECONDS_PER_DAY = 86400.0
WINDOW_30D = 30
WINDOW_90D = 90
WINDOW_365D = 365
FLAT_THRESHOLD_PCT = 1.5


class TrendDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


@dataclass(frozen=True)
class PriceDriftReport:
    drift_30d_pct: Optional[float]
    drift_90d_pct: Optional[float]
    drift_1y_pct: Optional[float]
    trend_direction: TrendDirection
    volatility_index: int
    slope_pct_per_90d: Optional[float]
    insufficient_history: bool = True


def _parse_series_ts(series: PriceHistorySeries) -> List[tuple[float, float]]:
    out: List[tuple[float, float]] = []
    for iso, price in zip(series.timestamps, series.prices):
        try:
            from datetime import datetime, timezone

            ts = datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
            out.append((ts, float(price)))
        except (ValueError, TypeError):
            continue
    out.sort(key=lambda x: x[0])
    return out


def _ols_slope_pct_per_day(points: Sequence[tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    xs = [p[0] / SECONDS_PER_DAY for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    if y_mean <= 0:
        return None
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return None
    slope_usd_per_day = num / den
    return (slope_usd_per_day / y_mean) * 100.0


def _window_drift_pct(points: Sequence[tuple[float, float]], window_days: int) -> Optional[float]:
    if len(points) < 2:
        return None
    end_ts = points[-1][0]
    start_cutoff = end_ts - window_days * SECONDS_PER_DAY
    window = [p for p in points if p[0] >= start_cutoff]
    if len(window) < 2:
        window = points[:2]
    p0 = window[0][1]
    p1 = window[-1][1]
    if p0 <= 0:
        return None
    return ((p1 - p0) / p0) * 100.0


def _volatility_index(points: Sequence[tuple[float, float]]) -> int:
    if len(points) < 3:
        return 0
    returns: List[float] = []
    for i in range(1, len(points)):
        p0, p1 = points[i - 1][1], points[i][1]
        if p0 > 0:
            returns.append((p1 - p0) / p0)
    if not returns:
        return 0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = math.sqrt(var)
    # Scale: std ~0.05 -> index ~50, cap 100
    idx = int(min(100, max(0, round(std * 1000))))
    return idx


def _trend_from_drift(d90: Optional[float], slope_90: Optional[float]) -> TrendDirection:
    ref = d90 if d90 is not None else (slope_90 * 3.0 if slope_90 is not None else None)
    if ref is None:
        return TrendDirection.FLAT
    if ref > FLAT_THRESHOLD_PCT:
        return TrendDirection.UP
    if ref < -FLAT_THRESHOLD_PCT:
        return TrendDirection.DOWN
    return TrendDirection.FLAT


def analyze_price_drift(series: PriceHistorySeries) -> PriceDriftReport:
    """Compute drift windows, trend direction, and volatility index."""
    points = _parse_series_ts(series)
    if series.insufficient_history or len(points) < 5:
        return PriceDriftReport(
            drift_30d_pct=None,
            drift_90d_pct=None,
            drift_1y_pct=None,
            trend_direction=TrendDirection.FLAT,
            volatility_index=0,
            slope_pct_per_90d=None,
            insufficient_history=True,
        )

    d30 = _window_drift_pct(points, WINDOW_30D)
    d90 = _window_drift_pct(points, WINDOW_90D)
    d1y = _window_drift_pct(points, WINDOW_365D)
    slope_day = _ols_slope_pct_per_day(points)
    slope_90 = slope_day * 90.0 if slope_day is not None else None
    vol = _volatility_index(points)

    return PriceDriftReport(
        drift_30d_pct=d30,
        drift_90d_pct=d90,
        drift_1y_pct=d1y,
        trend_direction=_trend_from_drift(d90, slope_90),
        volatility_index=vol,
        slope_pct_per_90d=slope_90,
        insufficient_history=False,
    )
