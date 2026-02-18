"""Real-time indicator calculations: VWAP, EMA, ATR, CVD, Volume SMA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ─────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """Represents a single price bar (time-based or tick-based).

    Attributes:
        timestamp: Bar open time (epoch seconds or datetime).
        open: Opening price.
        high: Highest price during the bar.
        low: Lowest price during the bar.
        close: Closing price.
        volume: Total volume traded in the bar.
        buy_volume: Volume traded at the ask (estimated).
        sell_volume: Volume traded at the bid (estimated).
    """
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float = 0.0
    sell_volume: float = 0.0


# ─────────────────────────────────────────────────────────────
# VWAP (Volume-Weighted Average Price)
# ─────────────────────────────────────────────────────────────

class VWAP:
    """Session VWAP with configurable standard-deviation bands.

    Resets at RTH open (call :meth:`reset`).  Feed it each bar via
    :meth:`update` and read out :attr:`value`, :attr:`upper_bands`,
    and :attr:`lower_bands`.

    Args:
        band_multipliers: Iterable of σ multipliers for bands (e.g. ``[1, 2]``).
    """

    def __init__(self, band_multipliers: list[float] | None = None) -> None:
        self.band_multipliers: list[float] = band_multipliers or [1.0, 2.0]
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators (call at session open)."""
        self._cum_vol: float = 0.0
        self._cum_pv: float = 0.0
        self._cum_pv2: float = 0.0
        self.value: float = 0.0
        self.std: float = 0.0
        self.upper_bands: list[float] = [0.0] * len(self.band_multipliers)
        self.lower_bands: list[float] = [0.0] * len(self.band_multipliers)

    def update(self, bar: Bar) -> None:
        """Incorporate a new bar into the VWAP calculation.

        Args:
            bar: The latest completed bar.
        """
        typical = (bar.high + bar.low + bar.close) / 3.0
        vol = bar.volume
        if vol <= 0:
            return

        self._cum_vol += vol
        self._cum_pv += typical * vol
        self._cum_pv2 += (typical ** 2) * vol

        self.value = self._cum_pv / self._cum_vol

        # Variance = E[X²] - (E[X])²
        variance = (self._cum_pv2 / self._cum_vol) - (self.value ** 2)
        self.std = math.sqrt(max(variance, 0.0))

        for i, m in enumerate(self.band_multipliers):
            self.upper_bands[i] = self.value + m * self.std
            self.lower_bands[i] = self.value - m * self.std


# ─────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average)
# ─────────────────────────────────────────────────────────────

class EMA:
    """Incremental Exponential Moving Average.

    Args:
        period: Lookback period for the EMA.
    """

    def __init__(self, period: int) -> None:
        self.period: int = period
        self.multiplier: float = 2.0 / (period + 1)
        self.value: float = 0.0
        self._count: int = 0
        self._sum: float = 0.0

    def reset(self) -> None:
        """Reset the EMA state."""
        self.value = 0.0
        self._count = 0
        self._sum = 0.0

    def update(self, price: float) -> float:
        """Feed a new price and return the updated EMA value.

        For the first *period* values, a simple average (SMA) is used
        to seed the EMA.

        Args:
            price: Closing price of the latest bar.

        Returns:
            Current EMA value.
        """
        self._count += 1
        if self._count <= self.period:
            self._sum += price
            self.value = self._sum / self._count
        else:
            self.value = (price - self.value) * self.multiplier + self.value
        return self.value


# ─────────────────────────────────────────────────────────────
# ATR (Average True Range)
# ─────────────────────────────────────────────────────────────

class ATR:
    """Incremental Average True Range (Wilder smoothing).

    Args:
        period: Lookback period.
    """

    def __init__(self, period: int = 14) -> None:
        self.period: int = period
        self.value: float = 0.0
        self._prev_close: float | None = None
        self._count: int = 0
        self._tr_sum: float = 0.0

    def reset(self) -> None:
        """Reset the ATR state."""
        self.value = 0.0
        self._prev_close = None
        self._count = 0
        self._tr_sum = 0.0

    def update(self, bar: Bar) -> float:
        """Feed a new bar and return the updated ATR value.

        Args:
            bar: Latest completed bar.

        Returns:
            Current ATR value.
        """
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._prev_close = bar.close
        self._count += 1

        if self._count <= self.period:
            self._tr_sum += tr
            self.value = self._tr_sum / self._count
        else:
            # Wilder smoothing
            self.value = (self.value * (self.period - 1) + tr) / self.period

        return self.value


# ─────────────────────────────────────────────────────────────
# CVD (Cumulative Volume Delta)
# ─────────────────────────────────────────────────────────────

class CVD:
    """Cumulative Volume Delta tracker.

    Delta per bar = buy_volume - sell_volume.  The running sum is
    tracked since session open.

    Keeps a rolling window of per-bar deltas for trend detection.

    Args:
        lookback: Number of recent bar deltas to retain for trend checks.
    """

    def __init__(self, lookback: int = 3) -> None:
        self.lookback: int = lookback
        self.reset()

    def reset(self) -> None:
        """Reset at session open."""
        self.cumulative: float = 0.0
        self._bar_deltas: list[float] = []

    def update(self, bar: Bar) -> float:
        """Add a bar's volume delta.

        Args:
            bar: Latest completed bar with ``buy_volume`` / ``sell_volume``.

        Returns:
            Updated cumulative delta.
        """
        delta = bar.buy_volume - bar.sell_volume
        self.cumulative += delta
        self._bar_deltas.append(delta)
        if len(self._bar_deltas) > self.lookback + 10:
            self._bar_deltas = self._bar_deltas[-(self.lookback + 10):]
        return self.cumulative

    def is_rising(self, n: int | None = None) -> bool:
        """Check whether CVD has risen over the last *n* bars.

        The check verifies that each successive cumulative delta snapshot
        is greater than the prior one over the last *n* bar deltas.

        Args:
            n: Number of bars to look back (defaults to ``self.lookback``).

        Returns:
            ``True`` if CVD is rising.
        """
        n = n or self.lookback
        if len(self._bar_deltas) < n:
            return False
        recent = self._bar_deltas[-n:]
        # Build cumulative snapshots from recent deltas
        cum = 0.0
        snapshots: list[float] = []
        for d in recent:
            cum += d
            snapshots.append(cum)
        return all(snapshots[i] < snapshots[i + 1] for i in range(len(snapshots) - 1))

    def is_falling(self, n: int | None = None) -> bool:
        """Check whether CVD has fallen over the last *n* bars.

        Args:
            n: Number of bars to look back (defaults to ``self.lookback``).

        Returns:
            ``True`` if CVD is falling.
        """
        n = n or self.lookback
        if len(self._bar_deltas) < n:
            return False
        recent = self._bar_deltas[-n:]
        cum = 0.0
        snapshots: list[float] = []
        for d in recent:
            cum += d
            snapshots.append(cum)
        return all(snapshots[i] > snapshots[i + 1] for i in range(len(snapshots) - 1))


# ─────────────────────────────────────────────────────────────
# Volume SMA
# ─────────────────────────────────────────────────────────────

class VolumeSMA:
    """Simple Moving Average of bar volume with spike detection.

    Args:
        period: Number of bars for the SMA window.
    """

    def __init__(self, period: int = 20) -> None:
        self.period: int = period
        self._buffer: list[float] = []
        self.value: float = 0.0

    def reset(self) -> None:
        """Reset state."""
        self._buffer.clear()
        self.value = 0.0

    def update(self, volume: float) -> float:
        """Feed a bar's volume and return the updated SMA.

        Args:
            volume: Volume of the latest completed bar.

        Returns:
            Current volume SMA.
        """
        self._buffer.append(volume)
        if len(self._buffer) > self.period:
            self._buffer.pop(0)
        self.value = sum(self._buffer) / len(self._buffer)
        return self.value

    def is_spike(self, current_volume: float, multiplier: float = 1.3) -> bool:
        """Return True if *current_volume* exceeds the SMA by *multiplier*.

        Args:
            current_volume: Volume of the current (potentially still open) bar.
            multiplier: Required ratio of current volume to SMA.

        Returns:
            ``True`` if volume spike is detected.
        """
        if self.value <= 0:
            return False
        return current_volume >= multiplier * self.value


# ─────────────────────────────────────────────────────────────
# Aggregate indicator state
# ─────────────────────────────────────────────────────────────

@dataclass
class IndicatorState:
    """Snapshot of all indicator values at a point in time.

    Used to pass a consistent set of readings to the signal engine.
    """
    vwap: float = 0.0
    vwap_upper_1: float = 0.0
    vwap_upper_2: float = 0.0
    vwap_lower_1: float = 0.0
    vwap_lower_2: float = 0.0
    vwap_std: float = 0.0
    ema_fast: float = 0.0
    ema_slow: float = 0.0
    atr: float = 0.0
    cvd_rising: bool = False
    cvd_falling: bool = False
    cvd_cumulative: float = 0.0
    volume_sma: float = 0.0
    current_volume: float = 0.0
    volume_is_spike: bool = False
    last_1min_bar_green: bool = False
    last_1min_bar_red: bool = False
    current_price: float = 0.0


class IndicatorEngine:
    """Orchestrates all indicator calculations across both timeframes.

    Args:
        cfg: Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        ind = cfg["indicators"]
        entry = cfg["entry"]

        # Execution timeframe indicators
        self.vwap = VWAP(band_multipliers=ind["vwap_band_multipliers"])
        self.ema_fast = EMA(period=ind["ema_fast"])
        self.ema_slow = EMA(period=ind["ema_slow"])
        self.cvd = CVD(lookback=entry["cvd_lookback_bars"])
        self.volume_sma = VolumeSMA(period=ind["volume_sma_period"])

        # Confirmation timeframe (1-min) indicators
        self.atr_1min = ATR(period=ind["atr_period"])
        self._last_1min_bar: Bar | None = None

        self._volume_spike_mult: float = entry["volume_spike_multiplier"]

    def reset_session(self) -> None:
        """Reset all session-level accumulators (call at RTH open)."""
        self.vwap.reset()
        self.ema_fast.reset()
        self.ema_slow.reset()
        self.cvd.reset()
        self.volume_sma.reset()
        self.atr_1min.reset()
        self._last_1min_bar = None

    def on_exec_bar(self, bar: Bar) -> None:
        """Process a completed execution-timeframe bar.

        Updates VWAP, EMAs, CVD, and volume SMA.

        Args:
            bar: Completed 5-second or tick bar.
        """
        self.vwap.update(bar)
        self.ema_fast.update(bar.close)
        self.ema_slow.update(bar.close)
        self.cvd.update(bar)
        self.volume_sma.update(bar.volume)

    def on_1min_bar(self, bar: Bar) -> None:
        """Process a completed 1-minute confirmation bar.

        Updates the ATR and stores the bar for color checks.

        Args:
            bar: Completed 1-minute bar.
        """
        self.atr_1min.update(bar)
        self._last_1min_bar = bar

    def snapshot(self, current_price: float, current_volume: float) -> IndicatorState:
        """Return a frozen snapshot of all current indicator values.

        Args:
            current_price: Latest tick / mid price.
            current_volume: Volume of the current (open) execution bar.

        Returns:
            :class:`IndicatorState` with all fields populated.
        """
        last_green = False
        last_red = False
        if self._last_1min_bar is not None:
            last_green = self._last_1min_bar.close > self._last_1min_bar.open
            last_red = self._last_1min_bar.close < self._last_1min_bar.open

        return IndicatorState(
            vwap=self.vwap.value,
            vwap_upper_1=self.vwap.upper_bands[0] if self.vwap.upper_bands else 0.0,
            vwap_upper_2=self.vwap.upper_bands[1] if len(self.vwap.upper_bands) > 1 else 0.0,
            vwap_lower_1=self.vwap.lower_bands[0] if self.vwap.lower_bands else 0.0,
            vwap_lower_2=self.vwap.lower_bands[1] if len(self.vwap.lower_bands) > 1 else 0.0,
            vwap_std=self.vwap.std,
            ema_fast=self.ema_fast.value,
            ema_slow=self.ema_slow.value,
            atr=self.atr_1min.value,
            cvd_rising=self.cvd.is_rising(),
            cvd_falling=self.cvd.is_falling(),
            cvd_cumulative=self.cvd.cumulative,
            volume_sma=self.volume_sma.value,
            current_volume=current_volume,
            volume_is_spike=self.volume_sma.is_spike(current_volume, self._volume_spike_mult),
            last_1min_bar_green=last_green,
            last_1min_bar_red=last_red,
            current_price=current_price,
        )
