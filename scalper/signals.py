"""Entry and exit signal logic.

Evaluates all entry conditions (long/short) against the current
:class:`~scalper.indicators.IndicatorState` and returns a signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from scalper.indicators import IndicatorState
from scalper.utils import TimeFilter

logger = logging.getLogger(__name__)


class Signal(Enum):
    """Trade signal produced by the signal engine."""
    HOLD = auto()
    BUY = auto()
    SELL = auto()


@dataclass
class SignalResult:
    """Result of a signal evaluation.

    Attributes:
        signal: The trade direction (BUY / SELL / HOLD).
        reasons: Human-readable list of which conditions passed.
        failed: List of conditions that did NOT pass (for debugging).
    """
    signal: Signal = Signal.HOLD
    reasons: list[str] | None = None
    failed: list[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = []
        if self.failed is None:
            self.failed = []


class SignalEngine:
    """Evaluates entry conditions and produces BUY / SELL / HOLD signals.

    All configurable thresholds are read from ``cfg``.

    Args:
        cfg: Full configuration dictionary.
        time_filter: An initialised :class:`~scalper.utils.TimeFilter`.
    """

    def __init__(self, cfg: dict[str, Any], time_filter: TimeFilter) -> None:
        self._entry = cfg["entry"]
        self._exits = cfg["exits"]
        self._time_filter = time_filter

        self._pullback_dist: float = self._entry["pullback_distance"]
        self._atr_min: float = self._entry["atr_min"]
        self._atr_max: float = self._entry["atr_max"]
        self._vol_spike_mult: float = self._entry["volume_spike_multiplier"]

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def evaluate(self, state: IndicatorState, dt=None) -> SignalResult:
        """Evaluate all entry conditions and return a signal.

        Args:
            state: Current indicator snapshot.
            dt: Optional datetime for time-filter checks (defaults to now).

        Returns:
            A :class:`SignalResult`.
        """
        long = self._check_long(state, dt)
        if long.signal == Signal.BUY:
            return long

        short = self._check_short(state, dt)
        if short.signal == Signal.SELL:
            return short

        # Merge failure reasons from both for debug
        all_failed = (long.failed or []) + (short.failed or [])
        return SignalResult(signal=Signal.HOLD, failed=all_failed)

    # ─────────────────────────────────────────────────────────
    # Long conditions
    # ─────────────────────────────────────────────────────────

    def _check_long(self, s: IndicatorState, dt=None) -> SignalResult:
        """Check all LONG entry conditions.

        Args:
            s: Indicator state.
            dt: Datetime for time filter.

        Returns:
            SignalResult with BUY or HOLD.
        """
        reasons: list[str] = []
        failed: list[str] = []

        # (a) Price location
        above_vwap = s.current_price > s.vwap
        near_vwap = abs(s.current_price - s.vwap) <= self._pullback_dist
        near_ema21 = abs(s.current_price - s.ema_slow) <= self._pullback_dist
        pullback = near_vwap or near_ema21

        if above_vwap and pullback:
            reasons.append("price_above_vwap_and_pullback")
        else:
            if not above_vwap:
                failed.append("price_not_above_vwap")
            if not pullback:
                failed.append("no_pullback_to_vwap_or_ema21")

        # (b) Momentum confirmation
        ema_cross = s.ema_fast > s.ema_slow
        bar_green = s.last_1min_bar_green

        if ema_cross:
            reasons.append("ema9_above_ema21")
        else:
            failed.append("ema9_not_above_ema21")

        if bar_green:
            reasons.append("1min_bar_green")
        else:
            failed.append("1min_bar_not_green")

        # (c) Order flow confirmation
        if s.cvd_rising:
            reasons.append("cvd_rising")
        else:
            failed.append("cvd_not_rising")

        if s.volume_is_spike:
            reasons.append("volume_spike")
        else:
            failed.append("no_volume_spike")

        # (d) Volatility filter
        atr_ok = self._atr_min <= s.atr <= self._atr_max
        if atr_ok:
            reasons.append("atr_in_range")
        else:
            failed.append(f"atr_out_of_range({s.atr:.2f})")

        # (e) Time filter
        time_ok = self._time_filter.is_trading_allowed(dt)
        if time_ok:
            reasons.append("time_ok")
        else:
            failed.append("outside_trading_window")

        if not failed:
            logger.info("LONG signal fired: %s", reasons)
            return SignalResult(signal=Signal.BUY, reasons=reasons, failed=[])
        return SignalResult(signal=Signal.HOLD, reasons=reasons, failed=failed)

    # ─────────────────────────────────────────────────────────
    # Short conditions
    # ─────────────────────────────────────────────────────────

    def _check_short(self, s: IndicatorState, dt=None) -> SignalResult:
        """Check all SHORT entry conditions.

        Args:
            s: Indicator state.
            dt: Datetime for time filter.

        Returns:
            SignalResult with SELL or HOLD.
        """
        reasons: list[str] = []
        failed: list[str] = []

        # (a) Price location — below VWAP, pulled back toward VWAP or EMA21
        below_vwap = s.current_price < s.vwap
        near_vwap = abs(s.current_price - s.vwap) <= self._pullback_dist
        near_ema21 = abs(s.current_price - s.ema_slow) <= self._pullback_dist
        pullback = near_vwap or near_ema21

        if below_vwap and pullback:
            reasons.append("price_below_vwap_and_pullback")
        else:
            if not below_vwap:
                failed.append("price_not_below_vwap")
            if not pullback:
                failed.append("no_pullback_to_vwap_or_ema21")

        # (b) Momentum
        ema_cross = s.ema_fast < s.ema_slow
        bar_red = s.last_1min_bar_red

        if ema_cross:
            reasons.append("ema9_below_ema21")
        else:
            failed.append("ema9_not_below_ema21")

        if bar_red:
            reasons.append("1min_bar_red")
        else:
            failed.append("1min_bar_not_red")

        # (c) Order flow
        if s.cvd_falling:
            reasons.append("cvd_falling")
        else:
            failed.append("cvd_not_falling")

        if s.volume_is_spike:
            reasons.append("volume_spike")
        else:
            failed.append("no_volume_spike")

        # (d) Volatility
        atr_ok = self._atr_min <= s.atr <= self._atr_max
        if atr_ok:
            reasons.append("atr_in_range")
        else:
            failed.append(f"atr_out_of_range({s.atr:.2f})")

        # (e) Time
        time_ok = self._time_filter.is_trading_allowed(dt)
        if time_ok:
            reasons.append("time_ok")
        else:
            failed.append("outside_trading_window")

        if not failed:
            logger.info("SHORT signal fired: %s", reasons)
            return SignalResult(signal=Signal.SELL, reasons=reasons, failed=[])
        return SignalResult(signal=Signal.HOLD, reasons=reasons, failed=failed)
