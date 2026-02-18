"""Unit tests for scalper.signals module."""

from __future__ import annotations

from datetime import datetime

import pytest
import pytz

from scalper.indicators import IndicatorState
from scalper.signals import Signal, SignalEngine, SignalResult
from scalper.utils import TimeFilter

ET = pytz.timezone("US/Eastern")


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> dict:
    """Minimal config for signal engine tests."""
    return {
        "entry": {
            "pullback_distance": 2.0,
            "cvd_lookback_bars": 3,
            "volume_spike_multiplier": 1.3,
            "atr_min": 1.5,
            "atr_max": 8.0,
        },
        "exits": {
            "stop_atr_multiplier": 1.2,
            "stop_min_points": 4.0,
            "stop_max_points": 10.0,
            "tp1_atr_multiplier": 1.0,
            "tp1_pct": 0.50,
            "tp2_atr_multiplier": 2.0,
            "tp2_pct": 0.50,
            "trail_atr_multiplier": 0.8,
            "breakeven_atr_multiplier": 0.5,
            "breakeven_offset_ticks": 1,
            "max_trade_duration_seconds": 300,
            "eod_close_time": "15:50",
        },
        "time_filter": {
            "timezone": "US/Eastern",
            "sessions": [
                {"start": "09:35", "end": "11:30"},
                {"start": "13:30", "end": "15:45"},
            ],
            "avoid_after_open_minutes": 5,
            "avoid_before_close_minutes": 15,
            "news_blackout_minutes": 5,
            "economic_calendar_file": "",
        },
    }


@pytest.fixture
def time_filter(cfg: dict) -> TimeFilter:
    return TimeFilter(cfg)


@pytest.fixture
def engine(cfg: dict, time_filter: TimeFilter) -> SignalEngine:
    return SignalEngine(cfg, time_filter)


def _trading_time() -> datetime:
    """Return a datetime that falls within the allowed trading window."""
    return ET.localize(datetime(2024, 6, 3, 10, 0, 0))


def _make_long_state() -> IndicatorState:
    """Create an indicator state where ALL long conditions pass."""
    return IndicatorState(
        vwap=18000.0,
        vwap_upper_1=18010.0,
        vwap_upper_2=18020.0,
        vwap_lower_1=17990.0,
        vwap_lower_2=17980.0,
        vwap_std=5.0,
        ema_fast=18002.0,      # EMA9 > EMA21
        ema_slow=18000.0,
        atr=3.0,               # Within 1.5-8.0 range
        cvd_rising=True,
        cvd_falling=False,
        cvd_cumulative=500.0,
        volume_sma=100.0,
        current_volume=150.0,
        volume_is_spike=True,  # 150 >= 1.3 * 100
        last_1min_bar_green=True,
        last_1min_bar_red=False,
        current_price=18001.0,  # Above VWAP, within 2 pts
    )


def _make_short_state() -> IndicatorState:
    """Create an indicator state where ALL short conditions pass."""
    return IndicatorState(
        vwap=18000.0,
        vwap_upper_1=18010.0,
        vwap_upper_2=18020.0,
        vwap_lower_1=17990.0,
        vwap_lower_2=17980.0,
        vwap_std=5.0,
        ema_fast=17998.0,      # EMA9 < EMA21
        ema_slow=18000.0,
        atr=3.0,
        cvd_rising=False,
        cvd_falling=True,
        cvd_cumulative=-500.0,
        volume_sma=100.0,
        current_volume=150.0,
        volume_is_spike=True,
        last_1min_bar_green=False,
        last_1min_bar_red=True,
        current_price=17999.0,  # Below VWAP, within 2 pts
    )


# ─────────────────────────────────────────────────────────────
# Long signal tests
# ─────────────────────────────────────────────────────────────

class TestLongSignal:
    def test_all_conditions_met(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal == Signal.BUY
        assert not result.failed

    def test_price_below_vwap_fails(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.current_price = 17998.0  # Below VWAP
        result = engine.evaluate(state, dt=_trading_time())
        # Should not get BUY (price location fails for long)
        # Might trigger SHORT checks too but those will also fail
        assert result.signal != Signal.BUY

    def test_no_pullback_fails(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.current_price = 18010.0  # Too far from VWAP/EMA21
        state.ema_slow = 18000.0
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal == Signal.HOLD

    def test_ema_crossover_wrong_direction(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.ema_fast = 17999.0  # EMA9 < EMA21
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY

    def test_1min_bar_not_green(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.last_1min_bar_green = False
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY

    def test_cvd_not_rising(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.cvd_rising = False
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY

    def test_no_volume_spike(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.volume_is_spike = False
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY

    def test_atr_too_low(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.atr = 1.0  # Below minimum 1.5
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY

    def test_atr_too_high(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        state.atr = 9.0  # Above maximum 8.0
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.BUY


# ─────────────────────────────────────────────────────────────
# Short signal tests
# ─────────────────────────────────────────────────────────────

class TestShortSignal:
    def test_all_conditions_met(self, engine: SignalEngine) -> None:
        state = _make_short_state()
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal == Signal.SELL
        assert not result.failed

    def test_price_above_vwap_fails(self, engine: SignalEngine) -> None:
        state = _make_short_state()
        state.current_price = 18001.0  # Above VWAP
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.SELL

    def test_ema_crossover_wrong_direction(self, engine: SignalEngine) -> None:
        state = _make_short_state()
        state.ema_fast = 18001.0  # EMA9 > EMA21
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.SELL

    def test_1min_bar_not_red(self, engine: SignalEngine) -> None:
        state = _make_short_state()
        state.last_1min_bar_red = False
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.SELL

    def test_cvd_not_falling(self, engine: SignalEngine) -> None:
        state = _make_short_state()
        state.cvd_falling = False
        result = engine.evaluate(state, dt=_trading_time())
        assert result.signal != Signal.SELL


# ─────────────────────────────────────────────────────────────
# Time filter tests
# ─────────────────────────────────────────────────────────────

class TestTimeFilter:
    def test_outside_session(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        # 8:00 AM ET — before any session
        dt = ET.localize(datetime(2024, 6, 3, 8, 0, 0))
        result = engine.evaluate(state, dt=dt)
        assert result.signal == Signal.HOLD

    def test_during_lunch_gap(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        # 12:00 PM ET — between sessions
        dt = ET.localize(datetime(2024, 6, 3, 12, 0, 0))
        result = engine.evaluate(state, dt=dt)
        assert result.signal == Signal.HOLD

    def test_after_close(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        # 16:00 ET
        dt = ET.localize(datetime(2024, 6, 3, 16, 0, 0))
        result = engine.evaluate(state, dt=dt)
        assert result.signal == Signal.HOLD

    def test_within_first_session(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        dt = ET.localize(datetime(2024, 6, 3, 10, 0, 0))
        result = engine.evaluate(state, dt=dt)
        assert result.signal == Signal.BUY

    def test_within_second_session(self, engine: SignalEngine) -> None:
        state = _make_long_state()
        dt = ET.localize(datetime(2024, 6, 3, 14, 0, 0))
        result = engine.evaluate(state, dt=dt)
        assert result.signal == Signal.BUY


# ─────────────────────────────────────────────────────────────
# SignalResult
# ─────────────────────────────────────────────────────────────

class TestSignalResult:
    def test_hold_default(self) -> None:
        sr = SignalResult()
        assert sr.signal == Signal.HOLD
        assert sr.reasons == []
        assert sr.failed == []

    def test_reasons_tracked(self) -> None:
        sr = SignalResult(signal=Signal.BUY, reasons=["a", "b"])
        assert len(sr.reasons) == 2
