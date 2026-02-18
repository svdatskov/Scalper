"""Unit tests for scalper.indicators module."""

from __future__ import annotations

import math

import pytest

from scalper.indicators import ATR, CVD, EMA, VWAP, Bar, IndicatorEngine, VolumeSMA


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def make_bar(
    o: float = 100.0,
    h: float = 101.0,
    l: float = 99.0,
    c: float = 100.5,
    vol: float = 100.0,
    buy_vol: float = 60.0,
    sell_vol: float = 40.0,
    ts: float = 0.0,
) -> Bar:
    """Create a Bar with the given values."""
    return Bar(
        timestamp=ts, open=o, high=h, low=l, close=c,
        volume=vol, buy_volume=buy_vol, sell_volume=sell_vol,
    )


# ─────────────────────────────────────────────────────────────
# VWAP
# ─────────────────────────────────────────────────────────────

class TestVWAP:
    def test_single_bar(self) -> None:
        vwap = VWAP(band_multipliers=[1.0, 2.0])
        bar = make_bar(o=100, h=102, l=98, c=100, vol=50)
        vwap.update(bar)

        typical = (102 + 98 + 100) / 3.0
        assert vwap.value == pytest.approx(typical)

    def test_two_bars_weighted(self) -> None:
        vwap = VWAP(band_multipliers=[1.0])
        bar1 = make_bar(o=100, h=102, l=98, c=100, vol=100)
        bar2 = make_bar(o=100, h=106, l=100, c=104, vol=200)
        vwap.update(bar1)
        vwap.update(bar2)

        tp1 = (102 + 98 + 100) / 3.0
        tp2 = (106 + 100 + 104) / 3.0
        expected = (tp1 * 100 + tp2 * 200) / 300
        assert vwap.value == pytest.approx(expected)

    def test_reset(self) -> None:
        vwap = VWAP()
        vwap.update(make_bar(vol=100))
        vwap.reset()
        assert vwap.value == 0.0
        assert vwap._cum_vol == 0.0

    def test_bands_exist(self) -> None:
        vwap = VWAP(band_multipliers=[1.0, 2.0])
        # Feed several bars so std > 0
        for i in range(10):
            bar = make_bar(h=100 + i, l=99 + i, c=99.5 + i, vol=50)
            vwap.update(bar)
        assert vwap.std > 0
        assert vwap.upper_bands[0] > vwap.value
        assert vwap.lower_bands[0] < vwap.value
        assert vwap.upper_bands[1] > vwap.upper_bands[0]

    def test_zero_volume_bar_ignored(self) -> None:
        vwap = VWAP()
        bar = make_bar(vol=0)
        vwap.update(bar)
        assert vwap.value == 0.0


# ─────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────

class TestEMA:
    def test_seed_with_sma(self) -> None:
        ema = EMA(period=3)
        ema.update(10)
        ema.update(20)
        val = ema.update(30)
        # First 3 values → SMA = 20
        assert val == pytest.approx(20.0)

    def test_exponential_weighting(self) -> None:
        ema = EMA(period=3)
        for v in [10, 20, 30]:
            ema.update(v)
        # Now feed 40 → should use EMA formula
        val = ema.update(40)
        mult = 2.0 / 4.0  # 0.5
        expected = (40 - 20) * mult + 20
        assert val == pytest.approx(expected)

    def test_reset(self) -> None:
        ema = EMA(period=5)
        for v in range(10):
            ema.update(float(v))
        ema.reset()
        assert ema.value == 0.0
        assert ema._count == 0

    def test_constant_series(self) -> None:
        ema = EMA(period=10)
        for _ in range(50):
            ema.update(42.0)
        assert ema.value == pytest.approx(42.0)


# ─────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────

class TestATR:
    def test_single_bar_tr(self) -> None:
        atr = ATR(period=14)
        bar = make_bar(h=105, l=100, c=103)
        val = atr.update(bar)
        assert val == pytest.approx(5.0)  # high-low = 5

    def test_gap_up(self) -> None:
        atr = ATR(period=14)
        bar1 = make_bar(h=105, l=100, c=104)
        atr.update(bar1)
        # Gap up: prev close=104, new high=110, low=107
        bar2 = make_bar(h=110, l=107, c=109)
        val = atr.update(bar2)
        # TR = max(110-107, |110-104|, |107-104|) = max(3, 6, 3) = 6
        expected = (5 + 6) / 2  # avg of first two TRs
        assert val == pytest.approx(expected)

    def test_wilder_smoothing_after_period(self) -> None:
        atr = ATR(period=3)
        trs: list[float] = []
        bars = [
            make_bar(h=101, l=99, c=100),
            make_bar(h=103, l=100, c=102),
            make_bar(h=104, l=101, c=103),
            make_bar(h=106, l=102, c=105),
        ]
        atr.update(bars[0])  # TR=2
        atr.update(bars[1])  # TR=max(3,3,0)=3
        atr.update(bars[2])  # TR=max(3,2,1)=3

        # After 3 bars: ATR = (2+3+3)/3 = 2.667
        assert atr.value == pytest.approx(8.0 / 3.0)

        # 4th bar: Wilder smoothing
        # TR = max(4, |106-103|, |102-103|) = max(4,3,1) = 4
        atr.update(bars[3])
        expected = (atr.value * 2 + 4) / 3  # Wait, already updated
        # Let's compute manually
        atr_3 = 8.0 / 3.0
        tr4 = 4.0
        expected = (atr_3 * 2 + tr4) / 3
        # But atr.value is already updated, recompute:
        atr2 = ATR(period=3)
        atr2.update(bars[0])
        atr2.update(bars[1])
        atr2.update(bars[2])
        atr_before = atr2.value
        atr2.update(bars[3])
        assert atr2.value == pytest.approx((atr_before * 2 + 4.0) / 3.0)

    def test_reset(self) -> None:
        atr = ATR(period=5)
        atr.update(make_bar(h=110, l=100, c=105))
        atr.reset()
        assert atr.value == 0.0
        assert atr._prev_close is None


# ─────────────────────────────────────────────────────────────
# CVD
# ─────────────────────────────────────────────────────────────

class TestCVD:
    def test_cumulative(self) -> None:
        cvd = CVD(lookback=3)
        bar = make_bar(buy_vol=60, sell_vol=40)
        cvd.update(bar)
        assert cvd.cumulative == pytest.approx(20.0)

    def test_rising(self) -> None:
        cvd = CVD(lookback=3)
        # Each bar has positive delta → CVD rises
        for i in range(5):
            bar = make_bar(buy_vol=60 + i * 10, sell_vol=40)
            cvd.update(bar)
        assert cvd.is_rising()

    def test_falling(self) -> None:
        cvd = CVD(lookback=3)
        for i in range(5):
            bar = make_bar(buy_vol=40, sell_vol=60 + i * 10)
            cvd.update(bar)
        assert cvd.is_falling()

    def test_not_rising_with_negative_deltas(self) -> None:
        cvd = CVD(lookback=3)
        # Mix of positive and negative
        deltas = [(60, 40), (30, 70), (60, 40)]  # +20, -40, +20
        for bv, sv in deltas:
            bar = make_bar(buy_vol=bv, sell_vol=sv)
            cvd.update(bar)
        assert not cvd.is_rising()

    def test_not_enough_bars(self) -> None:
        cvd = CVD(lookback=5)
        cvd.update(make_bar(buy_vol=60, sell_vol=40))
        assert not cvd.is_rising()
        assert not cvd.is_falling()

    def test_reset(self) -> None:
        cvd = CVD()
        cvd.update(make_bar(buy_vol=60, sell_vol=40))
        cvd.reset()
        assert cvd.cumulative == 0.0


# ─────────────────────────────────────────────────────────────
# VolumeSMA
# ─────────────────────────────────────────────────────────────

class TestVolumeSMA:
    def test_simple_average(self) -> None:
        vsma = VolumeSMA(period=3)
        vsma.update(10)
        vsma.update(20)
        val = vsma.update(30)
        assert val == pytest.approx(20.0)

    def test_rolling_window(self) -> None:
        vsma = VolumeSMA(period=3)
        for v in [10, 20, 30, 40]:
            vsma.update(v)
        # Buffer should be [20, 30, 40]
        assert vsma.value == pytest.approx(30.0)

    def test_spike_detected(self) -> None:
        vsma = VolumeSMA(period=3)
        for v in [10, 10, 10]:
            vsma.update(v)
        assert vsma.is_spike(13, multiplier=1.3)
        assert not vsma.is_spike(12, multiplier=1.3)

    def test_spike_with_zero_sma(self) -> None:
        vsma = VolumeSMA(period=5)
        assert not vsma.is_spike(100, multiplier=1.3)

    def test_reset(self) -> None:
        vsma = VolumeSMA(period=5)
        vsma.update(100)
        vsma.reset()
        assert vsma.value == 0.0
        assert len(vsma._buffer) == 0


# ─────────────────────────────────────────────────────────────
# IndicatorEngine
# ─────────────────────────────────────────────────────────────

class TestIndicatorEngine:
    @pytest.fixture
    def cfg(self) -> dict:
        return {
            "indicators": {
                "ema_fast": 9,
                "ema_slow": 21,
                "atr_period": 14,
                "volume_sma_period": 20,
                "vwap_band_multipliers": [1.0, 2.0],
            },
            "entry": {
                "cvd_lookback_bars": 3,
                "volume_spike_multiplier": 1.3,
            },
        }

    def test_snapshot_returns_state(self, cfg: dict) -> None:
        engine = IndicatorEngine(cfg)
        bar = make_bar(o=100, h=102, l=98, c=101, vol=50, buy_vol=30, sell_vol=20)
        engine.on_exec_bar(bar)
        engine.on_1min_bar(bar)

        state = engine.snapshot(current_price=101.0, current_volume=60.0)
        assert state.current_price == 101.0
        assert state.current_volume == 60.0
        assert state.vwap > 0

    def test_session_reset(self, cfg: dict) -> None:
        engine = IndicatorEngine(cfg)
        bar = make_bar(vol=50)
        engine.on_exec_bar(bar)
        engine.reset_session()
        assert engine.vwap.value == 0.0
        assert engine.ema_fast.value == 0.0

    def test_1min_bar_color_detection(self, cfg: dict) -> None:
        engine = IndicatorEngine(cfg)
        green_bar = make_bar(o=100, c=102)
        engine.on_1min_bar(green_bar)
        state = engine.snapshot(102, 0)
        assert state.last_1min_bar_green is True
        assert state.last_1min_bar_red is False

        red_bar = make_bar(o=102, c=100)
        engine.on_1min_bar(red_bar)
        state = engine.snapshot(100, 0)
        assert state.last_1min_bar_green is False
        assert state.last_1min_bar_red is True
