"""Broker connection and real-time bar aggregation via Interactive Brokers (ib_insync)."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import datetime, time
from typing import Any, Callable

import pytz
from ib_insync import IB, Contract, Future, Ticker, util

from scalper.indicators import Bar

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


# ─────────────────────────────────────────────────────────────
# Bar aggregator
# ─────────────────────────────────────────────────────────────

class BarAggregator:
    """Aggregates raw ticks into fixed-time or tick-count bars.

    Calls *on_bar* callback with a completed :class:`Bar` whenever a
    bar closes.

    Args:
        bar_type: ``"time"`` for time-based bars, ``"tick"`` for tick-count bars.
        bar_size: Seconds (if time-based) or tick count.
        on_bar: Callback invoked with each completed bar.
    """

    def __init__(
        self,
        bar_type: str = "time",
        bar_size: int = 5,
        on_bar: Callable[[Bar], None] | None = None,
    ) -> None:
        self.bar_type = bar_type
        self.bar_size = bar_size
        self.on_bar = on_bar

        self._open: float = 0.0
        self._high: float = -1e9
        self._low: float = 1e9
        self._close: float = 0.0
        self._volume: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._tick_count: int = 0
        self._bar_start: float = 0.0
        self._started: bool = False

    def reset(self) -> None:
        """Reset the aggregator state for a new session."""
        self._started = False
        self._tick_count = 0
        self._volume = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0

    def on_tick(self, price: float, size: float, side: str, timestamp: float) -> None:
        """Process one incoming tick.

        Args:
            price: Trade price.
            size: Trade size (contracts).
            side: ``"BUY"`` if traded at ask, ``"SELL"`` if at bid, ``"UNKNOWN"`` otherwise.
            timestamp: Epoch seconds of the tick.
        """
        if not self._started:
            self._start_bar(price, timestamp)

        # Update OHLCV
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._volume += size
        if side == "BUY":
            self._buy_volume += size
        elif side == "SELL":
            self._sell_volume += size
        self._tick_count += 1

        # Check bar completion
        if self.bar_type == "time":
            if timestamp - self._bar_start >= self.bar_size:
                self._emit_bar(timestamp)
        elif self.bar_type == "tick":
            if self._tick_count >= self.bar_size:
                self._emit_bar(timestamp)

    def force_close(self, timestamp: float) -> None:
        """Force-close the current bar (e.g. at session end).

        Args:
            timestamp: Epoch seconds.
        """
        if self._started and self._volume > 0:
            self._emit_bar(timestamp)

    # ── internal ─────────────────────────────────────────────

    def _start_bar(self, price: float, timestamp: float) -> None:
        """Initialise a new bar."""
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._tick_count = 0
        self._bar_start = timestamp
        self._started = True

    def _emit_bar(self, timestamp: float) -> None:
        """Create a Bar and invoke the callback."""
        bar = Bar(
            timestamp=self._bar_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            buy_volume=self._buy_volume,
            sell_volume=self._sell_volume,
        )
        if self.on_bar:
            self.on_bar(bar)
        # Reset for the next bar
        self._start_bar(self._close, timestamp)


# ─────────────────────────────────────────────────────────────
# 1-minute bar aggregator (confirmation timeframe)
# ─────────────────────────────────────────────────────────────

class MinuteBarAggregator:
    """Aggregates ticks into 1-minute bars aligned to clock minutes.

    Args:
        on_bar: Callback invoked with each completed 1-minute bar.
    """

    def __init__(self, on_bar: Callable[[Bar], None] | None = None) -> None:
        self.on_bar = on_bar
        self._current_minute: int = -1
        self._open: float = 0.0
        self._high: float = -1e9
        self._low: float = 1e9
        self._close: float = 0.0
        self._volume: float = 0.0
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._bar_start: float = 0.0
        self._started: bool = False

    def reset(self) -> None:
        """Reset state."""
        self._current_minute = -1
        self._started = False

    def on_tick(self, price: float, size: float, side: str, timestamp: float) -> None:
        """Process one incoming tick.

        Args:
            price: Trade price.
            size: Trade size.
            side: ``"BUY"`` / ``"SELL"`` / ``"UNKNOWN"``.
            timestamp: Epoch seconds.
        """
        minute = int(timestamp // 60)

        if not self._started:
            self._start_bar(price, timestamp, minute)
            self._update(price, size, side)
            return

        if minute != self._current_minute:
            # New minute → close the previous bar and start fresh
            self._emit_bar()
            self._start_bar(price, timestamp, minute)

        self._update(price, size, side)

    def _update(self, price: float, size: float, side: str) -> None:
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._volume += size
        if side == "BUY":
            self._buy_volume += size
        elif side == "SELL":
            self._sell_volume += size

    def _start_bar(self, price: float, timestamp: float, minute: int) -> None:
        self._current_minute = minute
        self._open = price
        self._high = price
        self._low = price
        self._close = price
        self._volume = 0.0
        self._buy_volume = 0.0
        self._sell_volume = 0.0
        self._bar_start = timestamp
        self._started = True

    def _emit_bar(self) -> None:
        bar = Bar(
            timestamp=self._bar_start,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=self._volume,
            buy_volume=self._buy_volume,
            sell_volume=self._sell_volume,
        )
        if self.on_bar:
            self.on_bar(bar)


# ─────────────────────────────────────────────────────────────
# IB data feed
# ─────────────────────────────────────────────────────────────

class IBDataFeed:
    """Manages the Interactive Brokers connection and tick subscription.

    Bridges raw IB tick data to :class:`BarAggregator` and
    :class:`MinuteBarAggregator` instances.

    Args:
        cfg: Full configuration dictionary.
        on_exec_bar: Callback for completed execution-timeframe bars.
        on_1min_bar: Callback for completed 1-minute bars.
        on_tick: Optional raw-tick callback.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        on_exec_bar: Callable[[Bar], None] | None = None,
        on_1min_bar: Callable[[Bar], None] | None = None,
        on_tick: Callable[[float, float, str, float], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._broker = cfg["broker"]
        self._instr = cfg["instrument"]
        self._data = cfg["data"]

        self.ib = IB()
        self._contract: Contract | None = None
        self._ticker: Ticker | None = None
        self._on_tick = on_tick

        self._exec_agg = BarAggregator(
            bar_type=self._data["execution_bar_type"],
            bar_size=self._data["execution_bar_size"]
            if self._data["execution_bar_type"] == "time"
            else self._data["tick_bar_size"],
            on_bar=on_exec_bar,
        )
        self._min_agg = MinuteBarAggregator(on_bar=on_1min_bar)

        self.last_price: float = 0.0
        self.last_volume: float = 0.0

    async def connect(self) -> None:
        """Establish the IB gateway connection and subscribe to market data."""
        logger.info(
            "Connecting to IB gateway at %s:%s (client %s)...",
            self._broker["gateway_host"],
            self._broker["gateway_port"],
            self._broker["client_id"],
        )
        await self.ib.connectAsync(
            host=self._broker["gateway_host"],
            port=self._broker["gateway_port"],
            clientId=self._broker["client_id"],
            timeout=self._broker["timeout"],
            readonly=self._broker["readonly"],
        )
        logger.info("Connected to IB.")

        # Build contract for the front-month NQ/MNQ future
        self._contract = Future(
            symbol=self._instr["symbol"],
            exchange=self._instr["exchange"],
            currency=self._instr["currency"],
        )
        contracts = await self.ib.qualifyContractsAsync(self._contract)
        if not contracts:
            raise RuntimeError(f"Could not qualify contract for {self._instr['symbol']}")
        self._contract = contracts[0]
        logger.info("Qualified contract: %s", self._contract)

        # Subscribe to tick-by-tick trades
        self._ticker = self.ib.reqTickByTickData(
            self._contract, tickType="AllLast"
        )
        self._ticker.updateEvent += self._on_tick_update
        logger.info("Subscribed to tick-by-tick data for %s", self._contract.localSymbol)

    async def disconnect(self) -> None:
        """Disconnect from IB."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB.")

    def reset_aggregators(self) -> None:
        """Reset bar aggregators at session start."""
        self._exec_agg.reset()
        self._min_agg.reset()

    def _on_tick_update(self, ticker: Ticker) -> None:
        """Handle an incoming tick-by-tick trade update.

        Args:
            ticker: IB Ticker object with new ``tickByTicks`` data.
        """
        for tick in ticker.tickByTicks:
            if tick is None:
                continue

            price = tick.price
            size = tick.size
            ts = tick.time.timestamp() if hasattr(tick.time, "timestamp") else _time.time()

            # Classify trade direction via tick rule
            if hasattr(tick, "pastLimit"):
                # ib_insync provides pastLimit attribute for tick direction
                pass
            # Simple approach: compare to last price
            if price > self.last_price:
                side = "BUY"
            elif price < self.last_price:
                side = "SELL"
            else:
                side = "UNKNOWN"

            self.last_price = price
            self.last_volume = size

            # Feed aggregators
            self._exec_agg.on_tick(price, size, side, ts)
            self._min_agg.on_tick(price, size, side, ts)

            # Raw tick callback
            if self._on_tick:
                self._on_tick(price, size, side, ts)

    async def get_account_equity(self) -> float:
        """Fetch current account net liquidation value.

        Returns:
            Account equity in USD.
        """
        account = self._broker.get("account", "")
        summary = self.ib.accountSummary(account)
        for item in summary:
            if item.tag == "NetLiquidation":
                return float(item.value)
        # Fallback: iterate account values
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        raise RuntimeError("Could not retrieve account equity.")

    async def run(self) -> None:
        """Run the IB event loop (blocking)."""
        logger.info("Starting IB event loop...")
        await self.ib.runAsync()
