"""Order placement, bracket management, and position tracking.

Handles entry orders (market), bracket stop/target orders, partial exits,
trailing stop logic, breakeven moves, and time-based exits.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from scalper.indicators import IndicatorState
from scalper.signals import Signal
from scalper.utils import get_point_value, round_to_tick

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Direction of an open trade."""
    LONG = auto()
    SHORT = auto()


class ExitReason(Enum):
    """Reason a trade was closed."""
    TP1 = "TP1"
    TP2 = "TP2"
    STOP_LOSS = "SL"
    TRAILING_STOP = "trailing"
    TIME_EXIT = "time"
    EOD_CLOSE = "eod"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL = "manual"
    BREAKEVEN = "breakeven"


@dataclass
class TradeRecord:
    """Complete record of one round-trip trade.

    Attributes:
        entry_time: Epoch seconds of entry fill.
        exit_time: Epoch seconds of exit fill (0 if still open).
        direction: LONG or SHORT.
        entry_price: Average fill price at entry.
        exit_price: Average fill price at exit.
        contracts: Number of contracts at entry.
        remaining_contracts: Contracts still open.
        stop_price: Current stop level.
        tp1_price: Target 1 price.
        tp2_price: Target 2 price.
        pnl: Realised P&L in USD.
        exit_reason: Why the trade was closed.
        vwap_at_entry: VWAP reading at time of entry.
        atr_at_entry: ATR reading at time of entry.
        cvd_direction: ``"rising"`` or ``"falling"`` at entry.
        tp1_hit: Whether Target 1 has been reached.
        trail_stop: Current trailing-stop level (active after TP1).
    """
    entry_time: float = 0.0
    exit_time: float = 0.0
    direction: TradeDirection = TradeDirection.LONG
    entry_price: float = 0.0
    exit_price: float = 0.0
    contracts: int = 0
    remaining_contracts: int = 0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    pnl: float = 0.0
    exit_reason: ExitReason | None = None
    vwap_at_entry: float = 0.0
    atr_at_entry: float = 0.0
    cvd_direction: str = ""
    tp1_hit: bool = False
    trail_stop: float = 0.0


class ExecutionEngine:
    """Manages order lifecycle: entries, bracket exits, trailing stops.

    The engine is broker-agnostic at this level — it calculates prices
    and delegates actual order placement to callback functions that can
    be wired to IB, a backtester, or a paper-trading shim.

    Args:
        cfg: Full configuration dictionary.
        place_order_fn: ``async (direction, contracts, order_type, price) -> fill_price``.
            Used to send orders to the broker.
        cancel_order_fn: ``async (order_id) -> None``.
        on_trade_closed_fn: ``(TradeRecord) -> None``.  Called when a trade fully closes.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        place_order_fn: Callable | None = None,
        cancel_order_fn: Callable | None = None,
        on_trade_closed_fn: Callable[[TradeRecord], None] | None = None,
    ) -> None:
        self._cfg = cfg
        exits = cfg["exits"]
        self._stop_atr_mult: float = exits["stop_atr_multiplier"]
        self._stop_min: float = exits["stop_min_points"]
        self._stop_max: float = exits["stop_max_points"]
        self._tp1_mult: float = exits["tp1_atr_multiplier"]
        self._tp1_pct: float = exits["tp1_pct"]
        self._tp2_mult: float = exits["tp2_atr_multiplier"]
        self._tp2_pct: float = exits["tp2_pct"]
        self._trail_mult: float = exits["trail_atr_multiplier"]
        self._be_mult: float = exits["breakeven_atr_multiplier"]
        self._be_ticks: int = exits["breakeven_offset_ticks"]
        self._max_dur: int = exits["max_trade_duration_seconds"]
        self._tick_size: float = cfg["instrument"]["tick_size"]
        self._symbol: str = cfg["instrument"]["symbol"]
        self._point_value: float = get_point_value(self._symbol)

        self._place_order = place_order_fn
        self._cancel_order = cancel_order_fn
        self._on_trade_closed = on_trade_closed_fn

        self.active_trade: TradeRecord | None = None

    # ─────────────────────────────────────────────────────────
    # Entry
    # ─────────────────────────────────────────────────────────

    async def enter_trade(
        self,
        signal: Signal,
        contracts: int,
        current_price: float,
        state: IndicatorState,
        timestamp: float | None = None,
    ) -> TradeRecord | None:
        """Open a new position with bracket orders.

        Args:
            signal: BUY or SELL.
            contracts: Number of contracts to trade.
            current_price: Latest market price.
            state: Indicator snapshot at entry time.
            timestamp: Epoch seconds (defaults to now).

        Returns:
            The new :class:`TradeRecord`, or ``None`` if entry failed.
        """
        if self.active_trade is not None:
            logger.warning("Already in a trade — ignoring new signal.")
            return None

        ts = timestamp or _time.time()
        direction = TradeDirection.LONG if signal == Signal.BUY else TradeDirection.SHORT
        atr = state.atr

        # Calculate stop distance
        raw_stop = self._stop_atr_mult * atr
        stop_dist = max(self._stop_min, min(raw_stop, self._stop_max))
        stop_dist = round_to_tick(stop_dist, self._tick_size)

        # Calculate target prices
        tp1_dist = round_to_tick(self._tp1_mult * atr, self._tick_size)
        tp2_dist = round_to_tick(self._tp2_mult * atr, self._tick_size)

        if direction == TradeDirection.LONG:
            stop_price = round_to_tick(current_price - stop_dist, self._tick_size)
            tp1_price = round_to_tick(current_price + tp1_dist, self._tick_size)
            tp2_price_atr = round_to_tick(current_price + tp2_dist, self._tick_size)
            # TP2 can also be VWAP +1σ band (whichever comes first)
            tp2_price = min(tp2_price_atr, state.vwap_upper_1) if state.vwap_upper_1 > current_price else tp2_price_atr
        else:
            stop_price = round_to_tick(current_price + stop_dist, self._tick_size)
            tp1_price = round_to_tick(current_price - tp1_dist, self._tick_size)
            tp2_price_atr = round_to_tick(current_price - tp2_dist, self._tick_size)
            tp2_price = max(tp2_price_atr, state.vwap_lower_1) if state.vwap_lower_1 < current_price else tp2_price_atr

        # Place market entry
        fill_price = current_price
        if self._place_order:
            fill_price = await self._place_order(direction, contracts, "MKT", 0.0)

        trade = TradeRecord(
            entry_time=ts,
            direction=direction,
            entry_price=fill_price,
            contracts=contracts,
            remaining_contracts=contracts,
            stop_price=stop_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            vwap_at_entry=state.vwap,
            atr_at_entry=atr,
            cvd_direction="rising" if state.cvd_rising else ("falling" if state.cvd_falling else "neutral"),
        )
        self.active_trade = trade

        logger.info(
            "ENTERED %s %d @ %.2f | SL=%.2f TP1=%.2f TP2=%.2f",
            direction.name, contracts, fill_price, stop_price, tp1_price, tp2_price,
        )
        return trade

    # ─────────────────────────────────────────────────────────
    # Exit management (called every tick / bar close)
    # ─────────────────────────────────────────────────────────

    async def manage_exits(
        self,
        current_price: float,
        state: IndicatorState,
        timestamp: float | None = None,
        is_eod: bool = False,
    ) -> TradeRecord | None:
        """Check all exit conditions and close (partially or fully) if triggered.

        Should be called on every tick or bar close while a trade is open.

        Args:
            current_price: Latest price.
            state: Current indicator snapshot.
            timestamp: Epoch seconds.
            is_eod: ``True`` if past the EOD forced-close time.

        Returns:
            The :class:`TradeRecord` if it was fully closed, else ``None``.
        """
        trade = self.active_trade
        if trade is None:
            return None

        ts = timestamp or _time.time()
        direction = trade.direction

        # ── EOD forced close ─────────────────────────────────
        if is_eod:
            return await self._close_trade(current_price, ts, ExitReason.EOD_CLOSE)

        # ── Time-based exit ──────────────────────────────────
        elapsed = ts - trade.entry_time
        if elapsed > self._max_dur and not trade.tp1_hit:
            logger.info("Time exit: trade open %.0fs without TP1.", elapsed)
            return await self._close_trade(current_price, ts, ExitReason.TIME_EXIT)

        # ── Stop loss ────────────────────────────────────────
        if direction == TradeDirection.LONG and current_price <= trade.stop_price:
            reason = ExitReason.TRAILING_STOP if trade.tp1_hit else ExitReason.STOP_LOSS
            return await self._close_trade(trade.stop_price, ts, reason)
        if direction == TradeDirection.SHORT and current_price >= trade.stop_price:
            reason = ExitReason.TRAILING_STOP if trade.tp1_hit else ExitReason.STOP_LOSS
            return await self._close_trade(trade.stop_price, ts, reason)

        # ── Take Profit 1 (partial) ─────────────────────────
        if not trade.tp1_hit:
            hit_tp1 = (
                (direction == TradeDirection.LONG and current_price >= trade.tp1_price)
                or (direction == TradeDirection.SHORT and current_price <= trade.tp1_price)
            )
            if hit_tp1:
                trade.tp1_hit = True
                close_qty = max(1, int(trade.contracts * self._tp1_pct))
                trade.remaining_contracts -= close_qty
                pnl = self._calc_pnl(trade.entry_price, trade.tp1_price, close_qty, direction)
                trade.pnl += pnl
                logger.info("TP1 hit: closed %d contracts @ %.2f (partial PnL=%.2f)", close_qty, trade.tp1_price, pnl)

                if self._place_order:
                    await self._place_order(
                        TradeDirection.SHORT if direction == TradeDirection.LONG else TradeDirection.LONG,
                        close_qty, "MKT", 0.0,
                    )

                # Activate trailing stop
                trail_dist = round_to_tick(self._trail_mult * trade.atr_at_entry, self._tick_size)
                if direction == TradeDirection.LONG:
                    trade.trail_stop = round_to_tick(current_price - trail_dist, self._tick_size)
                    trade.stop_price = max(trade.stop_price, trade.trail_stop)
                else:
                    trade.trail_stop = round_to_tick(current_price + trail_dist, self._tick_size)
                    trade.stop_price = min(trade.stop_price, trade.trail_stop)

                if trade.remaining_contracts <= 0:
                    return await self._close_trade(trade.tp1_price, ts, ExitReason.TP1)

        # ── Take Profit 2 (full close) ──────────────────────
        if trade.tp1_hit and trade.remaining_contracts > 0:
            hit_tp2 = (
                (direction == TradeDirection.LONG and current_price >= trade.tp2_price)
                or (direction == TradeDirection.SHORT and current_price <= trade.tp2_price)
            )
            if hit_tp2:
                return await self._close_trade(trade.tp2_price, ts, ExitReason.TP2)

        # ── Breakeven rule ───────────────────────────────────
        if not trade.tp1_hit:
            be_dist = self._be_mult * trade.atr_at_entry
            if direction == TradeDirection.LONG:
                if current_price >= trade.entry_price + be_dist:
                    be_price = round_to_tick(
                        trade.entry_price + self._be_ticks * self._tick_size,
                        self._tick_size,
                    )
                    if be_price > trade.stop_price:
                        trade.stop_price = be_price
                        logger.info("Moved stop to breakeven: %.2f", be_price)
            else:
                if current_price <= trade.entry_price - be_dist:
                    be_price = round_to_tick(
                        trade.entry_price - self._be_ticks * self._tick_size,
                        self._tick_size,
                    )
                    if be_price < trade.stop_price:
                        trade.stop_price = be_price
                        logger.info("Moved stop to breakeven: %.2f", be_price)

        # ── Update trailing stop (after TP1, on each bar close) ──
        if trade.tp1_hit and trade.remaining_contracts > 0:
            trail_dist = round_to_tick(self._trail_mult * trade.atr_at_entry, self._tick_size)
            if direction == TradeDirection.LONG:
                new_trail = round_to_tick(current_price - trail_dist, self._tick_size)
                if new_trail > trade.stop_price:
                    trade.stop_price = new_trail
                    logger.debug("Trail stop updated: %.2f", new_trail)
            else:
                new_trail = round_to_tick(current_price + trail_dist, self._tick_size)
                if new_trail < trade.stop_price:
                    trade.stop_price = new_trail
                    logger.debug("Trail stop updated: %.2f", new_trail)

        return None

    # ─────────────────────────────────────────────────────────
    # Force close (for circuit breakers / EOD)
    # ─────────────────────────────────────────────────────────

    async def force_close(
        self,
        current_price: float,
        timestamp: float | None = None,
        reason: ExitReason = ExitReason.CIRCUIT_BREAKER,
    ) -> TradeRecord | None:
        """Forcefully close the entire active position.

        Args:
            current_price: Price for P&L calculation.
            timestamp: Epoch seconds.
            reason: Reason for the forced close.

        Returns:
            Closed :class:`TradeRecord` or ``None``.
        """
        if self.active_trade is None:
            return None
        return await self._close_trade(current_price, timestamp or _time.time(), reason)

    @property
    def has_position(self) -> bool:
        """Return True if there is an active open trade."""
        return self.active_trade is not None

    # ─────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────

    async def _close_trade(
        self, exit_price: float, timestamp: float, reason: ExitReason,
    ) -> TradeRecord:
        """Close the active trade and invoke the callback.

        Args:
            exit_price: Fill price for remaining contracts.
            timestamp: Epoch seconds.
            reason: Why the trade was closed.

        Returns:
            The completed :class:`TradeRecord`.
        """
        trade = self.active_trade
        assert trade is not None

        remaining = trade.remaining_contracts
        if remaining > 0:
            pnl = self._calc_pnl(trade.entry_price, exit_price, remaining, trade.direction)
            trade.pnl += pnl

            if self._place_order:
                close_dir = TradeDirection.SHORT if trade.direction == TradeDirection.LONG else TradeDirection.LONG
                await self._place_order(close_dir, remaining, "MKT", 0.0)

        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.exit_reason = reason
        trade.remaining_contracts = 0

        logger.info(
            "CLOSED %s @ %.2f | reason=%s PnL=%.2f",
            trade.direction.name, exit_price, reason.value, trade.pnl,
        )

        self.active_trade = None

        if self._on_trade_closed:
            self._on_trade_closed(trade)

        return trade

    def _calc_pnl(
        self,
        entry: float,
        exit_price: float,
        qty: int,
        direction: TradeDirection,
    ) -> float:
        """Calculate P&L for a partial or full close.

        Args:
            entry: Entry price.
            exit_price: Exit price.
            qty: Number of contracts being closed.
            direction: Trade direction.

        Returns:
            P&L in USD (before commissions).
        """
        if direction == TradeDirection.LONG:
            return (exit_price - entry) * qty * self._point_value
        else:
            return (entry - exit_price) * qty * self._point_value
