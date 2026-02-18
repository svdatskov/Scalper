"""Entry point and main event loop for the NQ/MNQ scalping bot.

Wires together all components: data feed, indicators, signals,
execution, risk management, logging, and alerts.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time as _time
from datetime import datetime
from typing import Any

import pytz

from scalper.backtester import run_backtest
from scalper.data_feed import IBDataFeed
from scalper.execution import ExecutionEngine, ExitReason, TradeDirection, TradeRecord
from scalper.indicators import Bar, IndicatorEngine
from scalper.logger import AlertDispatcher, TradeLogger
from scalper.risk_manager import RiskManager
from scalper.signals import Signal, SignalEngine
from scalper.utils import TimeFilter, get_point_value, load_config

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class ScalpingBot:
    """Main bot orchestrator.

    Connects to IB, subscribes to market data, and runs the
    signal → execution → risk loop on every bar/tick.

    Args:
        cfg: Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._symbol: str = cfg["instrument"]["symbol"]
        self._point_value: float = get_point_value(self._symbol)
        self._tick_size: float = cfg["instrument"]["tick_size"]

        # Components
        self._time_filter = TimeFilter(cfg)
        self._indicators = IndicatorEngine(cfg)
        self._signal_engine = SignalEngine(cfg, self._time_filter)
        self._risk_manager = RiskManager(cfg)
        self._trade_logger = TradeLogger(db_path=cfg["logging"]["trade_log_db"])
        self._alerts = AlertDispatcher(cfg)

        # Execution engine
        self._execution = ExecutionEngine(
            cfg,
            place_order_fn=self._place_ib_order,
            on_trade_closed_fn=self._on_trade_closed,
        )

        # Data feed (created on connect)
        self._feed: IBDataFeed | None = None

        # State
        self._equity: float = 0.0
        self._running: bool = False
        self._session_started: bool = False

    # ─────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to IB, initialise components, and begin trading."""
        self._trade_logger.open()

        self._feed = IBDataFeed(
            self._cfg,
            on_exec_bar=self._on_exec_bar,
            on_1min_bar=self._on_1min_bar,
        )
        await self._feed.connect()

        # Get initial equity
        self._equity = await self._feed.get_account_equity()
        self._risk_manager.reset_daily(self._equity)
        self._indicators.reset_session()
        self._session_started = True

        logger.info(
            "Bot started: %s | Equity=$%.2f",
            self._symbol, self._equity,
        )
        await self._alerts.send(
            f"🤖 *Bot started* | {self._symbol} | Equity: ${self._equity:,.2f}"
        )

        self._running = True
        await self._run_loop()

    async def stop(self) -> None:
        """Gracefully shut down the bot."""
        self._running = False

        # Close any open position
        if self._execution.has_position and self._feed:
            await self._execution.force_close(
                self._feed.last_price, _time.time(), ExitReason.MANUAL,
            )

        # Send daily summary
        metrics = self._trade_logger.compute_metrics()
        await self._alerts.send_daily_summary(metrics)
        logger.info("Daily summary: %s", self._trade_logger.format_metrics())

        # Disconnect
        if self._feed:
            await self._feed.disconnect()
        self._trade_logger.close()
        logger.info("Bot stopped.")

    # ─────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Run the IB event loop.  Bar callbacks drive the strategy."""
        assert self._feed is not None
        try:
            # ib_insync event loop — ticks/bars arrive via callbacks
            while self._running:
                self._feed.ib.sleep(0.1)
                await asyncio.sleep(0.01)

                # Check for EOD close
                if self._execution.has_position:
                    if self._time_filter.is_eod_close_time(
                        eod_time_str=self._cfg["exits"]["eod_close_time"]
                    ):
                        logger.info("EOD close triggered.")
                        await self._execution.force_close(
                            self._feed.last_price, _time.time(), ExitReason.EOD_CLOSE,
                        )

                # Check for shutdown
                if self._risk_manager.is_shutdown:
                    logger.critical("Risk manager shutdown — stopping bot.")
                    await self._alerts.send_circuit_breaker("Max drawdown reached — bot shut down.")
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            await self.stop()

    # ─────────────────────────────────────────────────────────
    # Bar callbacks
    # ─────────────────────────────────────────────────────────

    def _on_exec_bar(self, bar: Bar) -> None:
        """Handle a completed execution-timeframe bar.

        This is where the core strategy logic runs: update indicators,
        evaluate signals, manage exits.

        Args:
            bar: Completed 5-second (or tick) bar.
        """
        self._indicators.on_exec_bar(bar)

        # Build snapshot
        state = self._indicators.snapshot(bar.close, bar.volume)

        # Schedule async exit/entry management
        asyncio.ensure_future(self._process_bar(bar, state))

    def _on_1min_bar(self, bar: Bar) -> None:
        """Handle a completed 1-minute confirmation bar.

        Args:
            bar: Completed 1-minute bar.
        """
        self._indicators.on_1min_bar(bar)

    async def _process_bar(self, bar: Bar, state: Any) -> None:
        """Async handler for bar processing: exits then entries.

        Args:
            bar: Latest execution bar.
            state: Indicator snapshot.
        """
        ts = bar.timestamp

        # ── Manage exits ─────────────────────────────────────
        if self._execution.has_position:
            eod = self._time_filter.is_eod_close_time(
                eod_time_str=self._cfg["exits"]["eod_close_time"],
            )
            await self._execution.manage_exits(bar.close, state, ts, is_eod=eod)

        # ── Check for new entries ────────────────────────────
        if not self._execution.has_position:
            can, reason = self._risk_manager.can_trade(self._equity)
            if not can:
                return

            result = self._signal_engine.evaluate(state)
            if result.signal in (Signal.BUY, Signal.SELL):
                # Compute stop and position size
                atr = state.atr
                stop_dist = max(
                    self._cfg["exits"]["stop_min_points"],
                    min(
                        self._cfg["exits"]["stop_atr_multiplier"] * atr,
                        self._cfg["exits"]["stop_max_points"],
                    ),
                )
                contracts = self._risk_manager.compute_position_size(
                    self._equity, stop_dist,
                )
                trade = await self._execution.enter_trade(
                    result.signal, contracts, bar.close, state, ts,
                )
                if trade:
                    await self._alerts.send_trade_entry(trade)

        # Log metrics periodically
        if self._risk_manager.daily_trades > 0 and self._risk_manager.daily_trades % 5 == 0:
            logger.info("Metrics: %s", self._trade_logger.format_metrics())

    # ─────────────────────────────────────────────────────────
    # Order placement (IB bridge)
    # ─────────────────────────────────────────────────────────

    async def _place_ib_order(
        self,
        direction: TradeDirection,
        contracts: int,
        order_type: str,
        price: float,
    ) -> float:
        """Place an order via Interactive Brokers.

        Args:
            direction: Trade direction.
            contracts: Number of contracts.
            order_type: ``"MKT"`` or ``"LMT"``.
            price: Limit price (ignored for MKT).

        Returns:
            Fill price.
        """
        assert self._feed is not None
        from ib_insync import MarketOrder

        action = "BUY" if direction == TradeDirection.LONG else "SELL"
        order = MarketOrder(action, contracts)

        trade = self._feed.ib.placeOrder(self._feed._contract, order)

        # Wait for fill (with timeout)
        timeout = 10
        elapsed = 0.0
        while not trade.isDone() and elapsed < timeout:
            self._feed.ib.sleep(0.1)
            await asyncio.sleep(0.05)
            elapsed += 0.15

        if trade.orderStatus.status == "Filled":
            fill_price = trade.orderStatus.avgFillPrice
            logger.info("Order filled: %s %d @ %.2f", action, contracts, fill_price)
            return fill_price
        else:
            logger.warning("Order not filled within timeout. Status: %s", trade.orderStatus.status)
            return self._feed.last_price

    # ─────────────────────────────────────────────────────────
    # Trade closed callback
    # ─────────────────────────────────────────────────────────

    def _on_trade_closed(self, trade: TradeRecord) -> None:
        """Handle a closed trade: log, update risk, send alert.

        Args:
            trade: The completed trade.
        """
        # Deduct commission
        commission = (
            self._cfg["instrument"]["commission_nq"]
            if self._symbol.upper() == "NQ"
            else self._cfg["instrument"]["commission_mnq"]
        )
        trade.pnl -= commission * trade.contracts

        self._equity += trade.pnl
        self._trade_logger.log_trade(trade)
        self._risk_manager.on_trade_closed(trade)

        logger.info(
            "Trade closed: %s PnL=$%.2f | Equity=$%.2f | %s",
            trade.direction.name, trade.pnl, self._equity,
            self._trade_logger.format_metrics(),
        )

        # Async alert
        asyncio.ensure_future(self._alerts.send_trade_exit(trade))

        # Check circuit breakers
        can, reason = self._risk_manager.can_trade(self._equity)
        if not can:
            logger.warning("Circuit breaker: %s", reason)
            asyncio.ensure_future(self._alerts.send_circuit_breaker(reason))


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def setup_logging(cfg: dict[str, Any]) -> None:
    """Configure Python logging from the config.

    Args:
        cfg: Full configuration dictionary.
    """
    log_cfg = cfg["logging"]
    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_file = log_cfg.get("log_file")
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> None:
    """CLI entry point: parse arguments and run the bot or backtester."""
    parser = argparse.ArgumentParser(description="NQ/MNQ Scalping Bot")
    parser.add_argument(
        "--config", "-c",
        default="scalper/config.yaml",
        help="Path to YAML config file (default: scalper/config.yaml)",
    )
    parser.add_argument(
        "--backtest", "-b",
        action="store_true",
        help="Run in backtest mode instead of live trading",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.backtest:
        logger.info("Running backtester...")
        asyncio.run(run_backtest(args.config))
    else:
        logger.info("Starting live scalping bot...")
        bot = ScalpingBot(cfg)
        asyncio.run(bot.start())


if __name__ == "__main__":
    main()
