"""Historical replay engine for backtesting the scalping strategy.

Reads tick or 1-second bar data from CSV/Parquet, replays through the
same indicator + signal + execution pipeline, and produces performance
reports.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz

from scalper.data_feed import BarAggregator, MinuteBarAggregator
from scalper.execution import (
    ExitReason,
    ExecutionEngine,
    TradeDirection,
    TradeRecord,
)
from scalper.indicators import Bar, IndicatorEngine, IndicatorState
from scalper.logger import TradeLogger
from scalper.risk_manager import RiskManager
from scalper.signals import Signal, SignalEngine
from scalper.utils import TimeFilter, get_commission, get_point_value, load_config

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


@dataclass
class BacktestResult:
    """Container for backtesting output.

    Attributes:
        trades: List of completed trade records.
        equity_curve: List of (timestamp, equity) tuples.
        metrics: Performance metrics dict.
        monthly_pnl: DataFrame of monthly P&L.
    """
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[float, float]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    monthly_pnl: pd.DataFrame | None = None


class Backtester:
    """Replays historical data through the full scalping pipeline.

    Supports CSV files with columns: ``timestamp, price, size`` (tick data)
    or ``timestamp, open, high, low, close, volume`` (bar data).

    Applies configurable slippage and commissions.

    Args:
        cfg: Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        bt = cfg["backtest"]
        self._data_file: str = bt["data_file"]
        self._slippage_ticks: int = bt["slippage_ticks"]
        self._initial_equity: float = bt["initial_equity"]
        self._tick_size: float = cfg["instrument"]["tick_size"]
        self._symbol: str = cfg["instrument"]["symbol"]
        self._point_value: float = get_point_value(self._symbol)
        self._commission: float = get_commission(self._symbol)

        # Components
        self._time_filter = TimeFilter(cfg)
        self._indicators = IndicatorEngine(cfg)
        self._signal_engine = SignalEngine(cfg, self._time_filter)
        self._risk_manager = RiskManager(cfg)
        self._trade_logger = TradeLogger(db_path=":memory:")

        # Execution engine with simulated order fills
        self._execution = ExecutionEngine(
            cfg,
            place_order_fn=self._sim_order,
            on_trade_closed_fn=self._on_trade_closed,
        )

        # State
        self._equity: float = self._initial_equity
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[tuple[float, float]] = []
        self._last_sim_price: float = 0.0
        self._current_date: str = ""

        # Bar collectors
        self._exec_bars: list[Bar] = []
        self._min_bars: list[Bar] = []

    # ─────────────────────────────────────────────────────────
    # Simulated order execution
    # ─────────────────────────────────────────────────────────

    async def _sim_order(
        self,
        direction: TradeDirection,
        contracts: int,
        order_type: str,
        price: float,
    ) -> float:
        """Simulate an order fill with slippage.

        Args:
            direction: Trade direction.
            contracts: Number of contracts.
            order_type: ``"MKT"`` or ``"LMT"``.
            price: Limit price (unused for MKT).

        Returns:
            Simulated fill price.
        """
        slip = self._slippage_ticks * self._tick_size
        if direction == TradeDirection.LONG:
            return self._last_sim_price + slip
        else:
            return self._last_sim_price - slip

    def _on_trade_closed(self, trade: TradeRecord) -> None:
        """Handle a closed trade during backtesting."""
        # Deduct commission
        commission = self._commission * trade.contracts
        trade.pnl -= commission

        self._equity += trade.pnl
        self._trades.append(trade)
        self._trade_logger.log_trade(trade)
        self._risk_manager.on_trade_closed(trade)
        self._equity_curve.append((trade.exit_time, self._equity))

    # ─────────────────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────────────────

    def _load_data(self) -> pd.DataFrame:
        """Load tick/bar data from CSV or Parquet.

        Expected columns for tick data: ``timestamp, price, size``
        Expected columns for bar data: ``timestamp, open, high, low, close, volume``

        Returns:
            DataFrame sorted by timestamp.
        """
        path = Path(self._data_file)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {self._data_file}")

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Normalise timestamp
        if "timestamp" in df.columns:
            if df["timestamp"].dtype == object:
                df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
            elif df["timestamp"].max() > 1e12:
                # Milliseconds
                df["timestamp"] = df["timestamp"] / 1000.0

        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Loaded %d rows from %s", len(df), self._data_file)
        return df

    # ─────────────────────────────────────────────────────────
    # Main backtest loop
    # ─────────────────────────────────────────────────────────

    async def run(self) -> BacktestResult:
        """Execute the full backtest.

        Returns:
            :class:`BacktestResult` with trades, equity curve, and metrics.
        """
        self._trade_logger.open()
        data = self._load_data()
        is_tick = "price" in data.columns

        self._equity = self._initial_equity
        self._equity_curve = [(0, self._equity)]
        self._risk_manager.reset_daily(self._equity)
        self._indicators.reset_session()

        # Aggregators
        exec_agg = BarAggregator(
            bar_type=self._cfg["data"]["execution_bar_type"],
            bar_size=self._cfg["data"]["execution_bar_size"]
            if self._cfg["data"]["execution_bar_type"] == "time"
            else self._cfg["data"]["tick_bar_size"],
            on_bar=self._on_exec_bar,
        )
        min_agg = MinuteBarAggregator(on_bar=self._on_1min_bar)

        total_rows = len(data)
        report_interval = max(1, total_rows // 20)

        for idx, row in data.iterrows():
            ts = float(row["timestamp"])

            # Check for new day → reset
            dt = datetime.utcfromtimestamp(ts)
            date_str = dt.strftime("%Y-%m-%d")
            if date_str != self._current_date:
                self._current_date = date_str
                self._indicators.reset_session()
                self._risk_manager.reset_daily(self._equity)
                exec_agg.reset()
                min_agg.reset()

            if is_tick:
                price = float(row["price"])
                size = float(row.get("size", 1))
                side = "BUY"  # Simplified; real data should include side
                if "side" in row.index:
                    side = str(row["side"]).upper()
                elif price > self._last_sim_price:
                    side = "BUY"
                elif price < self._last_sim_price:
                    side = "SELL"
                else:
                    side = "UNKNOWN"

                self._last_sim_price = price
                exec_agg.on_tick(price, size, side, ts)
                min_agg.on_tick(price, size, side, ts)
            else:
                # Bar data — treat each row as a completed bar
                bar = Bar(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0)),
                    buy_volume=float(row.get("buy_volume", row.get("volume", 0) / 2)),
                    sell_volume=float(row.get("sell_volume", row.get("volume", 0) / 2)),
                )
                self._last_sim_price = bar.close
                self._on_exec_bar(bar)
                self._on_1min_bar(bar)

            # Manage exits on every row
            if self._execution.has_position:
                eod = self._time_filter.is_eod_close_time(
                    datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc),
                    self._cfg["exits"]["eod_close_time"],
                )
                state = self._indicators.snapshot(self._last_sim_price, 0)
                await self._execution.manage_exits(
                    self._last_sim_price, state, ts, is_eod=eod,
                )

            # Signal evaluation
            if not self._execution.has_position:
                can, reason = self._risk_manager.can_trade(self._equity)
                if can:
                    state = self._indicators.snapshot(self._last_sim_price, 0)
                    dt_et = datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc).astimezone(ET)
                    result = self._signal_engine.evaluate(state, dt_et)
                    if result.signal in (Signal.BUY, Signal.SELL):
                        # Compute stop and size
                        stop_dist = max(
                            self._cfg["exits"]["stop_min_points"],
                            min(
                                self._cfg["exits"]["stop_atr_multiplier"] * state.atr,
                                self._cfg["exits"]["stop_max_points"],
                            ),
                        )
                        contracts = self._risk_manager.compute_position_size(
                            self._equity, stop_dist,
                        )
                        await self._execution.enter_trade(
                            result.signal, contracts, self._last_sim_price, state, ts,
                        )

            # Progress reporting
            if int(idx) % report_interval == 0:  # type: ignore[arg-type]
                pct = int(idx) / total_rows * 100  # type: ignore[arg-type]
                logger.info("Backtest progress: %.0f%% | Equity: $%.2f", pct, self._equity)

        # Force-close any remaining position
        if self._execution.has_position:
            await self._execution.force_close(
                self._last_sim_price, data.iloc[-1]["timestamp"], ExitReason.EOD_CLOSE,
            )

        # Build results
        metrics = self._trade_logger.compute_metrics(window=len(self._trades))
        monthly = self._compute_monthly_pnl()

        result = BacktestResult(
            trades=self._trades,
            equity_curve=self._equity_curve,
            metrics=metrics,
            monthly_pnl=monthly,
        )

        self._print_report(result)
        self._trade_logger.close()
        return result

    # ─────────────────────────────────────────────────────────
    # Bar callbacks
    # ─────────────────────────────────────────────────────────

    def _on_exec_bar(self, bar: Bar) -> None:
        """Process an execution-timeframe bar."""
        self._indicators.on_exec_bar(bar)
        self._exec_bars.append(bar)

    def _on_1min_bar(self, bar: Bar) -> None:
        """Process a 1-minute bar."""
        self._indicators.on_1min_bar(bar)
        self._min_bars.append(bar)

    # ─────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────

    def _compute_monthly_pnl(self) -> pd.DataFrame:
        """Compute a monthly P&L table from trade records.

        Returns:
            DataFrame with columns ``month`` and ``pnl``.
        """
        if not self._trades:
            return pd.DataFrame(columns=["month", "pnl"])

        records = []
        for t in self._trades:
            dt = datetime.utcfromtimestamp(t.exit_time)
            records.append({"month": dt.strftime("%Y-%m"), "pnl": t.pnl})

        df = pd.DataFrame(records)
        return df.groupby("month")["pnl"].sum().reset_index()

    def _compute_drawdown_curve(self) -> list[tuple[float, float]]:
        """Compute the drawdown curve from the equity curve.

        Returns:
            List of (timestamp, drawdown_pct) tuples.
        """
        if not self._equity_curve:
            return []

        hwm = self._initial_equity
        curve: list[tuple[float, float]] = []
        for ts, eq in self._equity_curve:
            hwm = max(hwm, eq)
            dd = (hwm - eq) / hwm if hwm > 0 else 0.0
            curve.append((ts, dd))
        return curve

    def _print_report(self, result: BacktestResult) -> None:
        """Print a formatted backtest report to the log."""
        m = result.metrics
        dd_curve = self._compute_drawdown_curve()
        max_dd = max(d for _, d in dd_curve) if dd_curve else 0.0

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                   BACKTEST RESULTS                          ║
╠══════════════════════════════════════════════════════════════╣
║  Initial Equity:   ${self._initial_equity:>12,.2f}                     ║
║  Final Equity:     ${self._equity:>12,.2f}                     ║
║  Total P&L:        ${m.get('total_pnl', 0):>12,.2f}                     ║
║  Total Trades:     {m.get('total_trades', 0):>6}                           ║
║  Win Rate:         {m.get('win_rate', 0):>8.1%}                         ║
║  Avg Win:          ${m.get('avg_win', 0):>10,.2f}                       ║
║  Avg Loss:         ${m.get('avg_loss', 0):>10,.2f}                       ║
║  Profit Factor:    {m.get('profit_factor', 0):>8.2f}                         ║
║  Sharpe Estimate:  {m.get('sharpe_estimate', 0):>8.2f}                         ║
║  Max Consec Losses:{m.get('max_consecutive_losses', 0):>6}                           ║
║  Max Drawdown:     {max_dd:>8.2%}                         ║
╚══════════════════════════════════════════════════════════════╝
"""
        logger.info(report)

        if result.monthly_pnl is not None and not result.monthly_pnl.empty:
            logger.info("\nMonthly P&L:")
            for _, row in result.monthly_pnl.iterrows():
                logger.info("  %s: $%.2f", row["month"], row["pnl"])


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

async def run_backtest(config_path: str = "scalper/config.yaml") -> BacktestResult:
    """Load config and run the backtester.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        :class:`BacktestResult`.
    """
    cfg = load_config(config_path)
    bt = Backtester(cfg)
    return await bt.run()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_backtest())
