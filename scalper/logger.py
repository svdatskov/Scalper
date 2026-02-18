"""Trade logging to SQLite, real-time metrics, and alert dispatch."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from scalper.execution import ExitReason, TradeDirection, TradeRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# SQLite trade log
# ─────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_entry TEXT    NOT NULL,
    timestamp_exit  TEXT    NOT NULL,
    direction       TEXT    NOT NULL,
    entry_price     REAL    NOT NULL,
    exit_price      REAL    NOT NULL,
    contracts       INTEGER NOT NULL,
    pnl             REAL    NOT NULL,
    stop_price      REAL,
    target_price    REAL,
    vwap_at_entry   REAL,
    atr_at_entry    REAL,
    cvd_direction   TEXT,
    exit_reason     TEXT
);
"""


class TradeLogger:
    """Persists trade records to SQLite and computes rolling metrics.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "trades.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._trades: list[TradeRecord] = []

    def open(self) -> None:
        """Open (or create) the database and ensure the schema exists."""
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()
        logger.info("Trade logger opened: %s", self._db_path)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def log_trade(self, trade: TradeRecord) -> None:
        """Insert a completed trade into the database.

        Args:
            trade: A fully closed :class:`TradeRecord`.
        """
        if self._conn is None:
            self.open()

        entry_dt = datetime.utcfromtimestamp(trade.entry_time).isoformat()
        exit_dt = datetime.utcfromtimestamp(trade.exit_time).isoformat()

        self._conn.execute(  # type: ignore[union-attr]
            """INSERT INTO trades
               (timestamp_entry, timestamp_exit, direction, entry_price,
                exit_price, contracts, pnl, stop_price, target_price,
                vwap_at_entry, atr_at_entry, cvd_direction, exit_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_dt,
                exit_dt,
                trade.direction.name,
                trade.entry_price,
                trade.exit_price,
                trade.contracts,
                trade.pnl,
                trade.stop_price,
                trade.tp1_price,
                trade.vwap_at_entry,
                trade.atr_at_entry,
                trade.cvd_direction,
                trade.exit_reason.value if trade.exit_reason else "",
            ),
        )
        self._conn.commit()  # type: ignore[union-attr]
        self._trades.append(trade)
        logger.info("Trade logged: %s %s PnL=%.2f", trade.direction.name, exit_dt, trade.pnl)

    # ─────────────────────────────────────────────────────────
    # Metrics
    # ─────────────────────────────────────────────────────────

    def compute_metrics(self, window: int = 20) -> dict[str, float]:
        """Compute rolling performance metrics over the last *window* trades.

        Args:
            window: Number of recent trades for rolling stats.

        Returns:
            Dictionary with ``win_rate``, ``avg_win``, ``avg_loss``,
            ``profit_factor``, ``max_consecutive_losses``, and
            ``sharpe_estimate``.
        """
        trades = self._trades[-window:] if self._trades else []
        if not trades:
            return {
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_consecutive_losses": 0,
                "sharpe_estimate": 0.0,
                "total_trades": 0,
                "total_pnl": 0.0,
            }

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        pnls = [t.pnl for t in trades]

        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max consecutive losses
        max_consec = 0
        current_consec = 0
        for t in trades:
            if t.pnl <= 0:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        # Sharpe estimate (annualised from trade returns)
        if len(pnls) > 1:
            mean_pnl = sum(pnls) / len(pnls)
            var = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(var) if var > 0 else 1e-9
            # Approximate: assume ~4 trades/day, ~252 days/year
            trades_per_year = 4 * 252
            sharpe = (mean_pnl / std_pnl) * math.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        return {
            "win_rate": round(win_rate, 4),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_consecutive_losses": max_consec,
            "sharpe_estimate": round(sharpe, 2),
            "total_trades": len(self._trades),
            "total_pnl": round(sum(t.pnl for t in self._trades), 2),
        }

    def format_metrics(self, window: int = 20) -> str:
        """Return a human-readable string of current metrics.

        Args:
            window: Rolling window size.

        Returns:
            Formatted metrics string.
        """
        m = self.compute_metrics(window)
        return (
            f"Trades: {m['total_trades']} | "
            f"WR: {m['win_rate']:.1%} | "
            f"AvgW: ${m['avg_win']:.0f} | "
            f"AvgL: ${m['avg_loss']:.0f} | "
            f"PF: {m['profit_factor']:.2f} | "
            f"MaxConsecL: {m['max_consecutive_losses']} | "
            f"Sharpe: {m['sharpe_estimate']:.2f} | "
            f"PnL: ${m['total_pnl']:.0f}"
        )


# ─────────────────────────────────────────────────────────────
# Alert dispatcher
# ─────────────────────────────────────────────────────────────

class AlertDispatcher:
    """Send alerts via Telegram or Discord webhook.

    Args:
        cfg: Full configuration dictionary (expects ``alerts`` key).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        alerts = cfg["alerts"]
        self._enabled: bool = alerts.get("enabled", False)
        self._method: str = alerts.get("method", "telegram")
        self._tg_token: str = alerts.get("telegram_bot_token", "")
        self._tg_chat: str = alerts.get("telegram_chat_id", "")
        self._discord_url: str = alerts.get("discord_webhook_url", "")

    async def send(self, message: str) -> None:
        """Send an alert message via the configured channel.

        Args:
            message: Alert text.
        """
        if not self._enabled:
            return

        try:
            if self._method == "telegram":
                await self._send_telegram(message)
            elif self._method == "discord":
                await self._send_discord(message)
        except Exception as exc:
            logger.error("Alert send failed: %s", exc)

    async def _send_telegram(self, text: str) -> None:
        """Post to Telegram Bot API."""
        if not self._tg_token or not self._tg_chat:
            logger.warning("Telegram credentials not configured.")
            return
        url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
        payload = {"chat_id": self._tg_chat, "text": text, "parse_mode": "Markdown"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error("Telegram error %d: %s", resp.status, body)

    async def _send_discord(self, text: str) -> None:
        """Post to a Discord webhook."""
        if not self._discord_url:
            logger.warning("Discord webhook URL not configured.")
            return
        payload = {"content": text}
        async with aiohttp.ClientSession() as session:
            async with session.post(self._discord_url, json=payload) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.error("Discord error %d: %s", resp.status, body)

    async def send_trade_entry(self, trade: TradeRecord) -> None:
        """Send an alert for a new trade entry.

        Args:
            trade: The newly opened trade.
        """
        msg = (
            f"📈 *ENTRY* | {trade.direction.name} "
            f"{trade.contracts} contracts @ {trade.entry_price:.2f}\n"
            f"SL: {trade.stop_price:.2f} | TP1: {trade.tp1_price:.2f} | TP2: {trade.tp2_price:.2f}\n"
            f"ATR: {trade.atr_at_entry:.2f} | VWAP: {trade.vwap_at_entry:.2f}"
        )
        await self.send(msg)

    async def send_trade_exit(self, trade: TradeRecord) -> None:
        """Send an alert for a trade exit.

        Args:
            trade: The closed trade.
        """
        emoji = "✅" if trade.pnl >= 0 else "❌"
        msg = (
            f"{emoji} *EXIT* | {trade.direction.name} "
            f"@ {trade.exit_price:.2f} | "
            f"PnL: ${trade.pnl:.2f} | "
            f"Reason: {trade.exit_reason.value if trade.exit_reason else 'unknown'}"
        )
        await self.send(msg)

    async def send_circuit_breaker(self, reason: str) -> None:
        """Send an alert when a circuit breaker activates.

        Args:
            reason: Description of the triggered breaker.
        """
        await self.send(f"🚨 *CIRCUIT BREAKER*: {reason}")

    async def send_daily_summary(self, metrics: dict[str, float]) -> None:
        """Send end-of-day performance summary.

        Args:
            metrics: Output of :meth:`TradeLogger.compute_metrics`.
        """
        msg = (
            f"📊 *DAILY SUMMARY*\n"
            f"Trades: {metrics.get('total_trades', 0)}\n"
            f"P&L: ${metrics.get('total_pnl', 0):.2f}\n"
            f"Win Rate: {metrics.get('win_rate', 0):.1%}\n"
            f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n"
            f"Sharpe: {metrics.get('sharpe_estimate', 0):.2f}"
        )
        await self.send(msg)
