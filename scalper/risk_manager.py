"""Risk management: position sizing, circuit breakers, daily limits."""

from __future__ import annotations

import logging
import math
import time as _time
from typing import Any

from scalper.execution import TradeRecord
from scalper.utils import get_point_value

logger = logging.getLogger(__name__)


class RiskManager:
    """Enforces all risk rules: sizing, daily loss, consecutive losses,
    trade count, and max drawdown.

    Args:
        cfg: Full configuration dictionary.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        risk = cfg["risk"]
        sizing = cfg["sizing"]
        self._symbol: str = cfg["instrument"]["symbol"]
        self._point_value: float = get_point_value(self._symbol)
        self._tick_size: float = cfg["instrument"]["tick_size"]

        # Sizing
        self._risk_pct: float = sizing["risk_pct"]
        self._max_contracts: int = (
            sizing["max_contracts_nq"]
            if self._symbol.upper() == "NQ"
            else sizing["max_contracts_mnq"]
        )

        # Circuit breakers
        self._daily_loss_pct: float = risk["daily_loss_limit_pct"]
        self._consec_loss_pause: int = risk["consecutive_loss_pause"]
        self._consec_pause_min: int = risk["consecutive_loss_pause_minutes"]
        self._daily_trade_limit: int = risk["daily_trade_limit"]
        self._max_dd_pct: float = risk["max_drawdown_pct"]

        # State
        self._starting_equity: float = 0.0
        self._high_water_mark: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._pause_until: float = 0.0
        self._shutdown: bool = False

    # ─────────────────────────────────────────────────────────
    # Session lifecycle
    # ─────────────────────────────────────────────────────────

    def reset_daily(self, equity: float) -> None:
        """Reset daily counters at the start of a new trading day.

        Args:
            equity: Current account equity.
        """
        self._starting_equity = equity
        self._high_water_mark = max(self._high_water_mark, equity)
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._consecutive_losses = 0
        self._pause_until = 0.0
        self._shutdown = False
        logger.info(
            "Risk manager reset: equity=%.2f HWM=%.2f",
            equity, self._high_water_mark,
        )

    # ─────────────────────────────────────────────────────────
    # Position sizing
    # ─────────────────────────────────────────────────────────

    def compute_position_size(self, equity: float, stop_distance: float) -> int:
        """Calculate the number of contracts to trade.

        ``contracts = floor(equity * risk_pct / (stop_dist * point_value))``

        Args:
            equity: Current account equity.
            stop_distance: Distance in points from entry to stop.

        Returns:
            Number of contracts (clamped between 1 and max).
        """
        if stop_distance <= 0:
            logger.warning("Stop distance <= 0, defaulting to 1 contract.")
            return 1
        risk_amount = equity * self._risk_pct
        contracts = int(math.floor(risk_amount / (stop_distance * self._point_value)))
        contracts = max(1, min(contracts, self._max_contracts))
        logger.debug(
            "Position size: equity=%.0f risk$=%.0f stop=%.2f → %d contracts",
            equity, risk_amount, stop_distance, contracts,
        )
        return contracts

    # ─────────────────────────────────────────────────────────
    # Trade result tracking
    # ─────────────────────────────────────────────────────────

    def on_trade_closed(self, trade: TradeRecord) -> None:
        """Update risk state after a trade closes.

        Args:
            trade: The completed trade record.
        """
        self._daily_pnl += trade.pnl
        self._daily_trades += 1

        if trade.pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check consecutive loss pause
        if self._consecutive_losses >= self._consec_loss_pause:
            self._pause_until = _time.time() + self._consec_pause_min * 60
            logger.warning(
                "CIRCUIT BREAKER: %d consecutive losses — pausing %d min.",
                self._consecutive_losses, self._consec_pause_min,
            )

        logger.info(
            "Risk update: daily_pnl=%.2f trades=%d consec_losses=%d",
            self._daily_pnl, self._daily_trades, self._consecutive_losses,
        )

    # ─────────────────────────────────────────────────────────
    # Pre-trade checks
    # ─────────────────────────────────────────────────────────

    def can_trade(self, current_equity: float | None = None) -> tuple[bool, str]:
        """Check whether a new trade is allowed.

        Args:
            current_equity: Current account equity (for drawdown check).

        Returns:
            ``(allowed, reason)`` tuple.
        """
        if self._shutdown:
            return False, "bot_shutdown_max_drawdown"

        # Daily loss limit
        if self._starting_equity > 0:
            loss_limit = self._starting_equity * self._daily_loss_pct
            if self._daily_pnl <= -loss_limit:
                return False, f"daily_loss_limit (${self._daily_pnl:.2f})"

        # Daily trade limit
        if self._daily_trades >= self._daily_trade_limit:
            return False, f"daily_trade_limit ({self._daily_trades})"

        # Consecutive loss pause
        if _time.time() < self._pause_until:
            remaining = int(self._pause_until - _time.time())
            return False, f"consecutive_loss_pause ({remaining}s remaining)"

        # Max drawdown
        if current_equity is not None and self._high_water_mark > 0:
            dd = (self._high_water_mark - current_equity) / self._high_water_mark
            if dd >= self._max_dd_pct:
                self._shutdown = True
                logger.critical(
                    "MAX DRAWDOWN reached: %.2f%% (HWM=%.2f, current=%.2f). SHUTTING DOWN.",
                    dd * 100, self._high_water_mark, current_equity,
                )
                return False, f"max_drawdown ({dd:.2%})"

        return True, "ok"

    @property
    def is_shutdown(self) -> bool:
        """Return True if the bot has been shut down due to max drawdown."""
        return self._shutdown

    @property
    def daily_pnl(self) -> float:
        """Current cumulative daily P&L."""
        return self._daily_pnl

    @property
    def daily_trades(self) -> int:
        """Number of round-trip trades today."""
        return self._daily_trades

    @property
    def consecutive_losses(self) -> int:
        """Current consecutive-loss streak."""
        return self._consecutive_losses
