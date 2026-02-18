"""Utility functions: time filters, economic calendar loader, config helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytz
import yaml

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


# ─────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────

def load_config(path: str | Path = "scalper/config.yaml") -> dict[str, Any]:
    """Load and return the YAML configuration file as a dict.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_point_value(symbol: str) -> float:
    """Return the dollar value per point for the given instrument.

    Args:
        symbol: Instrument symbol, e.g. ``"NQ"`` or ``"MNQ"``.

    Returns:
        Point value in USD.
    """
    return 20.0 if symbol.upper() == "NQ" else 2.0


def get_commission(symbol: str) -> float:
    """Return the round-trip commission for one contract.

    Args:
        symbol: Instrument symbol.

    Returns:
        Commission in USD.
    """
    return 4.0 if symbol.upper() == "NQ" else 1.0


def round_to_tick(price: float, tick_size: float = 0.25) -> float:
    """Round a price to the nearest tick increment.

    Args:
        price: Raw price value.
        tick_size: Minimum price increment.

    Returns:
        Price rounded to the nearest tick.
    """
    return round(round(price / tick_size) * tick_size, 2)


# ─────────────────────────────────────────────────────────────
# Time filter
# ─────────────────────────────────────────────────────────────

class TimeFilter:
    """Determines whether the current time is valid for trading.

    Checks RTH session windows, avoids open/close buffers, and
    enforces blackout periods around scheduled economic events.

    Args:
        cfg: Full configuration dictionary (expects ``time_filter`` key).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        tf = cfg["time_filter"]
        self.tz = pytz.timezone(tf["timezone"])
        self.sessions: list[tuple[time, time]] = []
        for s in tf["sessions"]:
            start = datetime.strptime(s["start"], "%H:%M").time()
            end = datetime.strptime(s["end"], "%H:%M").time()
            self.sessions.append((start, end))

        self.avoid_after_open_min: int = tf["avoid_after_open_minutes"]
        self.avoid_before_close_min: int = tf["avoid_before_close_minutes"]
        self.news_blackout_min: int = tf["news_blackout_minutes"]

        cal_path = tf.get("economic_calendar_file", "")
        self.events: list[datetime] = self._load_calendar(cal_path)

    # ── helpers ──────────────────────────────────────────────

    @staticmethod
    def _load_calendar(path: str) -> list[datetime]:
        """Load economic event datetimes from a JSON file.

        Expected format: a JSON array of objects each containing an
        ``"datetime"`` field in ISO-8601 format.

        Args:
            path: Path to the JSON calendar file.

        Returns:
            Sorted list of event datetimes (timezone-aware, ET).
        """
        if not path or not Path(path).exists():
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            events: list[datetime] = []
            for item in data:
                dt = datetime.fromisoformat(item["datetime"])
                if dt.tzinfo is None:
                    dt = ET.localize(dt)
                events.append(dt)
            events.sort()
            return events
        except Exception as exc:
            logger.warning("Failed to load economic calendar: %s", exc)
            return []

    def _rth_open(self, dt: datetime) -> time:
        """Return RTH open time (09:30 ET)."""
        return time(9, 30)

    def _rth_close(self, dt: datetime) -> time:
        """Return RTH close time (16:00 ET)."""
        return time(16, 0)

    # ── public API ───────────────────────────────────────────

    def is_trading_allowed(self, dt: datetime | None = None) -> bool:
        """Check whether a new entry is allowed at the given time.

        Args:
            dt: Datetime to check (defaults to *now* in ET).

        Returns:
            ``True`` if all time-based filters pass.
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)
        else:
            dt = dt.astimezone(self.tz)

        t = dt.time()

        # 1. Must be inside at least one session window
        in_session = any(start <= t <= end for start, end in self.sessions)
        if not in_session:
            return False

        # 2. Avoid first N minutes after RTH open
        rth_open_dt = dt.replace(
            hour=self._rth_open(dt).hour,
            minute=self._rth_open(dt).minute,
            second=0, microsecond=0,
        )
        if dt < rth_open_dt + timedelta(minutes=self.avoid_after_open_min):
            return False

        # 3. Avoid last N minutes before RTH close
        rth_close_dt = dt.replace(
            hour=self._rth_close(dt).hour,
            minute=self._rth_close(dt).minute,
            second=0, microsecond=0,
        )
        if dt > rth_close_dt - timedelta(minutes=self.avoid_before_close_min):
            return False

        # 4. Avoid ±N minutes around economic events
        if self._near_event(dt):
            return False

        return True

    def _near_event(self, dt: datetime) -> bool:
        """Return True if *dt* is within the news blackout window of any event."""
        delta = timedelta(minutes=self.news_blackout_min)
        for ev in self.events:
            if abs(dt - ev) <= delta:
                return True
        return False

    def is_eod_close_time(self, dt: datetime | None = None, eod_time_str: str = "15:50") -> bool:
        """Check whether it's past the end-of-day forced-close time.

        Args:
            dt: Datetime to check (defaults to *now* in ET).
            eod_time_str: ``HH:MM`` string for the cutoff.

        Returns:
            ``True`` if positions should be closed.
        """
        if dt is None:
            dt = datetime.now(self.tz)
        elif dt.tzinfo is None:
            dt = self.tz.localize(dt)
        else:
            dt = dt.astimezone(self.tz)

        eod = datetime.strptime(eod_time_str, "%H:%M").time()
        return dt.time() >= eod
