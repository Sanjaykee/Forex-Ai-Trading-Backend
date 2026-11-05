"""
TradeMonitor — broker reconciliation & safety net.
Runs every 60 seconds and enforces:
  1. Orphan detection  — broker position with no DB record → close immediately.
  2. Ghost detection   — DB record marked OPEN but no broker position → mark RECONCILED_MISSING.
  3. SL/TP reattach    — broker position missing SL or TP → re-attach from DB record.
  4. Timeout close     — any open trade older than 12 hours → force-close via market order.
  5. Full error safety — loop body wrapped in try/except, monitor never crashes.
  6. Clean stop        — stop() sets a flag that exits the loop gracefully.
"""

import os
import time
import threading
from datetime import datetime, timezone, timedelta

from app.core.logger import get_logger
from app.db.database import SessionLocal
from app.db import models

logger = get_logger(__name__)

# Maximum age of an open trade before it is force-closed.
# Override via environment: MAX_HOLD_HOURS=8
#
# Default raised 12 -> 24 for backtest parity. run_mtf_smc_backtest replays up to
# 400 M3 bars (~20 hours) looking for the 3R target, and in the Oct-2026 sample
# 11% of trades ran past 12 hours (max 62-68h). A 12h cut truncated exactly the
# trades the backtest counted as wins, so the live equity curve could not match
# the validated one. 24h still bounds exposure but no longer fights the target.
MAX_HOLD_HOURS = int(os.environ.get("MAX_HOLD_HOURS", "24"))

# Keep the old name as an alias so any external references still resolve.
TRADE_TIMEOUT_HOURS = MAX_HOLD_HOURS

# How often the monitor runs (seconds)
MONITOR_INTERVAL_S = 60


# Comments we stamp on our own orders. Orphan auto-close is restricted to
# positions carrying one of these tags, so the monitor can never close a
# manually-opened position or one belonging to another system on the same account.
OUR_ORDER_TAGS = ("VariantC", "ForexAI")


def _is_our_position(pos: dict) -> bool:
    comment = str(pos.get("comment") or "")
    return any(tag in comment for tag in OUR_ORDER_TAGS)


def _get_client(user):
    """Return a configured MetaApiClient for the given user, or None."""
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(
        token=user.metaapi_token,
        account_id=user.metaapi_account_id,
    )
    return client if client.is_configured else None


def _force_close(client, position_id: str, symbol: str, reason: str) -> bool:
    """Close a broker position via market order. Returns True on success."""
    result = client.close_order(position_id)
    if result.get("success"):
        logger.warning(
            "[TradeMonitor] %s position %s force-closed — reason: %s",
            symbol, position_id, reason,
        )
        return True
    logger.error(
        "[TradeMonitor] %s position %s force-close FAILED — reason: %s — error: %s",
        symbol, position_id, reason, result.get("error", "unknown"),
    )
    return False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_open_time(raw) -> datetime | None:
    """Parse broker openTime string to tz-aware UTC datetime."""
    if not raw:
        return None
    try:
        ts = str(raw).replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class TradeMonitor:
    """
    Synchronous trade monitor that runs in its own daemon thread.

    Usage:
        monitor = TradeMonitor()
        monitor.start()          # launches background thread
        ...
        monitor.stop()           # signals the loop to exit cleanly
    """

    def __init__(self):
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the monitor in a background daemon thread."""
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name="trade-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("[TradeMonitor] Started (interval=%ds, timeout=%dh).",
                    MONITOR_INTERVAL_S, MAX_HOLD_HOURS)

    def stop(self) -> None:
        """Signal the loop to exit after the current iteration completes."""
        self._stop_flag.set()
        logger.info("[TradeMonitor] Stop requested — will exit after current cycle.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """
        Loop every MONITOR_INTERVAL_S seconds.
        The entire loop body is wrapped in try/except so the monitor
        never crashes regardless of what goes wrong inside.
        """
        logger.info("[TradeMonitor] run_forever() entered.")
        while not self._stop_flag.is_set():
            try:
                self._run_cycle()
            except Exception as exc:
                # Requirement 6: catch everything, log, continue
                logger.error(
                    "[TradeMonitor] Unhandled exception in cycle: %s",
                    exc, exc_info=True,
                )
            # Sleep in small increments so stop() is responsive
            for _ in range(MONITOR_INTERVAL_S):
                if self._stop_flag.is_set():
                    break
                time.sleep(1)

        logger.info("[TradeMonitor] Exited cleanly.")

    # ── Single reconciliation cycle ───────────────────────────────────────────

    def _run_cycle(self) -> None:
        db = SessionLocal()
        try:
            users = db.query(models.User).filter(
                models.User.metaapi_token.isnot(None),
                models.User.metaapi_token != "",
                models.User.metaapi_account_id.isnot(None),
                models.User.metaapi_account_id != "",
            ).all()

            for user in users:
                try:
                    self._reconcile_user(db, user)
                except Exception as user_exc:
                    logger.error(
                        "[TradeMonitor] Error reconciling user %d: %s",
                        user.id, user_exc, exc_info=True,
                    )
        finally:
            db.close()

    def _reconcile_user(self, db, user) -> None:
        client = _get_client(user)
        if client is None:
            return

        # ── Step 1: fetch all open broker positions ───────────────────────────
        # None means the broker state is UNKNOWN (API error / rate limit).
        # Skipping the cycle is the only safe response: treating it as "empty"
        # would mark every live trade as a ghost and orphan every position.
        broker_positions = client.get_open_positions()
        if broker_positions is None:
            logger.warning(
                "[TradeMonitor] Broker position state UNKNOWN for user %d — "
                "skipping reconciliation this cycle.", user.id,
            )
            return
        broker_by_ticket = {p["id"]: p for p in broker_positions}

        # ── Step 2: orphan detection ──────────────────────────────────────────
        # Broker position exists but no matching DB trade record.
        # Restricted to positions tagged by us — see OUR_ORDER_TAGS.
        db_tickets = {
            t.mt5_ticket
            for t in db.query(models.Trade).filter(
                models.Trade.user_id == user.id,
                models.Trade.mt5_ticket.isnot(None),
            ).all()
        }

        for ticket, pos in broker_by_ticket.items():
            if ticket not in db_tickets:
                if not _is_our_position(pos):
                    logger.info(
                        "[TradeMonitor] Untracked position ticket=%s symbol=%s "
                        "comment=%r is not ours — leaving it alone.",
                        ticket, pos["symbol"], pos.get("comment", ""),
                    )
                    continue
                logger.critical(
                    "[TradeMonitor] ORPHAN POSITION detected — ticket=%s symbol=%s "
                    "volume=%.2f openPrice=%.5f — closing immediately.",
                    ticket, pos["symbol"], pos["volume"], pos["openPrice"],
                )
                _force_close(client, ticket, pos["symbol"], "ORPHAN_POSITION")

        # ── Step 3: ghost detection ───────────────────────────────────────────
        # DB record is OPEN but broker position no longer exists
        open_db_trades = db.query(models.Trade).filter(
            models.Trade.user_id == user.id,
            models.Trade.result == "OPEN",
        ).all()

        for trade in open_db_trades:
            ticket = trade.mt5_ticket
            if ticket and ticket not in broker_by_ticket:
                logger.warning(
                    "[TradeMonitor] GHOST TRADE — DB trade #%d %s %s ticket=%s "
                    "not found on broker — marking RECONCILED_MISSING.",
                    trade.id, trade.symbol, trade.direction, ticket,
                )
                trade.result     = "RECONCILED_MISSING"
                trade.close_time = datetime.utcnow()
                db.commit()

        # ── Steps 4 & 5: SL/TP reattach + timeout close ──────────────────────
        # Re-query after ghost cleanup so we only process genuinely open trades
        open_db_trades = db.query(models.Trade).filter(
            models.Trade.user_id == user.id,
            models.Trade.result == "OPEN",
        ).all()

        now = _utc_now()

        for trade in open_db_trades:
            ticket = trade.mt5_ticket
            if not ticket or ticket not in broker_by_ticket:
                continue

            pos = broker_by_ticket[ticket]

            # ── Step 5: timeout close (> 12 hours) ───────────────────────────
            open_time_raw = trade.open_time
            if open_time_raw is not None:
                # Normalise to tz-aware UTC
                if getattr(open_time_raw, "tzinfo", None) is None:
                    open_time_utc = open_time_raw.replace(tzinfo=timezone.utc)
                else:
                    open_time_utc = open_time_raw.astimezone(timezone.utc)

                age_hours = (now - open_time_utc).total_seconds() / 3600
                if age_hours >= MAX_HOLD_HOURS:
                    logger.info(
                        "[TradeMonitor] TIMEOUT pair=%s ticket=%s hours_open=%.1f "
                        "(limit=%dh) — force-closing.",
                        trade.symbol, ticket, age_hours, MAX_HOLD_HOURS,
                    )
                    closed = _force_close(
                        client, ticket, trade.symbol, "TIMEOUT"
                    )
                    if closed:
                        trade.result     = "TIMEOUT"
                        trade.close_time = datetime.utcnow()
                        db.commit()
                    continue  # skip SL/TP check for this trade

            # ── Step 4: SL/TP reattach ────────────────────────────────────────
            broker_sl = pos.get("stopLoss", 0)
            broker_tp = pos.get("takeProfit", 0)
            db_sl     = trade.stop_loss  or 0
            db_tp     = trade.take_profit or 0

            sl_missing = (not broker_sl or broker_sl == 0)
            tp_missing = (not broker_tp or broker_tp == 0)

            if (sl_missing or tp_missing) and db_sl and db_tp:
                result = client.modify_position(
                    position_id=ticket,
                    sl=db_sl,
                    tp=db_tp,
                )
                if result.get("success"):
                    logger.warning(
                        "[TradeMonitor] SL/TP REATTACHED — trade #%d %s %s "
                        "ticket=%s SL=%.5f TP=%.5f (broker had SL=%s TP=%s).",
                        trade.id, trade.symbol, trade.direction,
                        ticket, db_sl, db_tp, broker_sl, broker_tp,
                    )
                else:
                    logger.error(
                        "[TradeMonitor] SL/TP reattach FAILED — trade #%d %s "
                        "ticket=%s error=%s",
                        trade.id, trade.symbol, ticket,
                        result.get("error", "unknown"),
                    )
