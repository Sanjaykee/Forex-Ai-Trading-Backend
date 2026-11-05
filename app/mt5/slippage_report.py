"""
slippage_report.py
------------------
Query the last 7 days of trades (result IN ['OPEN','WIN','LOSS']) and report
entry slippage (signal_entry vs actual_fill_price) grouped by pair and session.

Migration guard: ensures entry_price and actual_fill_price columns exist in the
trades table before querying (safe no-op if they already exist).

Usage:
    python -m app.mt5.slippage_report          # from backend/
    python app/mt5/slippage_report.py          # direct
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from datetime import datetime, timedelta, timezone
from collections import defaultdict

from sqlalchemy import inspect, text
from app.db.database import engine, SessionLocal
from app.db import models

WARN_THRESHOLD_PIPS = 1.5   # warn if avg slippage for any pair/session exceeds this
LOOKBACK_DAYS       = 7


# ── Migration guard ──────────────────────────────────────────────────────────

def _ensure_columns():
    """
    Add entry_price / actual_fill_price to trades if missing.
    Uses the same pattern as database.sync_table_columns() — safe no-op when
    columns already exist (PostgreSQL IF NOT EXISTS).
    """
    needed = {
        "entry_price":       "FLOAT",
        "actual_fill_price": "FLOAT",
    }
    inspector = inspect(engine)
    if not inspector.has_table("trades"):
        return  # table doesn't exist yet — nothing to migrate
    existing = {c["name"] for c in inspector.get_columns("trades")}
    missing  = {col: typ for col, typ in needed.items() if col not in existing}
    if not missing:
        return
    with engine.connect() as conn:
        for col, typ in missing.items():
            conn.execute(text(
                f"ALTER TABLE trades ADD COLUMN IF NOT EXISTS {col} {typ};"
            ))
        conn.commit()
    print("[migration] Added columns to trades: %s" % ", ".join(missing))


# ── Session derivation ───────────────────────────────────────────────────────

def _session_from_utc(dt: datetime) -> str:
    """Derive trading session from UTC open_time (matches auto_scanner logic)."""
    if dt is None:
        return "Unknown"
    # Normalise to UTC-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    hour    = dt.hour
    weekday = dt.weekday()
    if (weekday == 4 and hour >= 22) or weekday == 5 or (weekday == 6 and hour < 21):
        return "Off-Session"
    if 7  <= hour < 12: return "London"
    if 12 <= hour < 16: return "London/NY"
    if 16 <= hour < 21: return "NewYork"
    return "Asian"


# ── Pip size ─────────────────────────────────────────────────────────────────

def _pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol.upper() else 0.0001


# ── Main report ──────────────────────────────────────────────────────────────

def run():
    _ensure_columns()

    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)

    db = SessionLocal()
    try:
        trades = db.query(models.Trade).filter(
            models.Trade.open_time  >= cutoff,
            models.Trade.result.in_(["OPEN", "WIN", "LOSS"]),
            models.Trade.entry_price.isnot(None),
            models.Trade.actual_fill_price.isnot(None),
        ).all()
    finally:
        db.close()

    if not trades:
        print("No trades with both entry_price and actual_fill_price in the last %d days." % LOOKBACK_DAYS)
        return

    # ── Compute per-trade slippage ────────────────────────────────────────────
    # Slippage = fill - signal_entry, sign-adjusted for direction:
    #   BUY:  positive slippage = filled higher than signal (worse for trader)
    #   SELL: positive slippage = filled lower than signal (worse for trader)
    # We report absolute pips so "worse than 1 pip" is direction-agnostic.

    # group[pair][session] = list of slippage_pips values
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    all_slippages: list[float] = []

    for t in trades:
        raw_slip = t.actual_fill_price - t.entry_price          # price units
        if t.direction and t.direction.upper() == "SELL":
            raw_slip = -raw_slip                                 # flip for SELL
        slip_pips = raw_slip / _pip_size(t.symbol)
        session   = _session_from_utc(t.open_time)
        groups[t.symbol][session].append(slip_pips)
        all_slippages.append(slip_pips)

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    warnings = []

    for pair in sorted(groups):
        for session in sorted(groups[pair]):
            slips      = groups[pair][session]
            n          = len(slips)
            avg        = sum(slips) / n
            mx         = max(slips)
            pct_worse  = round(sum(1 for s in slips if s > 1.0) / n * 100, 1)
            rows.append((pair, session, n, avg, mx, pct_worse))
            if avg > WARN_THRESHOLD_PIPS:
                warnings.append((pair, session, avg))

    # ── Print table ───────────────────────────────────────────────────────────
    HDR = "%-10s %-12s %7s %20s %20s %22s"
    ROW = "%-10s %-12s %7d %20.2f %20.2f %21.1f%%"
    DIV = "-" * 97

    print("")
    print("=" * 97)
    print("  SLIPPAGE REPORT  —  last %d days  (%d trades with fill data)" % (
        LOOKBACK_DAYS, len(trades)))
    print("=" * 97)
    print(HDR % ("Pair", "Session", "Trades",
                 "Avg Slippage (pips)", "Max Slippage (pips)", "% Worse Than 1 pip"))
    print(DIV)

    prev_pair = None
    for pair, session, n, avg, mx, pct in rows:
        if prev_pair and prev_pair != pair:
            print(DIV)
        print(ROW % (pair, session, n, avg, mx, pct))
        prev_pair = pair

    # ── Overall average ───────────────────────────────────────────────────────
    overall_avg = sum(all_slippages) / len(all_slippages)
    print("=" * 97)
    print("  Overall avg slippage : %.2f pips  (%d trades)" % (overall_avg, len(all_slippages)))
    print("=" * 97)

    # ── Warnings ─────────────────────────────────────────────────────────────
    if warnings:
        print("")
        print("  ⚠  WARNING — pair/session combos exceeding %.1f pip avg slippage:" % WARN_THRESHOLD_PIPS)
        for pair, session, avg in sorted(warnings, key=lambda x: -x[2]):
            print("     %-10s %-12s  avg=%.2f pips" % (pair, session, avg))
    else:
        print("  ✓  No pair/session combo exceeds %.1f pip avg slippage." % WARN_THRESHOLD_PIPS)

    print("")


if __name__ == "__main__":
    run()
