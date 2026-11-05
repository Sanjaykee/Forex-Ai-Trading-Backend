"""
compare_paper_vs_backtest.py
Compare paper_trades DB rows against backtest signals for the same date range.

Usage:
    python compare_paper_vs_backtest.py [START_DATE] [END_DATE]
    Dates in YYYY-MM-DD format. Defaults: 2026-10-01 / 2026-10-31

A signal is considered MATCHED when both paper and backtest agree on:
    symbol + direction + same H1 candle hour (rounded to the hour).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime, timezone
import pandas as pd

from app.db.database import SessionLocal
from app.db import models
from app.analysis.smc.mtf_strategy import run_mtf_smc_backtest

PAIRS     = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
             "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "historical_backup")
MIN_SCORE = 11
MIN_RR    = 3.0

START_DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-10-01"
END_DATE   = sys.argv[2] if len(sys.argv) > 2 else "2026-10-31"


def _load_csv(symbol, tf):
    path = os.path.join(DATA_DIR, f"{symbol}_{tf}_2026-09-01_2026-11-04.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["time"])
    return df.sort_values("time").reset_index(drop=True)


def _backtest_signals() -> list[dict]:
    """Run backtest for all pairs and return flat list of signal dicts."""
    signals = []
    for symbol in PAIRS:
        df_h1 = _load_csv(symbol, "H1")
        df_m5 = _load_csv(symbol, "M5")
        df_m3 = _load_csv(symbol, "M3")
        if df_h1.empty or df_m5.empty or df_m3.empty:
            continue
        result = run_mtf_smc_backtest(
            df15=df_h1, df5=df_m5, df3=df_m3,
            symbol=symbol,
            start_date=START_DATE,
            end_date=END_DATE,
            min_score=MIN_SCORE,
            min_rr=MIN_RR,
            apply_daily_protection=True,
            max_daily_trades=3,
            max_daily_loss_r=2.0,
            max_consec_losses=2,
        )
        for t in result.get("all_trades", []):
            # Normalise timestamp to a naive UTC datetime truncated to the hour
            ts = t.get("timestamp") or t.get("entry_time")
            if ts is None:
                continue
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            hour_key = ts.replace(minute=0, second=0, microsecond=0,
                                  tzinfo=None)
            signals.append({
                "symbol":    symbol,
                "direction": t["direction"],
                "hour_key":  hour_key,
            })
    return signals


def _paper_signals(start: datetime, end: datetime) -> list[dict]:
    """Load paper_trades rows from DB within the date range."""
    db = SessionLocal()
    try:
        rows = db.query(models.PaperTrade).filter(
            models.PaperTrade.signal_ts >= start,
            models.PaperTrade.signal_ts <= end,
        ).all()
        result = []
        for row in rows:
            ts = row.signal_ts
            if ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            result.append({
                "symbol":    row.symbol,
                "direction": row.direction,
                "hour_key":  hour_key,
                "signal_id": row.signal_id,
            })
        return result
    finally:
        db.close()


def _make_key(s: dict) -> str:
    return "%s|%s|%s" % (s["symbol"], s["direction"],
                         s["hour_key"].strftime("%Y-%m-%d %H"))


def run():
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE,   "%Y-%m-%d").replace(
                    hour=23, minute=59, second=59)

    print("=" * 60)
    print("  PAPER vs BACKTEST COMPARISON")
    print("  Range : %s  to  %s" % (START_DATE, END_DATE))
    print("=" * 60)

    print("\n  [1/2] Running backtest for all pairs...")
    bt_signals = _backtest_signals()
    bt_keys    = {_make_key(s) for s in bt_signals}

    print("  [2/2] Loading paper signals from DB...")
    paper_signals = _paper_signals(start_dt, end_dt)
    paper_keys    = {_make_key(s) for s in paper_signals}

    matched    = bt_keys & paper_keys
    bt_only    = bt_keys - paper_keys
    paper_only = paper_keys - bt_keys

    total_bt    = len(bt_keys)
    total_paper = len(paper_keys)
    match_count = len(matched)
    mismatch    = len(bt_only) + len(paper_only)
    denom       = max(total_bt, total_paper, 1)
    alignment   = round(match_count / denom * 100, 1)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print("  Total paper signals  : %d" % total_paper)
    print("  Total backtest signals: %d" % total_bt)
    print("  Matched (both agree) : %d" % match_count)
    print("  Mismatched           : %d  (BT-only=%d  Paper-only=%d)" % (
        mismatch, len(bt_only), len(paper_only)))
    print("  Alignment %%          : %.1f%%" % alignment)
    print("=" * 60)

    if bt_only:
        print("\n  Backtest-only signals (not seen in paper):")
        for k in sorted(bt_only)[:20]:
            print("    %s" % k)
        if len(bt_only) > 20:
            print("    ... and %d more" % (len(bt_only) - 20))

    if paper_only:
        print("\n  Paper-only signals (not in backtest):")
        for k in sorted(paper_only)[:20]:
            print("    %s" % k)
        if len(paper_only) > 20:
            print("    ... and %d more" % (len(paper_only) - 20))


if __name__ == "__main__":
    run()
