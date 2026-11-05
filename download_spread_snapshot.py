"""
Spread Snapshot Downloader
Fetches current candles (H1, M5, M3) + live spread for all 9 pairs
and saves/appends to CSV in data/spread_backup/.

Each row = one candle with spread_pips and fetch_time columns added.

Usage:
    python download_spread_snapshot.py          # manual one-shot run
    spread_snapshot_loop()                      # called automatically by main.py on startup

Runs automatically every hour when the FastAPI app is running.
Also runnable manually for a one-shot snapshot.

Output files:
    data/spread_backup/{SYMBOL}_{TF}_spread.csv

Columns:
    time, open, high, low, close, volume, spread_pips, fetch_time, session
"""

import sys
import os
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env.local")

import pandas as pd
from app.mt5.metaapi_client import MetaApiClient

logger = logging.getLogger(__name__)

# How often to take a snapshot (seconds). 3600 = every 1 hour.
SNAPSHOT_INTERVAL_S = 3600

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS    = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
              "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]
TIMEFRAMES = ["H1", "M5", "M3"]
CANDLE_COUNT = {
    "H1": 500,
    "M5": 500,
    "M3": 400,
}
OUTPUT_DIR = Path(__file__).parent / "data" / "spread_backup"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Session helper ────────────────────────────────────────────────────────────
def _get_session_name(utc_hour: int) -> str:
    """Map UTC hour to session name matching ALLOWED_SESSIONS in mtf_strategy.py."""
    if 0  <= utc_hour < 4:  return "Asian"
    if 7  <= utc_hour < 11: return "London"
    if 12 <= utc_hour < 17: return "New York"
    return "Off-Session"


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    client = MetaApiClient()

    if not client.is_configured:
        print("ERROR: MetaApi credentials not configured. Check .env.local")
        sys.exit(1)

    now_utc    = datetime.now(timezone.utc)
    fetch_time = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    session    = _get_session_name(now_utc.hour)

    print("=" * 60)
    print(f"  SPREAD SNAPSHOT  {fetch_time}")
    print(f"  Session: {session}")
    print("=" * 60)

    for symbol in SYMBOLS:
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        pip_mult = 100.0 if "JPY" in symbol else 10000.0

        # ── Fetch live spread via bid/ask ─────────────────────────────────────
        price = client.get_live_price(symbol)
        if price and price.get("bid") and price.get("ask"):
            bid          = float(price["bid"])
            ask          = float(price["ask"])
            spread_pips  = round((ask - bid) / pip_size, 2)
        else:
            # Fallback: use fixed backtest spread if live price unavailable
            FALLBACK = {
                "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.6,
                "AUDUSD": 1.4, "NZDUSD": 2.0, "USDCAD": 1.8,
                "EURJPY": 1.8, "GBPJPY": 2.5,
            }
            spread_pips = FALLBACK.get(symbol, 1.5)
            bid = ask = None
            print(f"  [{symbol}] live price unavailable — using fallback spread {spread_pips} pips")

        print(f"\n  [{symbol}]  spread={spread_pips:.2f} pips  bid={bid}  ask={ask}")

        for tf in TIMEFRAMES:
            count = CANDLE_COUNT.get(tf, 500)
            print(f"    {tf}...", end=" ", flush=True)

            try:
                df = client.get_candles(symbol, tf, count=count)

                if df.empty:
                    print("no data")
                    continue

                # Normalise time to tz-naive UTC
                if df["time"].dt.tz is not None:
                    df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

                df = df.sort_values("time").reset_index(drop=True)

                # Add spread and fetch metadata columns
                df["spread_pips"] = spread_pips
                df["fetch_time"]  = fetch_time
                df["session"]     = session

                out_file = OUTPUT_DIR / f"{symbol}_{tf}_spread.csv"

                if out_file.exists():
                    # Append only new candles (avoid duplicates by time)
                    existing = pd.read_csv(out_file, parse_dates=["time"])
                    combined = pd.concat([existing, df], ignore_index=True)
                    combined = (
                        combined
                        .drop_duplicates(subset=["time"], keep="last")
                        .sort_values("time")
                        .reset_index(drop=True)
                    )
                    combined.to_csv(out_file, index=False)
                    new_rows = len(combined) - len(existing)
                    print(f"{len(df)} fetched  +{new_rows} new rows  total={len(combined)}  → {out_file.name}")
                else:
                    df.to_csv(out_file, index=False)
                    print(f"{len(df)} candles saved  → {out_file.name}")

            except Exception as e:
                print(f"FAILED: {e}")

    print("\n" + "=" * 60)
    print("  Snapshot complete.")
    print(f"  Files saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Print spread summary ──────────────────────────────────────────────────
    print("\n  Live Spread Summary:")
    print(f"  {'Pair':<10} {'Spread (pips)':>14}  Session")
    print("  " + "-" * 38)
    for symbol in SYMBOLS:
        pip_size = 0.01 if "JPY" in symbol else 0.0001
        price    = client.get_live_price(symbol)
        if price and price.get("bid") and price.get("ask"):
            sp = round((float(price["ask"]) - float(price["bid"])) / pip_size, 2)
        else:
            sp = None
        sp_str = f"{sp:.2f}" if sp else "N/A"
        print(f"  {symbol:<10} {sp_str:>14}  {session}")


async def spread_snapshot_loop():
    """Background task registered in main.py startup_event.
    Takes a spread snapshot immediately on startup, then every SNAPSHOT_INTERVAL_S.
    """
    logger.info("[SpreadSnapshot] Background loop started — interval=%ds", SNAPSHOT_INTERVAL_S)
    # Small delay so MetaApi connection is ready before first fetch
    await asyncio.sleep(60)
    while True:
        try:
            run()
        except Exception as e:
            logger.error("[SpreadSnapshot] Snapshot failed: %s", e)
        await asyncio.sleep(SNAPSHOT_INTERVAL_S)


if __name__ == "__main__":
    run()
