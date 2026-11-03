"""
Download historical candle data via MetaApi and save to CSV.
Date range: 2024-01-01 to 2024-11-03  |  Timeframes: H1, M15, M5, M3
Usage: python download_history.py
"""

import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Allow imports from app/
sys.path.insert(0, os.path.dirname(__file__))

# Load .env.local
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env.local")

from app.mt5.metaapi_client import MetaApiClient

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]
TIMEFRAMES = ["H1", "M15", "M5", "M3"]
START_DT   = datetime(2026, 9, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT     = datetime(2026, 11, 4, 23, 59, 59, tzinfo=timezone.utc)
OUTPUT_DIR = Path(__file__).parent / "data" / "historical_backup"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = MetaApiClient()

if not client.is_configured:
    print("ERROR: MetaApi credentials not configured. Check .env.local")
    sys.exit(1)

print(f"Date range: {START_DT.date()} → {END_DT.date()}")
print(f"Symbols   : {len(SYMBOLS)} | Timeframes: {TIMEFRAMES}\n")

for symbol in SYMBOLS:
    print(f"\n[{symbol}]")
    for tf in TIMEFRAMES:
        print(f"  {tf}...", end=" ", flush=True)
        try:
            df = client.get_candles_range(
                symbol=symbol,
                timeframe=tf,
                start_dt=START_DT,
                end_dt=END_DT,
                warmup_candles=0,
            )
            if df.empty:
                print("No data returned.")
                continue

            if df["time"].dt.tz is not None:
                df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
            start_naive = START_DT.replace(tzinfo=None)
            end_naive   = END_DT.replace(tzinfo=None)
            df = df[(df["time"] >= start_naive) & (df["time"] <= end_naive)]

            out_file = OUTPUT_DIR / f"{symbol}_{tf}_2026-09-01_2026-11-04.csv"
            df.to_csv(out_file, index=False)
            print(f"{len(df)} candles → {out_file.name}")

        except Exception as e:
            print(f"FAILED: {e}")

print("\nAll done.")
