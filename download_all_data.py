"""
Download complete historical datasets from ICMarkets MT5 for all 9 pairs across all timeframes.
Timeframes: M1, M3, M5, M15, M30, H1, H4, D1
Output: backend/sample_data/<SYMBOL>_<TIMEFRAME>.csv
"""
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as mt5

MT5_PATH = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"
OUTPUT_DIR = Path(__file__).parent / "sample_data"
OUTPUT_DIR.mkdir(exist_ok=True)

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"
]

TIMEFRAMES = [
    ("D1",  mt5.TIMEFRAME_D1,  "range"),
    ("H4",  mt5.TIMEFRAME_H4,  "range"),
    ("H1",  mt5.TIMEFRAME_H1,  50000),
    ("M30", mt5.TIMEFRAME_M30, 50000),
    ("M15", mt5.TIMEFRAME_M15, 50000),
    ("M5",  mt5.TIMEFRAME_M5,  50000),
    ("M3",  mt5.TIMEFRAME_M3,  50000),
    ("M1",  mt5.TIMEFRAME_M1,  50000),
]

DATE_FROM = datetime(2015, 1, 1, tzinfo=timezone.utc)
DATE_TO   = datetime.now(timezone.utc)


def download_all():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from app.db.database import SessionLocal
    from app.db.models import User

    db = SessionLocal()
    user = db.query(User).filter(User.mt5_login != None).first()
    db.close()

    if not user:
        print("Error: No user with MT5 credentials found in database.")
        return

    print("=" * 60)
    print(f"Connecting to ICMarkets MT5 for account: {user.mt5_login}...")
    if not mt5.initialize(path=MT5_PATH, login=int(user.mt5_login), password=user.mt5_password, server=user.mt5_server, timeout=30000):
        print(f"MT5 initialize failed: {mt5.last_error()}")
        return

    acc = mt5.account_info()
    if not acc:
        print(f"Failed to read account info: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Connected: Account {acc.login} on {acc.server} (Balance: ${acc.balance})")
    print(f"Target pairs: {len(PAIRS)} | Timeframes: {len(TIMEFRAMES)}")
    print("=" * 60 + "\n")

    total_files = 0
    total_candles = 0

    for symbol in PAIRS:
        print(f"[{symbol}] Synchronizing Market Watch...")
        if not mt5.symbol_select(symbol, True):
            print(f"  Warning: Failed to select {symbol} in Market Watch.")

        # Give terminal a brief moment to sync server history
        time.sleep(0.5)

        for tf_name, tf_code, fetch_type in TIMEFRAMES:
            try:
                if fetch_type == "range":
                    rates = mt5.copy_rates_range(symbol, tf_code, DATE_FROM, DATE_TO)
                else:
                    rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, fetch_type)

                if rates is None or len(rates) == 0:
                    # Retry once if needed
                    time.sleep(0.5)
                    if fetch_type == "range":
                        rates = mt5.copy_rates_range(symbol, tf_code, DATE_FROM, DATE_TO)
                    else:
                        rates = mt5.copy_rates_from_pos(symbol, tf_code, 0, fetch_type)

                if rates is None or len(rates) == 0:
                    print(f"  {symbol} {tf_name:<5}: NO DATA")
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
                df = df[["time", "open", "high", "low", "close", "volume"]]

                file_path = OUTPUT_DIR / f"{symbol}_{tf_name}.csv"
                df.to_csv(file_path, index=False)

                t0 = df["time"].iloc[0].strftime("%Y-%m-%d")
                t1 = df["time"].iloc[-1].strftime("%Y-%m-%d")
                print(f"  {symbol} {tf_name:<5}: {len(df):>6} candles ({t0} -> {t1}) -> {file_path.name}")

                total_files += 1
                total_candles += len(df)

            except Exception as e:
                print(f"  {symbol} {tf_name:<5}: Error: {e}")

        print()

    mt5.shutdown()
    print("=" * 60)
    print(f"COMPLETED! Saved {total_files} files with {total_candles:,} total candles.")
    print(f"Location: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    download_all()
