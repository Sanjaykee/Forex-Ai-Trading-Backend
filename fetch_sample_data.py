r"""
Fetch maximum available candle data from MT5 for all pairs and timeframes.
Saves to: backend/sample_data/<SYMBOL>_<TIMEFRAME>.csv

Run: venv\Scripts\python.exe fetch_sample_data.py
"""
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF",
         "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

# Timeframes to fetch — M1/M5/M15/M30/H1 may be limited on demo
# H4 and D1 confirmed 11+ years available
TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

DATE_FROM = datetime(2015, 1, 1, tzinfo=timezone.utc)
DATE_TO   = datetime.now(timezone.utc)

OUTPUT_DIR = Path(__file__).parent / "sample_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def fetch_all():
    import MetaTrader5 as mt5
    from app.db.database import SessionLocal
    from app.db.models import User

    db = SessionLocal()
    user = db.query(User).filter(User.mt5_login != None).first()
    db.close()

    if not user or not all([user.mt5_login, user.mt5_password, user.mt5_server]):
        print("MT5 credentials not found. Connect via UI first.")
        return

    MT5_PATH = "C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe"

    if not mt5.initialize(path=MT5_PATH, login=int(user.mt5_login), password=user.mt5_password, server=user.mt5_server):
        print("MT5 initialize failed:", mt5.last_error())
        return

    if mt5.account_info() is None:
        print("MT5 account not found:", mt5.last_error())
        mt5.shutdown()
        return

    print(f"Using credentials for user: {user.email} (id={user.id})")
    print(f"Fetching {len(PAIRS)} pairs x {len(TIMEFRAMES)} timeframes from {DATE_FROM.date()} to {DATE_TO.date()}...\n")

    TF_MAP = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }

    total_saved = 0
    skipped     = 0

    for symbol in PAIRS:
        for tf in TIMEFRAMES:
            try:
                rates = mt5.copy_rates_range(symbol, TF_MAP[tf], DATE_FROM, DATE_TO)
                if rates is None or len(rates) == 0:
                    print(f"  {symbol} {tf:<5}: NO DATA (history not downloaded in MT5 terminal)")
                    skipped += 1
                    continue

                df = pd.DataFrame(rates)
                df["time"] = pd.to_datetime(df["time"], unit="s")
                df.rename(columns={"tick_volume": "volume"}, inplace=True)
                df = df[["time", "open", "high", "low", "close", "volume"]]

                path = OUTPUT_DIR / f"{symbol}_{tf}.csv"
                df.to_csv(path, index=False)

                start = df["time"].iloc[0].date()
                end   = df["time"].iloc[-1].date()
                years = round((df["time"].iloc[-1] - df["time"].iloc[0]).days / 365, 1)
                print(f"  {symbol} {tf:<5}: {len(df):>6} candles | {start} -> {end} ({years} yrs)")
                total_saved += 1

            except Exception as e:
                print(f"  {symbol} {tf:<5}: ERROR - {e}")
                skipped += 1

    mt5.shutdown()
    print(f"\nDone! {total_saved} files saved, {skipped} skipped.")
    print(f"Location: {OUTPUT_DIR}")

    if skipped > 0:
        print("\nTo fix skipped timeframes (M1/M5/M15/M30/H1):")
        print("  1. Open MT5 terminal")
        print("  2. Open EURUSD chart for each timeframe")
        print("  3. Press Ctrl+Home to scroll back and download history")
        print("  4. Run this script again")


if __name__ == "__main__":
    fetch_all()
