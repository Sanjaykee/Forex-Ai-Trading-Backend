import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MetaTrader5 as mt5
from app.db.database import SessionLocal
from app.db.models import User
import pandas as pd
from datetime import datetime, timezone

db = SessionLocal()
user = db.query(User).filter(User.mt5_login != None).first()
db.close()

print(f"Login: {user.mt5_login} | Server: {user.mt5_server}")

if not mt5.initialize():
    print("MT5 initialize failed:", mt5.last_error())
    sys.exit(1)

if not mt5.login(login=int(user.mt5_login), password=user.mt5_password, server=user.mt5_server):
    print("MT5 login failed:", mt5.last_error())
    mt5.shutdown()
    sys.exit(1)

acc = mt5.account_info()
print(f"Connected: {acc.login} | Balance: {acc.balance} {acc.currency}\n")

TIMEFRAMES = [
    ("M1",  mt5.TIMEFRAME_M1),
    ("M5",  mt5.TIMEFRAME_M5),
    ("M15", mt5.TIMEFRAME_M15),
    ("M30", mt5.TIMEFRAME_M30),
    ("H1",  mt5.TIMEFRAME_H1),
    ("H4",  mt5.TIMEFRAME_H4),
    ("D1",  mt5.TIMEFRAME_D1),
]

date_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
date_to   = datetime.now(timezone.utc)

print("Max available candles for EURUSD:")
print("-" * 55)
for tf_name, tf in TIMEFRAMES:
    rates = mt5.copy_rates_range("EURUSD", tf, date_from, date_to)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        start = df["time"].iloc[0].date()
        end   = df["time"].iloc[-1].date()
        years = round((df["time"].iloc[-1] - df["time"].iloc[0]).days / 365, 1)
        print(f"  {tf_name:<5} {len(rates):>7} candles | {start} -> {end} ({years} yrs)")
    else:
        print(f"  {tf_name:<5} NO DATA | error: {mt5.last_error()}")

mt5.shutdown()
