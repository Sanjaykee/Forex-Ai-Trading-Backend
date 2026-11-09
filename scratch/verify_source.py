import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.mt5.source_router import _resolve_user_data_source, get_candles_fetcher

ds = _resolve_user_data_source()
print("Resolved data_source:", ds)

fetcher = get_candles_fetcher()
print("Fetcher module:", getattr(fetcher, "__module__", "?"))
print("Fetcher name:", getattr(fetcher, "__name__", "?"))

df = fetcher("EURUSD", "M15", count=100)
n = 0 if df is None or df.empty else len(df)
print("Candles fetched:", n)
if n:
    print("Last candle time:", df["time"].max())
    print("Columns:", list(df.columns))
