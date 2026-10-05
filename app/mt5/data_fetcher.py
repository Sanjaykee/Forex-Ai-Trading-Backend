try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False

import pandas as pd
from datetime import datetime

TIMEFRAMES = {
    "M1":  getattr(mt5, "TIMEFRAME_M1", 1) if mt5 else 1,
    "M2":  getattr(mt5, "TIMEFRAME_M2", 2) if mt5 else 2,
    "M3":  getattr(mt5, "TIMEFRAME_M3", 3) if mt5 else 3,
    "M4":  getattr(mt5, "TIMEFRAME_M4", 4) if mt5 else 4,
    "M5":  getattr(mt5, "TIMEFRAME_M5", 5) if mt5 else 5,
    "M6":  getattr(mt5, "TIMEFRAME_M6", 6) if mt5 else 6,
    "M10": getattr(mt5, "TIMEFRAME_M10", 10) if mt5 else 10,
    "M12": getattr(mt5, "TIMEFRAME_M12", 12) if mt5 else 12,
    "M15": getattr(mt5, "TIMEFRAME_M15", 15) if mt5 else 15,
    "M20": getattr(mt5, "TIMEFRAME_M20", 20) if mt5 else 20,
    "M30": getattr(mt5, "TIMEFRAME_M30", 30) if mt5 else 30,
    "H1":  getattr(mt5, "TIMEFRAME_H1", 16385) if mt5 else 16385,
    "H2":  getattr(mt5, "TIMEFRAME_H2", 16386) if mt5 else 16386,
    "H3":  getattr(mt5, "TIMEFRAME_H3", 16387) if mt5 else 16387,
    "H4":  getattr(mt5, "TIMEFRAME_H4", 16388) if mt5 else 16388,
    "H6":  getattr(mt5, "TIMEFRAME_H6", 16390) if mt5 else 16390,
    "H8":  getattr(mt5, "TIMEFRAME_H8", 16392) if mt5 else 16392,
    "H12": getattr(mt5, "TIMEFRAME_H12", 16396) if mt5 else 16396,
    "D1":  getattr(mt5, "TIMEFRAME_D1", 16408) if mt5 else 16408,
    "W1":  getattr(mt5, "TIMEFRAME_W1", 32769) if mt5 else 32769,
    "MN":  getattr(mt5, "TIMEFRAME_MN1", 49153) if mt5 else 49153,
}

def get_candles(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    if not HAS_MT5 or mt5 is None:
        from app.mt5.yfinance_fetcher import get_candles as yf_candles
        return yf_candles(symbol, timeframe, count)
    tf  = TIMEFRAMES.get(timeframe, getattr(mt5, "TIMEFRAME_M15", 15))
    raw = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(raw)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]]

def get_live_price(symbol: str) -> dict:
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        return {}
    return {"bid": tick.bid, "ask": tick.ask, "spread": round((tick.ask - tick.bid) / mt5.symbol_info(symbol).point, 1)}

def get_symbol_info(symbol: str) -> dict:
    info = mt5.symbol_info(symbol)
    if not info:
        return {}
    return {
        "point":        info.point,
        "digits":       info.digits,
        "trade_contract_size": info.trade_contract_size,
        "volume_min":   info.volume_min,
        "volume_step":  info.volume_step,
    }

def get_all_pairs() -> list:
    return [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"
    ]
