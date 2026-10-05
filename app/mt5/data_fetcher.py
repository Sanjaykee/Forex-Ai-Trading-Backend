import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

TIMEFRAMES = {
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN":  mt5.TIMEFRAME_MN1,
}

def get_candles(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    tf  = TIMEFRAMES.get(timeframe, mt5.TIMEFRAME_M15)
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
