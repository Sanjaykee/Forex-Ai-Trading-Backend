import pandas as pd
import numpy as np


def extract_features(df_h4: pd.DataFrame, df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                     confirmations: list, direction: str) -> dict:
    """Extract numeric features from candle data + confirmations for ML."""

    def _last(df, col, default=0.0):
        try:
            return float(df[col].iloc[-1])
        except Exception:
            return default

    def _slope(df, col, n=5):
        try:
            s = df[col].dropna().tail(n)
            if len(s) < 2:
                return 0.0
            return float(np.polyfit(range(len(s)), s.values, 1)[0])
        except Exception:
            return 0.0

    conf_set = set(confirmations)

    features = {
        # H4 indicators
        "h4_ema20":      _last(df_h4, "ema20"),
        "h4_rsi":        _last(df_h4, "rsi"),
        "h4_atr":        _last(df_h4, "atr"),
        "h4_close":      _last(df_h4, "close"),
        "h4_ema_slope":  _slope(df_h4, "ema20"),
        "h4_rsi_slope":  _slope(df_h4, "rsi"),

        # H1 indicators
        "h1_ema20":      _last(df_h1, "ema20"),
        "h1_rsi":        _last(df_h1, "rsi"),
        "h1_atr":        _last(df_h1, "atr"),
        "h1_close":      _last(df_h1, "close"),
        "h1_ema_slope":  _slope(df_h1, "ema20"),

        # M15 indicators
        "m15_rsi":       _last(df_m15, "rsi"),
        "m15_atr":       _last(df_m15, "atr"),
        "m15_close":     _last(df_m15, "close"),

        # Price vs EMA (momentum)
        "h4_price_vs_ema": _last(df_h4, "close") - _last(df_h4, "ema20"),
        "h1_price_vs_ema": _last(df_h1, "close") - _last(df_h1, "ema20"),

        # Direction (1=BUY, 0=SELL)
        "direction": 1 if direction == "BUY" else 0,

        # Confirmation flags
        "has_bos":       1 if "BOS confirmed" in conf_set else 0,
        "has_choch":     1 if "CHoCH confirmed" in conf_set else 0,
        "has_ob":        1 if "Order Block" in conf_set else 0,
        "has_fvg":       1 if "Fair Value Gap" in conf_set else 0,
        "has_sweep":     1 if "Liquidity sweep" in conf_set else 0,
        "has_sr":        1 if any("Support" in c or "Resistance" in c for c in conf_set) else 0,
        "has_killzone":  1 if any("ICT Killzone" in c for c in conf_set) else 0,
        "has_crt":       1 if any("CRT" in c for c in conf_set) else 0,
        "has_pa":        1 if any("Price Action" in c for c in conf_set) else 0,
        "conf_count":    len(confirmations),
    }
    return features


FEATURE_COLUMNS = [
    "h4_ema20", "h4_rsi", "h4_atr", "h4_close", "h4_ema_slope", "h4_rsi_slope",
    "h1_ema20", "h1_rsi", "h1_atr", "h1_close", "h1_ema_slope",
    "m15_rsi", "m15_atr", "m15_close",
    "h4_price_vs_ema", "h1_price_vs_ema",
    "direction",
    "has_bos", "has_choch", "has_ob", "has_fvg", "has_sweep",
    "has_sr", "has_killzone", "has_crt", "has_pa", "conf_count",
]
