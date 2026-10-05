"""
ML Trainer — trains LightGBM on MT5 real data or Frankfurter fallback.
Run with MT5:        python -m app.ml.trainer --source mt5
Run with Frankfurter: python -m app.ml.trainer
"""
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.analysis.indicators import add_indicators
from app.analysis.market_structure import detect_structure, detect_trend
from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep
from app.analysis.smc.bos_choch import detect_bos_choch
from app.analysis.smc.order_block import detect_order_block
from app.analysis.smc.fvg import detect_fvg
from app.analysis.support_resistance import detect_sr_zones
from app.analysis.price_action import detect_price_action
from app.analysis.crt import detect_crt
from app.ml.feature_engineering import extract_features, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF",
         "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

RR_RATIO = 2.0


def _label_trade_real(df: pd.DataFrame, idx: int, direction: str, atr: float) -> int:
    """
    Label using REAL high/low from MT5 candles.
    WIN=1 if TP hit before SL in next 20 candles.
    """
    entry = float(df["close"].iloc[idx])
    sl = entry - atr if direction == "BUY" else entry + atr
    tp = entry + atr * RR_RATIO if direction == "BUY" else entry - atr * RR_RATIO

    future = df.iloc[idx + 1: idx + 21]
    for _, row in future.iterrows():
        if direction == "BUY":
            if float(row["low"]) <= sl:
                return 0
            if float(row["high"]) >= tp:
                return 1
        else:
            if float(row["high"]) >= sl:
                return 0
            if float(row["low"]) <= tp:
                return 1
    return 0


def _label_trade_frankfurter(df: pd.DataFrame, idx: int, direction: str, atr: float) -> int:
    """
    Label using close direction only (Frankfurter has fake high/low).
    """
    entry = float(df["close"].iloc[idx])
    target = atr * 0.5
    future_closes = df["close"].iloc[idx + 1: idx + 6]
    if future_closes.empty:
        return 0
    if direction == "BUY":
        max_gain = float(future_closes.max()) - entry
        max_loss = entry - float(future_closes.min())
    else:
        max_gain = entry - float(future_closes.min())
        max_loss = float(future_closes.max()) - entry
    return 1 if max_gain >= target and max_gain > max_loss else 0


def _build_dataset_mt5(symbol: str, get_candles_fn) -> list:
    """Build training samples from real MT5 candles — 5 years with recency weighting."""
    df_h4  = add_indicators(get_candles_fn(symbol, "H4",  5000))  # ~2.8 years
    df_h1  = add_indicators(get_candles_fn(symbol, "H1",  10000)) # ~1.1 years
    df_m15 = add_indicators(get_candles_fn(symbol, "M15", 25000)) # ~6 months

    if df_h4.empty or df_h1.empty or df_m15.empty:
        return []

    rows = []
    min_len = min(len(df_h4), len(df_h1), len(df_m15))
    total   = min_len

    for i in range(20, min_len - 21):
        try:
            h4_slice  = df_h4.iloc[:i + 1]
            h1_slice  = df_h1.iloc[:i + 1]
            m15_slice = df_m15.iloc[:i + 1]

            h4_struct = detect_structure(h4_slice)
            h1_trend  = detect_trend(h1_slice)

            if h4_struct == "RANGING" or h1_trend == "RANGING":
                continue
            if h4_struct != h1_trend:
                continue

            direction = "BUY" if h4_struct == "BULLISH" else "SELL"

            sweep = detect_liquidity_sweep(m15_slice)
            bos   = detect_bos_choch(m15_slice)
            ob    = detect_order_block(m15_slice)
            fvg   = detect_fvg(m15_slice)
            sr    = detect_sr_zones(m15_slice)
            pa    = detect_price_action(m15_slice)
            crt   = detect_crt(m15_slice)

            confirmations = []
            if sweep["detected"] and sweep["direction"] == h4_struct:
                confirmations.append("Liquidity sweep")
            if bos["bos"] and bos["direction"] == h4_struct:
                confirmations.append("BOS confirmed")
            if bos["choch"] and bos["direction"] == h4_struct:
                confirmations.append("CHoCH confirmed")
            if ob["detected"] and ob["direction"] == h4_struct:
                confirmations.append("Order Block")
            if fvg["detected"] and fvg["direction"] == h4_struct:
                confirmations.append("Fair Value Gap")
            if direction == "BUY" and sr["at_support"]:
                confirmations.append("At Support zone")
            if direction == "SELL" and sr["at_resistance"]:
                confirmations.append("At Resistance zone")
            if pa["direction"] == h4_struct:
                confirmations.append(f"Price Action: {pa['pattern']}")
            if crt["detected"] and crt["direction"] == ("BULLISH" if direction == "BUY" else "BEARISH"):
                confirmations.append(f"CRT: {crt['pattern']}")

            if len(confirmations) < 2:
                continue

            feats = extract_features(h4_slice, h1_slice, m15_slice, confirmations, direction)
            atr   = float(m15_slice["atr"].iloc[-1]) if "atr" in m15_slice.columns else 0.001
            label = _label_trade_real(df_m15, i, direction, atr)

            # Recency weight — recent candles are more important
            # Last 25% of data = weight 3, middle 25-75% = weight 2, oldest = weight 1
            position = i / total
            weight = 3 if position > 0.75 else 2 if position > 0.25 else 1

            feats["label"]  = label
            feats["weight"] = weight
            rows.append(feats)

        except Exception:
            continue

    return rows


def _build_dataset_frankfurter(symbol: str) -> list:
    """Build training samples from Frankfurter daily data (fallback)."""
    from app.mt5.yfinance_fetcher import _get_historical, _resample, PAIRS as FX_PAIRS
    base, quote = FX_PAIRS[symbol]
    df_raw = _get_historical(base, quote, days=730)
    if df_raw.empty or len(df_raw) < 50:
        return []

    df_h4  = add_indicators(_resample(df_raw, "4D"))
    df_h1  = add_indicators(_resample(df_raw, "1D"))
    df_m15 = add_indicators(_resample(df_raw, "1D"))

    if df_h4.empty or df_h1.empty or df_m15.empty:
        return []

    rows = []
    min_len = min(len(df_h4), len(df_h1), len(df_m15))

    for i in range(10, min_len - 11):
        try:
            h4_slice  = df_h4.iloc[:i + 1]
            h1_slice  = df_h1.iloc[:i + 1]
            m15_slice = df_m15.iloc[:i + 1]

            h4_struct = detect_structure(h4_slice)
            h1_trend  = detect_trend(h1_slice)

            if h4_struct == "RANGING" or h1_trend == "RANGING":
                continue
            if h4_struct != h1_trend:
                continue

            direction = "BUY" if h4_struct == "BULLISH" else "SELL"

            sweep = detect_liquidity_sweep(m15_slice)
            bos   = detect_bos_choch(m15_slice)
            ob    = detect_order_block(m15_slice)
            fvg   = detect_fvg(m15_slice)
            sr    = detect_sr_zones(m15_slice)
            pa    = detect_price_action(m15_slice)
            crt   = detect_crt(m15_slice)

            confirmations = []
            if sweep["detected"] and sweep["direction"] == h4_struct:
                confirmations.append("Liquidity sweep")
            if bos["bos"] and bos["direction"] == h4_struct:
                confirmations.append("BOS confirmed")
            if bos["choch"] and bos["direction"] == h4_struct:
                confirmations.append("CHoCH confirmed")
            if ob["detected"] and ob["direction"] == h4_struct:
                confirmations.append("Order Block")
            if fvg["detected"] and fvg["direction"] == h4_struct:
                confirmations.append("Fair Value Gap")
            if direction == "BUY" and sr["at_support"]:
                confirmations.append("At Support zone")
            if direction == "SELL" and sr["at_resistance"]:
                confirmations.append("At Resistance zone")
            if pa["direction"] == h4_struct:
                confirmations.append(f"Price Action: {pa['pattern']}")
            if crt["detected"] and crt["direction"] == ("BULLISH" if direction == "BUY" else "BEARISH"):
                confirmations.append(f"CRT: {crt['pattern']}")

            if len(confirmations) < 2:
                continue

            feats = extract_features(h4_slice, h1_slice, m15_slice, confirmations, direction)
            atr   = float(m15_slice["atr"].iloc[-1]) if "atr" in m15_slice.columns else 0.001
            label = _label_trade_frankfurter(df_m15, i, direction, atr)

            feats["label"] = label
            rows.append(feats)

        except Exception:
            continue

    return rows


def _get_candles_from_storage(symbol: str, timeframe: str, n: int = 50000) -> pd.DataFrame:
    sample_dir = Path(__file__).resolve().parents[2] / "sample_data"
    csv_file = sample_dir / f"{symbol}_{timeframe}.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            df["time"] = pd.to_datetime(df["time"])
            return df.tail(n).reset_index(drop=True)
        except Exception:
            pass
    from app.mt5.data_fetcher import get_candles
    return get_candles(symbol, timeframe, n)


def train(source: str = "mt5"):
    sample_dir = Path(__file__).resolve().parents[2] / "sample_data"
    has_local_data = any(sample_dir.glob("*.csv"))

    use_mt5 = source == "mt5"
    candle_fn = _get_candles_from_storage

    if use_mt5 and not has_local_data:
        try:
            from app.db.database import SessionLocal
            from app.db.models import User
            from app.mt5.connection import connect

            db = SessionLocal()
            user = db.query(User).filter(User.mt5_login != None).first()
            db.close()

            if user and connect(int(user.mt5_login), user.mt5_password, user.mt5_server):
                print("MT5 connected successfully.")
            else:
                print("MT5 connection failed. Falling back to Frankfurter.")
                use_mt5 = False
        except Exception as e:
            print(f"MT5 not available: {e}. Falling back to Frankfurter.")
            use_mt5 = False

    source_name = "MT5 real historical data" if (use_mt5 or has_local_data) else "Frankfurter daily"
    print(f"Collecting training data from {source_name}...")

    all_rows = []
    for symbol in PAIRS:
        print(f"  Processing {symbol}...", end=" ", flush=True)
        try:
            if use_mt5 or has_local_data:
                rows = _build_dataset_mt5(symbol, candle_fn)
            else:
                rows = _build_dataset_frankfurter(symbol)
            print(f"{len(rows)} samples")
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error: {e}")
            continue

    if use_mt5 and not has_local_data:
        import MetaTrader5 as mt5
        mt5.shutdown()

    if len(all_rows) < 20:
        print(f"Not enough data ({len(all_rows)} samples). Need at least 20.")
        return

    df = pd.DataFrame(all_rows)
    X  = df[FEATURE_COLUMNS].fillna(0)
    y  = df["label"]
    # Use recency weights if available (MT5 training)
    sample_weights = df["weight"].values if "weight" in df.columns else None

    print(f"\nTotal samples: {len(df)} | WIN: {y.sum()} | LOSS: {(y==0).sum()}")

    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=15,
        min_child_samples=10,
        reg_alpha=0.3,
        reg_lambda=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )

    n_splits = min(5, int(y.value_counts().min()))
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

    model.fit(X, y, sample_weight=sample_weights)
    joblib.dump({"model": model, "features": FEATURE_COLUMNS, "source": source_name}, MODEL_PATH)
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Source: {source_name}")
    print("Retrain with MT5:        python -m app.ml.trainer --source mt5")
    print("Retrain with Frankfurter: python -m app.ml.trainer")


if __name__ == "__main__":
    source = "frankfurter"
    if "--source" in sys.argv:
        idx = sys.argv.index("--source")
        if idx + 1 < len(sys.argv):
            source = sys.argv[idx + 1]
    train(source)
