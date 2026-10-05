"""
ML Predictor — loads model.pkl and predicts win probability.
Falls back to None if model not trained yet (signal_engine uses rule-based then).
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from app.ml.feature_engineering import extract_features, FEATURE_COLUMNS

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

_cache = {}  # module-level cache so model loads once


def _load_model():
    if "model" not in _cache:
        if not MODEL_PATH.exists():
            return None
        try:
            _cache["model"] = joblib.load(MODEL_PATH)
        except Exception:
            return None
    return _cache.get("model")


def predict_win_probability(df_h4: pd.DataFrame, df_h1: pd.DataFrame,
                            df_m15: pd.DataFrame, confirmations: list,
                            direction: str) -> int | None:
    """
    Returns win probability (0-99) from ML model.
    Returns None if model not available → caller uses rule-based fallback.
    """
    bundle = _load_model()
    if bundle is None:
        return None

    try:
        model    = bundle["model"]
        features = bundle.get("features", FEATURE_COLUMNS)
        feats    = extract_features(df_h4, df_h1, df_m15, confirmations, direction)
        row      = pd.DataFrame([feats])[features].fillna(0)
        prob     = float(model.predict_proba(row)[0][1])  # probability of WIN
        return min(int(prob * 100), 99)
    except Exception:
        return None


def reload_model():
    """Call this after retraining to pick up new model.pkl without restart."""
    _cache.clear()
