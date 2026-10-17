"""
Live Signal Scanner — SMC 3-Pattern Engine
Scans all 9 pairs on M15 and M3 using the new backtest engine logic.
Returns active signals with entry, SL, TP, pattern, score, confidence.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

HTF_MAP = {"M3": "M15", "M15": "H4"}

KILLZONES = {
    "London Open":   (7,  11),
    "New York Open": (12, 17),
}


def _pip_cfg(symbol: str):
    s = symbol.upper()
    if "JPY" in s: return 0.01,   100.0, 4,  25
    if "XAU" in s: return 0.1,     10.0, 5,  40
    return 0.0001, 10000.0, 3, 20


def _in_killzone(hour: int):
    for name, (s, e) in KILLZONES.items():
        if s <= hour < e:
            return True, name
    return False, None


def _session(hour: int) -> str:
    if 7  <= hour <= 11: return "London"
    if 12 <= hour <= 17: return "New York"
    if 0  <= hour <= 4:  return "Asian"
    return "Off-Session"


def _load_live(symbol: str, tf: str, count: int) -> pd.DataFrame:
    """Load latest candles from MetaApi."""
    from app.db.database import SessionLocal
    from app.db.models   import User
    from app.db          import crud
    from app.config      import settings as _cfg

    user = None
    region = "backup-new-york"
    try:
        db   = SessionLocal()
        user = db.query(User).first()
        if user:
            region = crud.get_setting(db, user.id, "metaapi_region") or \
                     _cfg.METAAPI_REGION or "backup-new-york"
        db.close()
    except Exception:
        pass

    token      = (user.metaapi_token      if user else None) or _cfg.METAAPI_TOKEN      or ""
    account_id = (user.metaapi_account_id if user else None) or _cfg.METAAPI_ACCOUNT_ID or ""

    if not token or not account_id:
        return pd.DataFrame()

    try:
        from app.mt5.metaapi_client import MetaApiClient
        from app.analysis.indicators import add_indicators
        client = MetaApiClient(token=token, account_id=account_id, region=region)
        df = client.get_candles(symbol, tf, count=count)
        if df.empty or len(df) < 30:
            return pd.DataFrame()
        t = pd.to_datetime(df["time"], errors="coerce")
        if getattr(t.dt, "tz", None) is not None:
            t = t.dt.tz_convert("UTC").dt.tz_localize(None)
        df["time"] = t
        df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        return add_indicators(df)
    except Exception as e:
        logger.error("Live load %s %s: %s", symbol, tf, e)
        return pd.DataFrame()


def _score(pattern: str, htf_aligned: bool, htf_opposing: bool,
           in_kz: bool, has_ind: bool, has_fvg: bool) -> int:
    base = {"A_CHOCH": 45, "B_BOS_SWEEP": 38, "C_ACCUM": 36}.get(pattern, 35)
    s = base
    if htf_aligned:  s += 20
    if htf_opposing: s -= 10
    if in_kz:        s += 15
    if has_ind:      s += 12
    if has_fvg:      s += 8
    return max(s, 0)


def _confidence(score: int) -> str:
    if score >= 75: return "HIGH"
    if score >= 55: return "MEDIUM"
    return "LOW"


def scan_pair(symbol: str, tf: str = "M15") -> Optional[dict]:
    """
    Scan a single pair on a single timeframe for a live signal.
    Returns signal dict if found, None otherwise.
    """
    from app.analysis.market_structure     import detect_structure
    from app.analysis.smc.bos_choch       import detect_bos_choch
    from app.analysis.smc.order_block     import detect_order_block, detect_accumulation_ob
    from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep, detect_inducement
    from app.analysis.smc.fvg             import detect_fvg

    # Load enough candles for analysis
    count = 200 if tf == "M15" else 400
    df = _load_live(symbol, tf, count)
    if df.empty or len(df) < 60:
        return None

    # Use last candle as "now"
    candle = df.iloc[-1]
    c_time = candle["time"]
    if getattr(c_time, "tzinfo", None) is not None:
        c_time = c_time.tz_convert("UTC").tz_localize(None)

    hour = c_time.hour
    in_kz, kz_name = _in_killzone(hour)
    sess = _session(hour)

    if not in_kz:
        # Outside killzone — still scan but score will be lower (no +15 bonus)
        pass  # do not return None

    # Candle slices
    sl60 = df.tail(60)
    sl40 = df.tail(40)
    sl20 = df.tail(20)

    bos = detect_bos_choch(sl60, lookback=50)

    pattern   = None
    direction = None
    ob        = None

    # Pattern A — CHoCH + OB Retest
    if bos["detected"] and bos["type"] == "CHoCH":
        direction = bos["direction"]
        ob = detect_order_block(sl40, direction, lookback=35)
        if ob["detected"]:
            pattern = "A_CHOCH"

    # Pattern B — BOS + Sweep + OB
    if pattern is None and bos["detected"] and bos["type"] == "BOS":
        direction = bos["direction"]
        sweep = detect_liquidity_sweep(sl40, lookback=25)
        if sweep["detected"] and sweep["direction"] == direction:
            ob = detect_order_block(sl40, direction, lookback=35)
            if ob["detected"]:
                pattern = "B_BOS_SWEEP"

    # Pattern C — Accumulation + OB Retest
    if pattern is None and bos["detected"]:
        direction = bos["direction"]
        ob = detect_accumulation_ob(sl60, direction, lookback=55)
        if ob["detected"]:
            pattern = "C_ACCUM"

    if pattern is None or ob is None or not ob["detected"]:
        return None

    # HTF bias
    htf_tf  = HTF_MAP.get(tf, "H4")
    htf_count = 150 if htf_tf == "H4" else 200
    df_htf  = _load_live(symbol, htf_tf, htf_count)
    htf_bias = detect_structure(df_htf.tail(80)) if not df_htf.empty and len(df_htf) >= 20 else "RANGING"

    htf_aligned  = htf_bias == direction
    htf_opposing = htf_bias not in ("RANGING", direction)

    # Extra confirmations
    ind    = detect_inducement(sl20, direction)
    fvg    = detect_fvg(sl20)
    fvg_ok = fvg["detected"] and fvg["direction"] == direction

    score = _score(pattern, htf_aligned, htf_opposing, in_kz, ind["detected"], fvg_ok)
    if score < 40:
        return None

    conf = _confidence(score)

    # Entry, SL, TP
    pip_unit, pip_mult, min_sl, max_sl = _pip_cfg(symbol)
    atr = float(candle["atr"]) if "atr" in candle.index and not pd.isna(candle["atr"]) \
          else float(np.mean((sl20["high"] - sl20["low"]).values))

    zh, zl = float(ob["zone_high"]), float(ob["zone_low"])
    entry  = round((zh + zl) / 2, 5)
    trade_dir = "BUY" if direction == "BULLISH" else "SELL"

    if trade_dir == "BUY":
        sl_dist  = abs(entry - zl) + pip_unit * 2
        sl_dist  = min(max_sl * pip_unit, max(min_sl * pip_unit, sl_dist))
        sl_price = round(entry - sl_dist, 5)
        tp2      = round(entry + sl_dist * 2.0, 5)
        tp3      = round(entry + sl_dist * 3.0, 5)
        tp5      = round(entry + sl_dist * 5.0, 5)
    else:
        sl_dist  = abs(zh - entry) + pip_unit * 2
        sl_dist  = min(max_sl * pip_unit, max(min_sl * pip_unit, sl_dist))
        sl_price = round(entry + sl_dist, 5)
        tp2      = round(entry - sl_dist * 2.0, 5)
        tp3      = round(entry - sl_dist * 3.0, 5)
        tp5      = round(entry - sl_dist * 5.0, 5)

    sl_pips = round(sl_dist * pip_mult, 1)

    pattern_names = {
        "A_CHOCH":    "CHoCH + OB Retest",
        "B_BOS_SWEEP": "BOS + Liquidity Sweep + OB",
        "C_ACCUM":    "Accumulation Breakout + OB",
    }

    tp_pips = round(sl_dist * 3.0 * pip_mult, 1)   # default display RR 1:3

    confirmations = []
    confirmations.append("CHoCH" if bos.get("type") == "CHoCH" else "BOS")
    confirmations.append(pattern_names.get(pattern, pattern))
    if ind["detected"]: confirmations.append("Inducement")
    if fvg_ok:          confirmations.append("Fair Value Gap")
    if htf_aligned:     confirmations.append(f"HTF {htf_bias} Aligned")
    if kz_name:         confirmations.append(f"Killzone: {kz_name}")

    return {
        # ── UI-compatible fields (Scanner.jsx expects these exact names) ──
        "symbol":        symbol,
        "direction":     trade_dir,
        "current_price": None,          # filled by frontend live price
        "entry_price":   entry,
        "entry_type":    "limit_ob",
        "entry_valid":   True,
        "entry_status":  f"Valid — enter at {entry} (OB 50% midpoint)",
        "stop_loss":     sl_price,
        "take_profit":   tp3,            # default 1:3
        "rr_ratio":      "1:3",
        "sl_pips":       sl_pips,
        "tp_pips":       tp_pips,
        "win_probability": score,        # score used as win probability
        "session":       sess,
        "h4_trend":      htf_bias,
        "h1_trend":      None,
        "lot_size":      None,
        "risk_usd":      1.0,
        "spread":        None,
        "strategy":      pattern_names.get(pattern, pattern),
        "confirmations": confirmations,
        # ── Extra fields for chatbot / advanced display ──
        "timeframe":     tf,
        "pattern":       pattern,
        "pattern_name":  pattern_names.get(pattern, pattern),
        "confidence":    conf,
        "score":         score,
        "htf_bias":      htf_bias,
        "htf_aligned":   htf_aligned,
        "killzone":      kz_name,
        "entry":         entry,
        "sl":            sl_price,
        "tp_1_2":        tp2,
        "tp_1_3":        tp3,
        "tp_1_5":        tp5,
        "ob_high":       zh,
        "ob_low":        zl,
        "inducement":    ind["detected"],
        "fvg":           fvg_ok,
        "signal_time":   str(c_time),
    }


def scan_all_pairs(timeframes=("M15", "M3"), min_confidence="LOW") -> list:
    """
    Scan all 9 pairs on M15 and M3.
    Returns list of active signals sorted by score descending.
    """
    conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    min_rank  = conf_rank.get(min_confidence, 1)
    signals   = []

    for tf in timeframes:
        for pair in PAIRS:
            try:
                sig = scan_pair(pair, tf)
                if sig and conf_rank.get(sig["confidence"], 0) >= min_rank:
                    signals.append(sig)
                    logger.info("SIGNAL %s %s %s score=%d conf=%s",
                                pair, tf, sig["direction"], sig["score"], sig["confidence"])
            except Exception as e:
                logger.error("scan_pair %s %s: %s", pair, tf, e)

    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


def get_active_signals(risk_usd: float = 1.0, rr_ratio: float = 3.0,
                       data_source: str = "auto") -> list:
    """Legacy-compatible wrapper used by signals route."""
    return scan_all_pairs()
