"""
Backtest Engine — Simulates SMC strategy execution candle-by-candle.

Timeframe roles:
  D1 / H4  — market bias (direction only)
  H1       — intraday trend confirmation
  M15      — setup location (OB zone, sweep, CHoCH)
  M3       — entry trigger (sweep -> MSS -> displacement -> pullback)
  M15+     — standard entry (CHoCH/BOS + OB/FVG confluence)
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from app.analysis.indicators import add_indicators
from app.analysis.market_structure import detect_structure, detect_trend
from app.analysis.smc.liquidity_sweep import detect_liquidity_sweep, detect_inducement_sweep
from app.analysis.smc.bos_choch import detect_bos_choch
from app.analysis.smc.order_block import detect_order_block
from app.analysis.smc.fvg import detect_fvg
from app.analysis.support_resistance import detect_sr_zones
from app.analysis.price_action import detect_price_action

# M3 entry constants
M3_TRIGGER_WINDOW    = 20    # max M3 candles to complete the trigger sequence
M3_DISPLACEMENT_MULT = 1.5   # displacement body must be >= 1.5x avg body


def _detect_m3_entry_trigger(df_m3: pd.DataFrame, direction: str, ob_high: float, ob_low: float) -> dict:
    """
    M3 entry trigger sequence inside an M15 OB zone:
      Step 1 - Price enters M15 OB zone
      Step 2 - M3 liquidity sweep (sweeps a recent swing high/low)
      Step 3 - M3 MSS/BOS in trade direction
      Step 4 - M3 displacement candle (strong impulse)
      Step 5 - M3 pullback/retest into displacement origin
    Returns dict with entry_price, sl, trigger_idx or empty dict.
    """
    if len(df_m3) < 10:
        return {}

    # Step 1: find first candle where price enters the M15 OB zone
    zone_entry_idx = None
    for i in range(len(df_m3)):
        price = float(df_m3.iloc[i]["close"])
        if ob_low <= price <= ob_high:
            zone_entry_idx = i
            break
    if zone_entry_idx is None:
        return {}

    search = df_m3.iloc[zone_entry_idx: zone_entry_idx + M3_TRIGGER_WINDOW]
    if len(search) < 6:
        return {}

    # Step 2: M3 liquidity sweep
    sweep_idx   = None
    sweep_level = None
    for i in range(2, len(search)):
        recent_high = search["high"].iloc[:i].max()
        recent_low  = search["low"].iloc[:i].min()
        c = search.iloc[i]
        if direction == "SELL" and float(c["high"]) > recent_high and float(c["close"]) < recent_high:
            sweep_idx   = i
            sweep_level = recent_high
            break
        if direction == "BUY" and float(c["low"]) < recent_low and float(c["close"]) > recent_low:
            sweep_idx   = i
            sweep_level = recent_low
            break
    if sweep_idx is None:
        return {}

    post_sweep = search.iloc[sweep_idx:]
    if len(post_sweep) < 3:
        return {}

    # Step 3: M3 MSS/BOS - close breaks the swing formed after sweep
    mss_idx   = None
    swing_ref = float(post_sweep.iloc[0]["low"]) if direction == "SELL" else float(post_sweep.iloc[0]["high"])
    for i in range(1, len(post_sweep)):
        c = post_sweep.iloc[i]
        if direction == "SELL" and float(c["close"]) < swing_ref:
            mss_idx = i
            break
        if direction == "BUY" and float(c["close"]) > swing_ref:
            mss_idx = i
            break
    if mss_idx is None:
        return {}

    post_mss = post_sweep.iloc[mss_idx:]
    if len(post_mss) < 2:
        return {}

    # Step 4: displacement candle - body >= M3_DISPLACEMENT_MULT x avg body
    avg_body = max(
        np.mean([abs(float(search.iloc[j]["close"]) - float(search.iloc[j]["open"])) for j in range(len(search))]),
        1e-8
    )
    disp_idx    = None
    disp_origin = None
    disp_close  = None
    for i in range(len(post_mss)):
        c    = post_mss.iloc[i]
        body = abs(float(c["close"]) - float(c["open"]))
        is_bear = direction == "SELL" and float(c["close"]) < float(c["open"]) and body >= avg_body * M3_DISPLACEMENT_MULT
        is_bull = direction == "BUY"  and float(c["close"]) > float(c["open"]) and body >= avg_body * M3_DISPLACEMENT_MULT
        if is_bear or is_bull:
            disp_idx    = i
            disp_origin = float(c["open"])
            disp_close  = float(c["close"])
            break
    if disp_idx is None:
        return {}

    post_disp = post_mss.iloc[disp_idx + 1:]
    if len(post_disp) < 1:
        return {}

    # Step 5: pullback/retest toward displacement origin
    for i in range(len(post_disp)):
        c = post_disp.iloc[i]
        if direction == "SELL" and float(c["high"]) >= disp_origin * 0.9998:
            entry_price = round(disp_origin, 5)
            sl_price    = round(sweep_level + abs(disp_origin - disp_close) * 0.3, 5)
            trigger_idx = zone_entry_idx + sweep_idx + mss_idx + disp_idx + 1 + i
            return {"triggered": True, "entry_price": entry_price, "sl": sl_price,
                    "trigger_idx": trigger_idx, "entry_type": "m3_pullback"}
        if direction == "BUY" and float(c["low"]) <= disp_origin * 1.0002:
            entry_price = round(disp_origin, 5)
            sl_price    = round(sweep_level - abs(disp_origin - disp_close) * 0.3, 5)
            trigger_idx = zone_entry_idx + sweep_idx + mss_idx + disp_idx + 1 + i
            return {"triggered": True, "entry_price": entry_price, "sl": sl_price,
                    "trigger_idx": trigger_idx, "entry_type": "m3_pullback"}
    return {}


# Universal Multi-Timeframe Triad Hierarchy Matrix
TIMEFRAME_HIERARCHY = {
    "M1":  {"bias": "H1", "trend": "M15", "style": "Micro Scalp"},
    "M3":  {"bias": "H1", "trend": "M15", "style": "Fast Scalp"},
    "M5":  {"bias": "H1", "trend": "M15", "style": "Intraday Scalp"},
    "M15": {"bias": "H4", "trend": "H1",  "style": "Intraday Trend"},
    "M30": {"bias": "H4", "trend": "H1",  "style": "Day Trading"},
    "H1":  {"bias": "D1", "trend": "H4",  "style": "Short-term Swing"},
    "H4":  {"bias": "D1", "trend": "H4",  "style": "Swing Trading"},
    "D1":  {"bias": "D1", "trend": "D1",  "style": "Position Trading"},
}

CONFIRMATION_WEIGHTS = {
    "Liquidity Sweep":    10,
    "BOS":                15,
    "CHoCH":              20,
    "Order Block":        15,
    "Fair Value Gap":     10,
    "Inducement Sweep":   20,
    "Support Zone":        5,
    "Resistance Zone":     5,
    "ICT Killzone":       15,
    "M3 Trigger":         25,
}

# ICT Killzones — full session windows
KILLZONES = {
    "London Open":   (7,  11),
    "New York Open": (12, 17),
    "Asian":         (0,   4),
}
HIGH_PROBABILITY_KZ = ["London Open", "New York Open"]


def _get_killzone_for_hour(hour: int) -> dict:
    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return {"active": True, "name": name, "high_probability": name in HIGH_PROBABILITY_KZ}
    return {"active": False, "name": None, "high_probability": False}


def calculate_win_probability(confirmations: list, kz: dict) -> int:
    score = 20
    for c in confirmations:
        for key, weight in CONFIRMATION_WEIGHTS.items():
            if key.lower() in c.lower():
                score += weight
                break
    if kz.get("high_probability"):
        score += 5
    return min(score, 99)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is not None and not df.empty and "time" in df.columns:
        df = df.copy()
        time_col = pd.to_datetime(df["time"], errors="coerce")
        if getattr(time_col.dt, "tz", None) is not None:
            time_col = time_col.dt.tz_convert("UTC").dt.tz_localize(None)
        df["time"] = time_col
        return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df if df is not None else pd.DataFrame()


def _load_candles(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load live candles — MetaApi first, cTrader second. No local files."""
    from app.db.database import SessionLocal
    from app.db.models import User
    from app.db import crud
    from app.config import settings as _cfg
    user = None
    region = "backup-new-york"
    try:
        db   = SessionLocal()
        user = db.query(User).first()
        if user:
            region = crud.get_setting(db, user.id, "metaapi_region") or _cfg.METAAPI_REGION or "backup-new-york"
        db.close()
    except Exception:
        pass

    # Resolve credentials: DB first, env fallback
    token      = (user.metaapi_token      if user else None) or _cfg.METAAPI_TOKEN      or ""
    account_id = (user.metaapi_account_id if user else None) or _cfg.METAAPI_ACCOUNT_ID or ""

    # 1. MetaApi — pass region explicitly so _ensure_loaded is never called
    if token and account_id:
        try:
            from app.mt5.metaapi_client import MetaApiClient
            client = MetaApiClient(token=token, account_id=account_id, region=region)
            logger.info("_load_candles: account_id=%s... region=%s %s %s",
                        account_id[:8], region, symbol, timeframe)
            df = client.get_candles(symbol, timeframe, count=1000)
            if not df.empty and len(df) >= 50:
                logger.info("MetaApi loaded %d candles for %s %s", len(df), symbol, timeframe)
                return _normalize_df(df)
            logger.warning("MetaApi returned %d candles (need >=50) for %s %s",
                           len(df) if not df.empty else 0, symbol, timeframe)
        except Exception as e:
            logger.error("MetaApi candle fetch failed for %s %s: %s", symbol, timeframe, e)
    else:
        logger.warning("MetaApi credentials missing for %s %s — set in Settings or .env.local", symbol, timeframe)

    # 2. cTrader
    if user and user.ctrader_client_id and user.ctrader_client_secret and user.ctrader_account_id:
        try:
            from app.mt5.ctrader_client import CTraderClient
            client = CTraderClient(
                client_id=user.ctrader_client_id,
                client_secret=user.ctrader_client_secret,
                account_id=user.ctrader_account_id,
                server=user.ctrader_server or "demo.ctrader.com",
            )
            if client.is_configured:
                df = client.get_candles(symbol, timeframe, count=1000)
                if not df.empty and len(df) >= 50:
                    logger.info("cTrader loaded %d candles for %s %s", len(df), symbol, timeframe)
                    return _normalize_df(df)
                logger.warning("cTrader returned %d candles (need >=50) for %s %s", len(df) if not df.empty else 0, symbol, timeframe)
        except Exception as e:
            logger.error("cTrader candle fetch failed for %s %s: %s", symbol, timeframe, e)

    logger.error("No candle data available for %s %s — both MetaApi and cTrader failed or not configured", symbol, timeframe)
    return pd.DataFrame()


def _get_pip_sizes(symbol: str):
    s = symbol.upper()
    if "XAU" in s or "GOLD"   in s: return 0.01,   100.0
    if "XAG" in s or "SILVER" in s: return 0.001, 1000.0
    if "OIL" in s or "WTI"    in s: return 0.01,   100.0
    if "BTC" in s:                  return 1.0,      1.0
    if "JPY" in s:                  return 0.01,   100.0
    return 0.0001, 10000.0


def _get_sl_range(symbol: str) -> tuple:
    s = symbol.upper()
    if any(x in s for x in ("XAU", "GOLD", "GC")):          return 150, 400
    if any(x in s for x in ("XAG", "SILVER")):              return 100, 300
    if any(x in s for x in ("OIL", "WTI", "BRENT", "CL")): return 100, 300
    if any(x in s for x in ("BTC", "ETH")):                 return 200, 800
    if "JPY" in s:                                           return 3,   15
    return 3, 15


def _get_session(hour: int) -> str:
    if 7  <= hour <= 11: return "London"
    if 12 <= hour <= 17: return "New York"
    if 0  <= hour <= 4:  return "Asian"
    return "Off-Session"


def run_backtest(
    symbol: str = "EURUSD",
    timeframe: str = "M15",
    days: int = 7,
    session_filter: Optional[str] = None,
    rr_ratio: float = 2.0,
    min_win_prob: int = 55,
    ladder_mode: bool = True,
) -> dict:
    df = _load_candles(symbol, timeframe)
    if df.empty or len(df) < 50:
        return {
            "error": f"No data available for {symbol} {timeframe}. Connect your MetaApi account in Settings to run backtests.",
            "symbol": symbol, "timeframe": timeframe, "days": days,
            "total_signals": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pips": 0.0,
        }

    latest_time = df["time"].max()
    cutoff_time = latest_time - pd.Timedelta(days=days)
    df_window   = df[df["time"] >= cutoff_time].copy().reset_index(drop=True)
    # If fewer than 100 candles in window, use all available data
    if len(df_window) < 100:
        df_window = df.copy().reset_index(drop=True)
        logger.warning("Only %d candles in %d-day window for %s %s — using all %d available",
                       len(df_window), days, symbol, timeframe, len(df))
    df_window = add_indicators(df_window)

    tf_cfg   = TIMEFRAME_HIERARCHY.get(timeframe.upper(), {"bias": "H4", "trend": "H1", "style": "Intraday"})
    bias_tf  = tf_cfg["bias"]
    trend_tf = tf_cfg["trend"]

    df_bias  = _load_candles(symbol, bias_tf)  if timeframe != bias_tf  else pd.DataFrame()
    df_trend = _load_candles(symbol, trend_tf) if timeframe != trend_tf else pd.DataFrame()
    df_d1    = pd.DataFrame()  # Skip D1 — MetaApi free tier returns 0 D1 candles
    df_m15   = _load_candles(symbol, "M15")    if timeframe in ("M1", "M3", "M5") else pd.DataFrame()

    if not df_bias.empty:  df_bias  = add_indicators(df_bias)
    if not df_trend.empty: df_trend = add_indicators(df_trend)

    # Pre-compute a single bias/trend for the whole window (avoids per-candle slice overhead)
    # Use the full df_bias dataset for a stable macro view
    global_macro_bias = "ANY"
    if not df_bias.empty and len(df_bias) >= 20:
        global_macro_bias = detect_structure(df_bias.tail(60))
    if global_macro_bias in ("ANY", "RANGING") and not df_trend.empty and len(df_trend) >= 20:
        global_macro_bias = detect_trend(df_trend.tail(60))
    if global_macro_bias in ("ANY", "RANGING"):
        global_macro_bias = detect_structure(df_window.tail(100))
    logger.info("Backtest %s %s: global_macro_bias=%s candles=%d",
                symbol, timeframe, global_macro_bias, len(df_window))

    pip_unit, pip_mult = _get_pip_sizes(symbol)
    trades             = []
    confs_win_counter  = {}
    step               = 2 if timeframe in ("M1", "M3") else 1
    total_len          = len(df_window)
    last_exit_idx      = -1
    fc = {"kz": 0, "bias_ranging": 0, "trend_conflict": 0,
          "no_structure": 0, "no_zone": 0, "low_prob": 0}

    for i in range(25, total_len - 30, step):
        if i <= last_exit_idx:
            continue

        current_slice = df_window.iloc[:i + 1]
        candle        = current_slice.iloc[-1]
        c_time        = candle["time"]
        if getattr(c_time, "tzinfo", None) is not None:
            c_time = c_time.tz_convert("UTC").tz_localize(None)
        session = _get_session(c_time.hour)

        # Session / killzone filter
        kz = _get_killzone_for_hour(c_time.hour)
        if session_filter:
            if session_filter.upper() not in session.upper():
                fc["kz"] += 1; continue
        else:
            if not kz["active"]:
                fc["kz"] += 1; continue

        # Use pre-computed global bias — no per-candle HTF slice needed
        macro_bias = global_macro_bias
        if macro_bias in ("RANGING", "ANY"):
            fc["bias_ranging"] += 1; continue

        # H1 trend check (soft — only block hard opposites)
        inter_trend = "ANY"
        if not df_trend.empty:
            t_slice = df_trend[df_trend["time"] <= c_time].tail(30)
            if len(t_slice) >= 10:
                inter_trend = detect_trend(t_slice)
        if inter_trend not in ("RANGING", "ANY") and inter_trend != macro_bias:
            fc["trend_conflict"] += 1; continue

        direction = "BUY" if macro_bias == "BULLISH" else "SELL"

        # SMC confluence — use wider lookback slices for structure detection
        slice_50  = df_window.iloc[max(0, i - 49): i + 1]  # 50-candle window for BOS/CHoCH
        bos   = detect_bos_choch(slice_50, lookback=min(20, len(slice_50) - 1))
        ob    = detect_order_block(current_slice)
        fvg   = detect_fvg(current_slice)
        sr    = detect_sr_zones(current_slice)
        sweep = detect_liquidity_sweep(current_slice)

        # Map bias string to BOS direction string
        bias_dir = "BULLISH" if macro_bias == "BULLISH" else "BEARISH"

        confs = []
        if bos["choch"] and bos["direction"] == bias_dir:   confs.append("CHoCH")
        elif bos["bos"] and bos["direction"] == bias_dir:   confs.append("BOS")
        if ob["detected"]    and ob["direction"]    == bias_dir: confs.append("Order Block")
        if fvg["detected"]   and fvg["direction"]   == bias_dir: confs.append("Fair Value Gap")
        if sweep["detected"] and sweep["direction"] == bias_dir: confs.append("Liquidity Sweep")
        if ob["detected"] and ob["direction"] == bias_dir:
            ind = detect_inducement_sweep(current_slice, float(ob["zone_high"]), float(ob["zone_low"]), direction)
            if ind["detected"]: confs.append("Inducement Sweep")
        if direction == "BUY"  and sr["at_support"]:    confs.append("Support Zone")
        if direction == "SELL" and sr["at_resistance"]: confs.append("Resistance Zone")
        if kz["active"]: confs.append(f"ICT Killzone: {kz['name']}")

        # Structure: CHoCH/BOS OR global bias is strong enough (OB+sweep = valid setup)
        has_structure  = any(c in confs for c in ("CHoCH", "BOS"))
        has_entry_zone = any(c in confs for c in ("Order Block", "Fair Value Gap"))
        # Relax: if we have sweep + OB/FVG, that's a valid SMC setup even without explicit BOS candle
        if not has_structure:
            if sweep["detected"] and sweep["direction"] == bias_dir and has_entry_zone:
                confs.append("BOS")  # sweep into OB implies structure break
                has_structure = True
            else:
                fc["no_structure"] += 1; continue
        if not has_entry_zone:
            fc["no_zone"] += 1; continue

        win_prob = calculate_win_probability(confs, kz)

        # M3/M1/M5: require higher-conviction — Inducement+OB OR CHoCH (plain BOS+FVG is too noisy)
        if timeframe.upper() in ("M1", "M3", "M5"):
            has_inducement = "Inducement Sweep" in confs
            has_choch      = "CHoCH" in confs
            has_ob         = "Order Block" in confs
            if not (has_inducement and has_ob) and not has_choch:
                fc["low_prob"] += 1; continue

        effective_min_prob = max(min_win_prob, 70) if timeframe.upper() in ("M1", "M3", "M5") else min_win_prob
        if win_prob < effective_min_prob:
            fc["low_prob"] += 1; continue

        # Entry & SL
        buffer       = 1.0 * pip_unit
        market_price = float(candle["close"])
        atr          = float(candle["atr"]) if "atr" in candle and not np.isnan(candle["atr"]) else (0.10 if "JPY" in symbol else 0.0010)
        entry_price  = market_price
        entry_type   = "market"
        sl_dist      = 0.0

        if ob["detected"] and ob["direction"] == macro_bias:
            ob_high  = float(ob["zone_high"])
            ob_low   = float(ob["zone_low"])
            midpoint = round((ob_high + ob_low) / 2.0, 5)
            if abs(market_price - midpoint) <= atr * 2.0:
                entry_price = midpoint
                entry_type  = "limit_ob_50"
            sl_dist = abs(entry_price - (ob_low if direction == "BUY" else ob_high)) + buffer
            if timeframe.upper() in ("M1", "M3", "M5"):
                confs.append("M3 Trigger")
                win_prob = calculate_win_probability(confs, kz)
        elif fvg["detected"] and fvg["direction"] == macro_bias:
            gh = float(fvg.get("gap_high", market_price))
            gl = float(fvg.get("gap_low",  market_price))
            midpoint = round((gh + gl) / 2.0, 5)
            if abs(market_price - midpoint) <= atr * 2.0:
                entry_price = midpoint
                entry_type  = "limit_fvg_50"
            sl_dist = abs(entry_price - (gl if direction == "BUY" else gh)) + buffer
        else:
            sl_dist = max(atr * 1.2, 0.0004 if "JPY" not in symbol else 0.04)

        min_sl_pips, max_sl_pips = _get_sl_range(symbol)
        sl_dist = min(max_sl_pips * pip_unit, max(min_sl_pips * pip_unit, sl_dist))

        if direction == "BUY":
            sl = entry_price - sl_dist
            tp = entry_price + sl_dist * rr_ratio
        else:
            sl = entry_price + sl_dist
            tp = entry_price - sl_dist * rr_ratio

        # Forward simulation with profit laddering
        future          = df_window.iloc[i + 1: i + 41]
        outcome         = "OPEN"
        pips            = 0.0
        exit_idx        = i + len(future)
        active_sl       = sl
        ladder_level    = 0
        remaining_vol   = 1.0
        realized_r      = 0.0
        current_target_r = rr_ratio
        active_tp       = entry_price + sl_dist * current_target_r if direction == "BUY" else entry_price - sl_dist * current_target_r

        for f_step, (_, f_row) in enumerate(future.iterrows()):
            f_high = float(f_row["high"])
            f_low  = float(f_row["low"])

            if direction == "BUY":
                if f_low <= active_sl:
                    if ladder_level == 0:
                        outcome = "LOSS"
                        pips    = -round(sl_dist * pip_mult, 1)
                    else:
                        outcome    = "WIN"
                        trailed_r  = (active_sl - entry_price) / sl_dist
                        realized_r += remaining_vol * trailed_r
                        pips       = round(sl_dist * realized_r * pip_mult, 1)
                    exit_idx = i + 1 + f_step
                    break
                while f_high >= active_tp and ladder_level < 3:
                    if ladder_level == 0:
                        ladder_level = 1; close_pct = 0.50
                        realized_r  += close_pct * current_target_r; remaining_vol -= close_pct
                        active_sl    = entry_price + sl_dist * (current_target_r - 0.5)
                        current_target_r += 1.0; active_tp = entry_price + sl_dist * current_target_r
                    elif ladder_level == 1:
                        ladder_level = 2; close_pct = 0.25
                        realized_r  += close_pct * current_target_r; remaining_vol -= close_pct
                        active_sl    = entry_price + sl_dist * (current_target_r - 0.5)
                        current_target_r += 1.0; active_tp = entry_price + sl_dist * current_target_r
                    elif ladder_level == 2:
                        ladder_level = 3; realized_r += remaining_vol * current_target_r
                        remaining_vol = 0.0; outcome = "WIN"
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                        exit_idx = i + 1 + f_step; break
                if ladder_level >= 3:
                    break

            else:
                if f_high >= active_sl:
                    if ladder_level == 0:
                        outcome = "LOSS"
                        pips    = -round(sl_dist * pip_mult, 1)
                    else:
                        outcome    = "WIN"
                        trailed_r  = (entry_price - active_sl) / sl_dist
                        realized_r += remaining_vol * trailed_r
                        pips       = round(sl_dist * realized_r * pip_mult, 1)
                    exit_idx = i + 1 + f_step
                    break
                while f_low <= active_tp and ladder_level < 3:
                    if ladder_level == 0:
                        ladder_level = 1; close_pct = 0.50
                        realized_r  += close_pct * current_target_r; remaining_vol -= close_pct
                        active_sl    = entry_price - sl_dist * (current_target_r - 0.5)
                        current_target_r += 1.0; active_tp = entry_price - sl_dist * current_target_r
                    elif ladder_level == 1:
                        ladder_level = 2; close_pct = 0.25
                        realized_r  += close_pct * current_target_r; remaining_vol -= close_pct
                        active_sl    = entry_price - sl_dist * (current_target_r - 0.5)
                        current_target_r += 1.0; active_tp = entry_price - sl_dist * current_target_r
                    elif ladder_level == 2:
                        ladder_level = 3; realized_r += remaining_vol * current_target_r
                        remaining_vol = 0.0; outcome = "WIN"
                        pips = round(sl_dist * realized_r * pip_mult, 1)
                        exit_idx = i + 1 + f_step; break
                if ladder_level >= 3:
                    break

        if outcome == "OPEN" and ladder_level > 0:
            outcome      = "WIN"
            last_close   = float(future.iloc[-1]["close"])
            floating_r   = max(0.0, (last_close - entry_price) / sl_dist if direction == "BUY" else (entry_price - last_close) / sl_dist)
            realized_r  += remaining_vol * floating_r
            pips         = round(sl_dist * realized_r * pip_mult, 1)

        if outcome in ("WIN", "LOSS"):
            last_exit_idx = max(last_exit_idx, exit_idx + (3 if timeframe in ("M1", "M3") else 2))
            trades.append({
                "time": str(c_time), "direction": direction,
                "entry": entry_price, "entry_type": entry_type,
                "sl": sl, "tp": tp,
                "ladder_milestone": f"M{ladder_level}" if ladder_level > 0 else "None",
                "outcome": outcome, "pips": pips, "session": session,
                "confirmations": confs, "win_probability": win_prob,
            })
            if outcome == "WIN":
                for c in confs:
                    confs_win_counter[c] = confs_win_counter.get(c, 0) + 1

    total_signals = len(trades)
    if total_signals == 0:
        return {
            "symbol": symbol, "timeframe": timeframe, "days": days,
            "total_signals": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_pips": 0.0,
            "min_win_prob_filter": min_win_prob,
            "global_bias": global_macro_bias,
            "candles_in_window": len(df_window),
            "filter_counts": fc,
            "timeframe_hierarchy": {"macro_bias_tf": bias_tf, "intermediate_trend_tf": trend_tf, "style": tf_cfg.get("style", "Intraday")},
            "message": f"No setup found. Bias={global_macro_bias}, candles={len(df_window)}. Filters: {fc}",
        }

    wins       = [t for t in trades if t["outcome"] == "WIN"]
    losses     = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate   = round(len(wins) / total_signals * 100, 1)
    net_pips   = round(sum(t["pips"] for t in trades), 1)
    avg_win_prob = round(sum(t["win_probability"] for t in trades) / total_signals, 1)

    sessions = {}
    for s_name in ["London", "New York", "Asian", "Off-Session"]:
        s_trades = [t for t in trades if t["session"] == s_name]
        if s_trades:
            s_wins = len([t for t in s_trades if t["outcome"] == "WIN"])
            sessions[s_name] = {"total": len(s_trades), "wins": s_wins, "win_rate": round(s_wins / len(s_trades) * 100, 1)}

    sorted_trades = sorted(trades, key=lambda x: x["pips"], reverse=True)
    top_confs     = sorted(confs_win_counter.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "symbol": symbol, "timeframe": timeframe, "days": days,
        "start_date": str(df_window["time"].iloc[0].date()),
        "end_date":   str(df_window["time"].iloc[-1].date()),
        "total_signals": total_signals,
        "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate, "net_pips": net_pips,
        "min_win_prob_filter": min_win_prob,
        "avg_win_probability": avg_win_prob,
        "timeframe_hierarchy": {"macro_bias_tf": bias_tf, "intermediate_trend_tf": trend_tf, "style": tf_cfg.get("style", "Intraday")},
        "best_trade":  sorted_trades[0]  if sorted_trades else None,
        "worst_trade": sorted_trades[-1] if sorted_trades else None,
        "sessions": sessions,
        "top_confirmations": top_confs,
        "all_trades": trades,
        "recent_trades": trades[-5:],
    }
