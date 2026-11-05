"""
Live Signal Scanner — SMC 3-Pattern Engine
Scans all 9 pairs on M15 and M3 using the new backtest engine logic.
Returns active signals with entry, SL, TP, pattern, score, confidence.
"""
import logging
import os
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime, timezone, timedelta
from app.analysis.smc.mtf_strategy import evaluate_mtf_smc_setup

# ── H1 bias freshness gate ────────────────────────────────────────────────────
# The backtest (run_mtf_smc_backtest) has NO freshness gate: it slices the H1
# frame to `time <= t_m5` and uses the last closed H1 bar, however old it is.
# For live/backtest parity this gate is therefore DISABLED by default.
# Set H1_FRESHNESS_GATE=true to re-enable it (max age measured from the CLOSE
# of the last closed H1 bar, in minutes).
H1_FRESHNESS_GATE       = os.environ.get("H1_FRESHNESS_GATE", "false").strip().lower() == "true"
H1_MAX_AGE_MINUTES      = int(os.environ.get("H1_MAX_AGE_MINUTES", "15"))

# ── Entry parity config ───────────────────────────────────────────────────────
# Backtest always fills at M3 trigger close with no drift or age check.
# These guards are DISABLED for backtest parity.
MAX_ENTRY_DRIFT_PIPS        = float(os.environ.get("MAX_ENTRY_DRIFT_PIPS", "999.0"))   # effectively disabled
MIN_RR_AFTER_FILL           = float(os.environ.get("MIN_RR_AFTER_FILL", "2.5"))
MAX_M3_TRIGGER_AGE_MINUTES  = float(os.environ.get("MAX_M3_TRIGGER_AGE_MINUTES", "999.0"))  # effectively disabled

# ── In-memory candle cache ────────────────────────────────────────────────────
# Structure: { "EURUSD_M15": {"df": DataFrame, "fetched_at": datetime} }
_candle_cache: dict = {}

# Per-timeframe TTL — tightened so swing/OB data stays as fresh as what the
# backtest sees on each M5 candle iteration. Old values (M5=10min, M15=30min)
# meant the scanner could analyse 30-min-old swing highs/lows.
CACHE_TTL_SECONDS: dict = {
    "M3":  3  * 60,    # 3 min  — refresh every M3 candle
    "M5":  5  * 60,    # 5 min  — refresh every M5 candle
    "M15": 10 * 60,    # 10 min — swing detection needs recent closed candles
    "H1":  60 * 60,    # 1 hour
    "H4":  4  * 3600,  # 4 hours
    "D1":  8  * 3600,  # 8 hours
}
CACHE_TTL_SECONDS_DEFAULT = 10 * 60   # 10 min fallback for unknown TFs

# Fix 3: Fetch enough tail candles to cover the full TTL gap safely
# M3 TTL=5min -> 5/3 = ~2 candles needed, fetch 6 for safety
# M5 TTL=10min -> 10/5 = 2 candles, fetch 6
# M15 TTL=30min -> 30/15 = 2 candles, fetch 6
CACHE_TAIL_FETCH = 6

logger = logging.getLogger(__name__)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

HTF_MAP = {"M3": "M15", "M5": "H1", "M15": "H4"}

KILLZONES = {
    "Asian Open":    (0,  5),   # 00:00 - 05:00 UTC (05:30 - 10:30 IST) Tokyo Open Killzone
    "London Open":   (7,  11),  # 07:00 - 11:00 UTC (12:30 - 16:30 IST) London Open Killzone
    "New York Open": (12, 17),  # 12:00 - 17:00 UTC (17:30 - 22:30 IST) NY Open & Overlap Killzone
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
    """Session map — mirrors ALLOWED_SESSIONS in mtf_strategy.py exactly.
    Previously used different windows (London 7-12, NY 16-21) which caused
    session label mismatches between live signals and backtest trades.
    """
    from app.analysis.smc.mtf_strategy import ALLOWED_SESSIONS
    for name, (start_h, end_h) in ALLOWED_SESSIONS.items():
        if start_h <= hour < end_h:
            return name
    return "Off-Session"


def _get_client():
    """Return configured MetaApiClient or None."""
    from app.db.database import SessionLocal
    from app.db.models   import User
    from app.db          import crud
    from app.config      import settings as _cfg
    from app.mt5.metaapi_client import MetaApiClient

    user = None
    region = "backup-new-york"
    try:
        db = SessionLocal()
        user = db.query(User).filter(
            User.metaapi_account_id.isnot(None),
            User.metaapi_account_id != ""
        ).order_by(User.id.desc()).first() or db.query(User).first()
        if user:
            region = crud.get_setting(db, user.id, "metaapi_region") or \
                     _cfg.METAAPI_REGION or "backup-new-york"
        db.close()
    except Exception:
        pass

    token      = (user.metaapi_token      if user else None) or _cfg.METAAPI_TOKEN      or ""
    account_id = (user.metaapi_account_id if user else None) or _cfg.METAAPI_ACCOUNT_ID or ""
    if not token or not account_id:
        return None
    return MetaApiClient(token=token, account_id=account_id, region=region)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure time is tz-naive UTC and rows are sorted."""
    t = pd.to_datetime(df["time"], errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df = df.copy()
    df["time"] = t
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _load_live(symbol: str, tf: str, count: int) -> pd.DataFrame:
    """
    Load candles with in-memory cache.
    First call / cache expired (>4h): full fetch of `count` candles.
    Subsequent calls within TTL: fetch only last 5 candles and merge.
    """
    from app.analysis.indicators import add_indicators

    key   = f"{symbol}_{tf}"
    now   = datetime.now(timezone.utc)
    entry = _candle_cache.get(key)

    ttl = CACHE_TTL_SECONDS.get(tf, CACHE_TTL_SECONDS_DEFAULT)
    need_full = (
        entry is None or
        (now - entry["fetched_at"]).total_seconds() > ttl
    )

    client = _get_client()
    if client is None:
        return pd.DataFrame()

    # Minimum candle requirements per timeframe for reliable indicator calculation
    MIN_CANDLES: dict = {
        "M3": 50, "M5": 50, "M15": 50, "H1": 100, "H4": 50, "D1": 30,
    }

    try:
        if need_full:
            raw = client.get_candles(symbol, tf, count=count)
            min_required = MIN_CANDLES.get(tf, 30)
            if raw.empty or len(raw) < min_required:
                logger.warning("[Cache] Full fetch %s %s returned %d candles (min %d required) — pair skipped",
                               symbol, tf, len(raw) if not raw.empty else 0, min_required)
                return pd.DataFrame()
            df = _normalise(raw)
            _candle_cache[key] = {"df": df, "fetched_at": now}
            logger.debug("[Cache] Full fetch %s %s — %d candles", symbol, tf, len(df))
        else:
            raw_new = client.get_candles(symbol, tf, count=CACHE_TAIL_FETCH)
            cached  = entry["df"]
            if raw_new.empty:
                df = cached
            else:
                new_df   = _normalise(raw_new)
                combined = pd.concat([cached, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["time"], keep="last") \
                                   .sort_values("time") \
                                   .tail(count) \
                                   .reset_index(drop=True)
                _candle_cache[key] = {"df": combined, "fetched_at": now}
                df = combined
                logger.debug("[Cache] Incremental %s %s — +%d new, total %d",
                             symbol, tf, len(new_df), len(df))

        if len(df) < 30:
            return pd.DataFrame()

        # Gap detection: flag if any consecutive candle gap exceeds 2× the TF period.
        # Weekend gaps (Fri 22:00 UTC -> Sun 21:00 UTC ≈ 2820 min) are normal for forex
        # and must not be flagged as data errors.
        tf_min = _TF_MINUTES.get(tf, 5)
        times  = df["time"].values
        if len(times) > 1:
            diffs = pd.Series(pd.to_datetime(times)).diff().dropna()
            # Filter out weekend gaps: any gap where the earlier candle falls on Friday
            # (weekday=4) after 21:00 UTC or Saturday (weekday=5)
            ts_series = pd.Series(pd.to_datetime(times))
            non_weekend_gaps = []
            for i in range(1, len(ts_series)):
                gap_min = (ts_series.iloc[i] - ts_series.iloc[i - 1]).total_seconds() / 60
                prev_wd = ts_series.iloc[i - 1].weekday()   # 4=Fri, 5=Sat, 6=Sun
                prev_hr = ts_series.iloc[i - 1].hour
                is_weekend_gap = (
                    (prev_wd == 4 and prev_hr >= 21) or  # Friday after 21:00
                    prev_wd == 5 or                       # Saturday
                    (prev_wd == 6 and prev_hr < 21)       # Sunday before 21:00
                )
                if not is_weekend_gap:
                    non_weekend_gaps.append(gap_min)
            if non_weekend_gaps:
                max_gap_min = max(non_weekend_gaps)
                if max_gap_min > tf_min * 2:
                    logger.warning(
                        "[Cache] %s %s has gap of %.0f min (max allowed %d min) — data may be incomplete",
                        symbol, tf, max_gap_min, tf_min * 2,
                    )

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
    if score >= 50: return "MEDIUM"
    return "LOW"


def scan_pair(symbol: str, tf: str = "M15") -> Optional[dict]:
    """
    DISABLED — legacy 0-100 scoring engine that does NOT match the backtest.
    All live scanning routes through scan_mtf_smc_pair() which uses the
    identical 14-point A+ system as run_yesterday_backtest.py.
    Always returns None to prevent wrong-engine signals firing.
    """
    logger.warning(
        "[scan_pair] DISABLED — legacy engine called for %s. "
        "Use scan_mtf_smc_pair() for backtest-parity signals.", symbol
    )
    return None


def _scan_pair_legacy_disabled(symbol: str, tf: str = "M15") -> Optional[dict]:
    """Original legacy body — kept for reference only, never called."""
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

    # Ensure we evaluate the last COMPLETED (closed) candle to match backtest engine
    now_utc = datetime.now(timezone.utc)
    tf_minutes = {"M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get(tf, 15)

    last_time = df.iloc[-1]["time"]
    if getattr(last_time, "tzinfo", None) is None:
        last_time_utc = last_time.replace(tzinfo=timezone.utc)
    else:
        last_time_utc = last_time.tz_convert(timezone.utc)

    # If the last candle in df is still forming (its close time is in the future), drop it
    if now_utc < last_time_utc + timedelta(minutes=tf_minutes) and len(df) > 1:
        df = df.iloc[:-1].copy().reset_index(drop=True)

    # Use last completed candle as "now"
    candle = df.iloc[-1]
    c_time = candle["time"]
    if getattr(c_time, "tzinfo", None) is not None:
        c_time = c_time.tz_convert("UTC").tz_localize(None)

    hour = c_time.hour
    in_kz, kz_name = _in_killzone(hour)
    sess = _session(hour)

    if sess == "Off-Session":
        return None  # Off-Session = low quality, skip

    # Outside killzone but still in London/NY/Asian — score only, no hard block

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
    if not df_htf.empty and len(df_htf) > 1:
        htf_minutes = {"M15": 15, "H1": 60, "H4": 240, "D1": 1440}.get(htf_tf, 60)
        htf_last_time = df_htf.iloc[-1]["time"]
        if getattr(htf_last_time, "tzinfo", None) is None:
            htf_last_time_utc = htf_last_time.replace(tzinfo=timezone.utc)
        else:
            htf_last_time_utc = htf_last_time.tz_convert(timezone.utc)
        if now_utc < htf_last_time_utc + timedelta(minutes=htf_minutes):
            df_htf = df_htf.iloc[:-1].copy().reset_index(drop=True)

    htf_bias = detect_structure(df_htf.tail(80)) if not df_htf.empty and len(df_htf) >= 20 else "RANGING"

    htf_aligned  = htf_bias == direction
    htf_opposing = htf_bias not in ("RANGING", direction)

    # Extra confirmations
    ind    = detect_inducement(sl20, direction)
    fvg    = detect_fvg(sl20)
    fvg_ok = fvg["detected"] and fvg["direction"] == direction

    score = _score(pattern, htf_aligned, htf_opposing, in_kz, ind["detected"], fvg_ok)
    if score < 50:
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


# ── Fix 2: Strip forming (incomplete) candle before analysis ─────────────────
_TF_MINUTES: dict = {
    "M1": 1, "M3": 3, "M5": 5, "M15": 15,
    "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}

def _strip_forming(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Drop the last candle if it is still forming (its close time is in the future).
    This ensures swing detection, OB detection, and RR calculation only use
    fully closed candles — matching backtest behaviour exactly.
    """
    if df.empty or len(df) < 2:
        return df
    tf_minutes = _TF_MINUTES.get(tf, 5)
    now_utc    = datetime.now(timezone.utc)
    last_time  = df.iloc[-1]["time"]
    if getattr(last_time, "tzinfo", None) is None:
        last_time = last_time.replace(tzinfo=timezone.utc)
    candle_close = last_time + timedelta(minutes=tf_minutes)
    if now_utc < candle_close:
        logger.debug("[StripForming] %s forming candle dropped (closes at %s)", tf, candle_close)
        return df.iloc[:-1].copy().reset_index(drop=True)
    return df


def scan_mtf_smc_pair(symbol: str, disp_threshold: float = 0.8, min_score: int = 11, min_rr: float = 3.0) -> Optional[dict]:
    """
    Variant C live scan — identical signal-generation logic as the backtest engine.
    M15 bias -> M5 confirmation -> M3 trigger (market entry on closed M3 candle).

    Issue 4: H1 bias freshness gate — only accept signals within 15 min of H1 candle close.
    If the M3 signal fires more than 15 min into the current H1 candle, the bias is too
    old to trust and the signal is skipped.

    M3 data is MANDATORY. If unavailable after one retry, the pair is skipped.
    No silent fallback to M5 candles.
    """
    # H1 bias (500 candles ~20 days) matches backtest — cleaner trend direction
    # M5 500 candles, M3 400 candles — enough swing history to match backtest depth
    df15 = _strip_forming(_load_live(symbol, "H1", count=500), "H1")
    df5  = _strip_forming(_load_live(symbol, "M5", count=500), "M5")

    if df15.empty or df5.empty:
        logger.debug("[VariantC] %s skipped: H1/M5 data unavailable", symbol)
        return None

    # H1 bias freshness gate — DISABLED by default for backtest parity.
    # `df15` has already been passed through _strip_forming(), so df15.iloc[-1]
    # is the OPEN time of the last CLOSED H1 bar. Age must therefore be measured
    # from that bar's CLOSE (open + 1h), not from its open. Measuring from the
    # open makes the minimum possible age 60 min, so a 15-min threshold could
    # never be satisfied — this silently blocked 100% of signals.
    if H1_FRESHNESS_GATE:
        now_utc = datetime.now(timezone.utc)
        last_h1_open = df15.iloc[-1]["time"]
        if getattr(last_h1_open, "tzinfo", None) is None:
            last_h1_open = last_h1_open.replace(tzinfo=timezone.utc)
        last_h1_close = last_h1_open + timedelta(hours=1)
        minutes_since_h1_close = (now_utc - last_h1_close).total_seconds() / 60
        if minutes_since_h1_close > H1_MAX_AGE_MINUTES:
            logger.debug(
                "[VariantC] %s skipped: H1 bias stale (%.1f min since close, max %d min)",
                symbol, minutes_since_h1_close, H1_MAX_AGE_MINUTES,
            )
            return None

    # M3 is mandatory — attempt once, retry once, then skip
    df3 = _strip_forming(_load_live(symbol, "M3", count=400), "M3")
    if df3.empty:
        logger.warning("[VariantC] %s M3 fetch attempt 1 empty — retrying", symbol)
        df3 = _strip_forming(_load_live(symbol, "M3", count=400), "M3")
    if df3.empty:
        logger.warning("[VariantC] %s skipped: M3 data unavailable after retry", symbol)
        return None

    setup = evaluate_mtf_smc_setup(
        df15=df15, df5=df5, df3=df3,
        symbol=symbol, disp_threshold=disp_threshold,
        min_score=min_score, min_rr=min_rr,
    )

    if not setup or not setup.get("valid"):
        reason = (setup or {}).get("reason", "unknown")
        raw    = (setup or {}).get("score", 0)
        norm   = round(raw / 14.0 * 100, 2)
        logger.debug(
            "[VariantC] %s no setup: reason=%s raw_score=%d/14 normalized=%.2f",
            symbol, reason, raw, norm,
        )
        return None

    raw_score        = setup["score"]
    normalized_score = round(raw_score / 14.0 * 100, 2)
    pip_mult         = 100.0 if "JPY" in symbol.upper() else 10000.0
    pip_size         = 0.01 if "JPY" in symbol.upper() else 0.0001

    from app.risk.position_sizer import get_min_sl_pips

    direction     = setup["direction"]
    sl_price      = float(setup["stop_loss"])
    tp_price      = float(setup["take_profit"])
    ref_entry     = float(setup["entry_price"])  # M3 trigger candle close = backtest fill price
    m3_trigger_ts = setup.get("entry_time")
    live_price    = float(df5.iloc[-1]["close"])

    # ── Guard 1: M3 trigger age ───────────────────────────────────────────────
    # Backtest fills at the M3 trigger close immediately on each M5 iteration.
    # Live must honour the same price context. If the trigger candle is older
    # than MAX_M3_TRIGGER_AGE_MINUTES the setup has expired — skip it.
    now_utc = datetime.now(timezone.utc)
    if m3_trigger_ts is not None:
        try:
            trigger_ts = pd.to_datetime(m3_trigger_ts)
            if getattr(trigger_ts, "tzinfo", None) is None:
                trigger_ts = trigger_ts.replace(tzinfo=timezone.utc)
            trigger_age_min = (now_utc - trigger_ts).total_seconds() / 60.0
            if trigger_age_min > MAX_M3_TRIGGER_AGE_MINUTES:
                logger.info(
                    "[VariantC] %s skipped: M3 trigger %.1f min old (max %.0f min) — stale setup",
                    symbol, trigger_age_min, MAX_M3_TRIGGER_AGE_MINUTES,
                )
                return None
        except Exception:
            pass

    # ── Guard 2: Entry drift — honour M3 trigger price, not live price ────────
    # Backtest entry = ref_entry (M3 trigger candle close).
    # If the market has drifted more than MAX_ENTRY_DRIFT_PIPS from ref_entry,
    # this is a structurally different trade — skip it entirely.
    # Do NOT rebase entry on live_price: that changes SL distance, RR, and lot
    # size, making the live trade different from what the backtest validated.
    drift      = abs(live_price - ref_entry)
    drift_pips = drift / pip_size

    if drift_pips > MAX_ENTRY_DRIFT_PIPS:
        logger.info(
            "[VariantC] %s skipped: entry drift %.1f pips > max %.0f pips "
            "(ref=%.5f live=%.5f) — not the same trade as backtest",
            symbol, drift_pips, MAX_ENTRY_DRIFT_PIPS, ref_entry, live_price,
        )
        return None

    # Entry is the M3 trigger close — identical to backtest fill price.
    entry_price = ref_entry

    # ── Guard 3: Setup invalidation — price already past SL or TP ────────────
    if direction == "BUY":
        if live_price <= sl_price:
            logger.info("[VariantC] %s skipped: BUY invalidated — price %.5f at/below SL %.5f",
                        symbol, live_price, sl_price)
            return None
        if live_price >= tp_price:
            logger.info("[VariantC] %s skipped: BUY target already reached — price %.5f >= TP %.5f",
                        symbol, live_price, tp_price)
            return None
        sl_dist = entry_price - sl_price
        tp_dist = tp_price - entry_price
    else:
        if live_price >= sl_price:
            logger.info("[VariantC] %s skipped: SELL invalidated — price %.5f at/above SL %.5f",
                        symbol, live_price, sl_price)
            return None
        if live_price <= tp_price:
            logger.info("[VariantC] %s skipped: SELL target already reached — price %.5f <= TP %.5f",
                        symbol, live_price, tp_price)
            return None
        sl_dist = sl_price - entry_price
        tp_dist = entry_price - tp_price

    # ── Guard 4: Degenerate stop — SL too tight for broker ────────────────────
    sl_pips_now    = sl_dist / pip_size
    min_sl_allowed = get_min_sl_pips(symbol)
    if sl_pips_now < min_sl_allowed:
        logger.info(
            "[VariantC] %s skipped: stop too tight (%.1f pips < %d pips broker min)",
            symbol, sl_pips_now, min_sl_allowed,
        )
        return None

    # ── Guard 5: RR check from M3 trigger entry (same basis as backtest) ──────
    effective_rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
    if effective_rr < MIN_RR_AFTER_FILL:
        logger.info(
            "[VariantC] %s skipped: R/R %.2f below min %.2f at M3 trigger entry",
            symbol, effective_rr, MIN_RR_AFTER_FILL,
        )
        return None

    return {
        # ── Core trade parameters ──
        "symbol":          symbol,
        "direction":       direction,
        # entry_price = M3 trigger candle close — identical to backtest fill price.
        # Drift from live_price is already validated above (Guard 2).
        "entry_price":     entry_price,
        "signal_entry":    entry_price,
        "entry_type":      "market",
        "stop_loss":       sl_price,
        "take_profit":     tp_price,
        "rr_ratio":        f"1:{effective_rr:.2f}",
        "rr_val":          round(effective_rr, 2),
        "sl_pips":         round(sl_dist * pip_mult, 1),
        "tp_pips":         round(tp_dist * pip_mult, 1),
        # ── Live entry diagnostics ──
        "live_price":      live_price,
        "entry_drift_pips": round(drift_pips, 1),
        "setup_rr":        round(float(setup["rr_val"]), 2),
        # ── Score (raw 0-14 and normalized 0-100) ──
        "score":           raw_score,              # raw 0-14 from calculate_a_plus_score()
        "normalized_score": normalized_score,      # (raw/14)*100 — display only
        # ── Context ──
        "session":         setup["session"],
        "h4_trend":        setup["m15_bias"],
        "h1_trend":        setup["m15_location"],
        "strategy":        "MTF SMC A+ Strategy",
        "pattern_name":    "A+ MTF SMC Setup",
        "timeframe":       "M15->M5->M3",
        "pattern":         "A_PLUS_SMC",
        "confidence":      "HIGH" if raw_score >= 12 else "MEDIUM",
        "htf_bias":        setup["m15_bias"],
        "htf_aligned":     True,
        "killzone":        setup["session"],
        "fvg":             setup["fvg_overlap"],
        # ── Duplicate-prevention identity ──
        "m3_trigger_ts":   m3_trigger_ts,          # closed M3 candle timestamp
        "signal_time":     str(m3_trigger_ts),
        # ── UI-compatible aliases ──
        "current_price":   live_price,
        "entry":           entry_price,
        "sl":              sl_price,
        "tp_1_3":          tp_price,
        "ob_high":         setup["ob_zone"],
        "entry_valid":     True,
        "entry_status":    f"Variant C market signal — {setup['setup_class']} raw={raw_score}/14 norm={normalized_score:.2f}",
        "lot_size":        None,
        "risk_usd":        1.0,
        "spread":          None,
        "confirmations": [
            f"M15 Bias: {setup['m15_bias']}",
            f"Location: {setup['m15_location']}",
            f"Sweep: {setup['liquidity_type']}",
            f"Break: {setup['choch_type']}",
            f"Trigger: {setup['m3_trigger']}",
            f"Score: {raw_score}/14 ({normalized_score:.2f}/100)",
        ],
    }


# ── Backtest-parity spread table (matches run_mtf_smc_backtest SPREAD_PIPS) ──
# Applied at order submission to shift effective_entry against the trader,
# exactly as the backtest does. Real broker spread is used only as a filter
# (50% of SL guard in auto_scanner). Using the fixed table here ensures the
# live entry price matches what the backtest modelled.
SPREAD_PIPS: dict = {
    "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.3, "USDCHF": 1.6,
    "AUDUSD": 1.4, "NZDUSD": 2.0, "USDCAD": 1.8,
    "EURJPY": 1.8, "GBPJPY": 2.5,
}


def _apply_spread(symbol: str, direction: str, entry: float) -> float:
    """Return spread-adjusted effective entry matching backtest SPREAD_PIPS table.

    BUY  fills at ask = mid + spread  (effective_entry > entry)
    SELL fills at bid = mid - spread  (effective_entry < entry)
    """
    pip_size     = 0.01 if "JPY" in symbol.upper() else 0.0001
    spread_price = SPREAD_PIPS.get(symbol.upper(), 1.5) * pip_size
    if direction == "BUY":
        return round(entry + spread_price, 5)
    return round(entry - spread_price, 5)


def scan_pair_all_m5_candles(
    symbol: str,
    last_scanned_m5_time,
    disp_threshold: float = 0.8,
    min_score: int = 11,
    min_rr: float = 3.0,
) -> list:
    """
    Backtest-parity scan: evaluate every closed M5 candle that closed after
    `last_scanned_m5_time`, in chronological order, exactly as the backtest
    iterates `for idx in test_indices`.

    Returns a list of signal dicts (may be empty or contain multiple signals
    if several M5 candles closed since the last scan cycle).

    The caller (auto_scanner) is responsible for:
      - enforcing last_exit_time sequencing across the returned signals
      - updating last_scanned_m5_time to the last M5 candle evaluated
    """
    df15 = _strip_forming(_load_live(symbol, "H1", count=500), "H1")
    df5_full = _strip_forming(_load_live(symbol, "M5", count=500), "M5")

    if df15.empty or df5_full.empty:
        logger.warning("[VariantC] %s scan_pair_all_m5_candles: H1/M5 data unavailable", symbol)
        return []

    df3 = _strip_forming(_load_live(symbol, "M3", count=400), "M3")
    if df3.empty:
        logger.warning("[VariantC] %s M3 fetch attempt 1 empty — retrying", symbol)
        df3 = _strip_forming(_load_live(symbol, "M3", count=400), "M3")
    if df3.empty:
        logger.warning("[VariantC] %s skipped: M3 data unavailable after retry", symbol)
        return []

    # Identify all M5 candles that closed after last_scanned_m5_time
    if last_scanned_m5_time is not None:
        last_ts = pd.to_datetime(last_scanned_m5_time)
        if getattr(last_ts, "tzinfo", None) is not None:
            last_ts = last_ts.tz_localize(None)
        new_candles = df5_full[df5_full["time"] > last_ts]
    else:
        # First run: evaluate only the most recent closed M5 candle
        new_candles = df5_full.tail(1)

    if new_candles.empty:
        return []

    signals = []
    for idx in new_candles.index:
        df5_slice = df5_full.iloc[: idx + 1]

        setup = evaluate_mtf_smc_setup(
            df15=df15, df5=df5_slice, df3=df3,
            symbol=symbol, disp_threshold=disp_threshold,
            min_score=min_score, min_rr=min_rr,
        )

        if not setup or not setup.get("valid"):
            continue

        raw_score        = setup["score"]
        normalized_score = round(raw_score / 14.0 * 100, 2)
        pip_mult         = 100.0 if "JPY" in symbol.upper() else 10000.0
        pip_size         = 0.01 if "JPY" in symbol.upper() else 0.0001

        from app.risk.position_sizer import get_min_sl_pips

        direction     = setup["direction"]
        sl_price      = float(setup["stop_loss"])
        tp_price      = float(setup["take_profit"])
        ref_entry     = float(setup["entry_price"])
        m3_trigger_ts = setup.get("entry_time")
        live_price    = float(df5_slice.iloc[-1]["close"])

        # Guard 1: M3 trigger age (90-min window matching backtest)
        now_utc = datetime.now(timezone.utc)
        if m3_trigger_ts is not None:
            try:
                trigger_ts = pd.to_datetime(m3_trigger_ts)
                if getattr(trigger_ts, "tzinfo", None) is None:
                    trigger_ts = trigger_ts.replace(tzinfo=timezone.utc)
                trigger_age_min = (now_utc - trigger_ts).total_seconds() / 60.0
                if trigger_age_min > MAX_M3_TRIGGER_AGE_MINUTES:
                    logger.info(
                        "[VariantC] %s M5[%d] skipped: M3 trigger %.1f min old (max %.0f min)",
                        symbol, idx, trigger_age_min, MAX_M3_TRIGGER_AGE_MINUTES,
                    )
                    continue
            except Exception:
                pass

        # Guard 2: Entry drift
        drift      = abs(live_price - ref_entry)
        drift_pips = drift / pip_size
        if drift_pips > MAX_ENTRY_DRIFT_PIPS:
            logger.info(
                "[VariantC] %s M5[%d] skipped: drift %.1f pips > max %.0f pips",
                symbol, idx, drift_pips, MAX_ENTRY_DRIFT_PIPS,
            )
            continue

        entry_price = ref_entry

        # Guard 3: Setup invalidation
        if direction == "BUY":
            if live_price <= sl_price or live_price >= tp_price:
                continue
            sl_dist = entry_price - sl_price
            tp_dist = tp_price - entry_price
        else:
            if live_price >= sl_price or live_price <= tp_price:
                continue
            sl_dist = sl_price - entry_price
            tp_dist = entry_price - tp_price

        # Guard 4: Degenerate stop
        sl_pips_now    = sl_dist / pip_size
        min_sl_allowed = get_min_sl_pips(symbol)
        if sl_pips_now < min_sl_allowed:
            continue

        # Guard 5: RR
        effective_rr = (tp_dist / sl_dist) if sl_dist > 0 else 0.0
        if effective_rr < MIN_RR_AFTER_FILL:
            continue

        signals.append({
            "symbol":           symbol,
            "direction":        direction,
            "entry_price":      entry_price,
            "signal_entry":     entry_price,
            "entry_type":       "market",
            "stop_loss":        sl_price,
            "take_profit":      tp_price,
            "rr_ratio":         "1:%.2f" % effective_rr,
            "rr_val":           round(effective_rr, 2),
            "sl_pips":          round(sl_dist * pip_mult, 1),
            "tp_pips":          round(tp_dist * pip_mult, 1),
            "live_price":       live_price,
            "entry_drift_pips": round(drift_pips, 1),
            "setup_rr":         round(float(setup["rr_val"]), 2),
            "score":            raw_score,
            "normalized_score": normalized_score,
            "session":          setup["session"],
            "h4_trend":         setup["m15_bias"],
            "h1_trend":         setup["m15_location"],
            "strategy":         "MTF SMC A+ Strategy",
            "pattern_name":     "A+ MTF SMC Setup",
            "timeframe":        "M15->M5->M3",
            "pattern":          "A_PLUS_SMC",
            "confidence":       "HIGH" if raw_score >= 12 else "MEDIUM",
            "htf_bias":         setup["m15_bias"],
            "htf_aligned":      True,
            "killzone":         setup["session"],
            "fvg":              setup["fvg_overlap"],
            "m3_trigger_ts":    m3_trigger_ts,
            "signal_time":      str(m3_trigger_ts),
            "m5_candle_time":   df5_full.iloc[idx]["time"],  # for last_exit_time sequencing
            "current_price":    live_price,
            "entry":            entry_price,
            "sl":               sl_price,
            "tp_1_3":           tp_price,
            "ob_high":          setup["ob_zone"],
            "entry_valid":      True,
            "entry_status":     "Variant C M5-replay signal — %s raw=%d/14" % (setup["setup_class"], raw_score),
            "lot_size":         None,
            "risk_usd":         1.0,
            "spread":           None,
            "confirmations": [
                "M15 Bias: %s" % setup["m15_bias"],
                "Location: %s" % setup["m15_location"],
                "Sweep: %s" % setup["liquidity_type"],
                "Break: %s" % setup["choch_type"],
                "Trigger: %s" % setup["m3_trigger"],
                "Score: %d/14 (%.2f/100)" % (raw_score, normalized_score),
            ],
        })
        logger.info("[VariantC] %s M5[%d] signal: %s score=%d/14",
                    symbol, idx, direction, raw_score)

    return signals


def scan_all_pairs(timeframes=("M15", "M5"), min_confidence="LOW", strategy: str = "smc_mtf") -> list:
    """
    Always routes to scan_mtf_smc_pair() — Variant C H1->M5->M3 strategy.
    scan_pair() legacy path removed to prevent wrong engine being called.
    """
    signals = []
    for pair in PAIRS:
        try:
            sig = scan_mtf_smc_pair(pair)
            if sig:
                signals.append(sig)
                logger.info("MTF A+ SIGNAL %s %s score=%d", pair, sig["direction"], sig["score"])
        except Exception as e:
            logger.error("scan_mtf_smc_pair %s: %s", pair, e)
    signals.sort(key=lambda x: x["score"], reverse=True)
    return signals


def get_active_signals(risk_usd: float = 1.0, rr_ratio: float = 3.0,
                       data_source: str = "auto", strategy: str = "smc_mtf") -> list:
    """Legacy-compatible wrapper used by signals route."""
    return scan_all_pairs(strategy=strategy)

