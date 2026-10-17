"""
SMC Backtest Engine — Fresh Build
==================================
3 Signal Patterns (any ONE fires a signal):

  Pattern A — CHoCH + OB Retest
    1. CHoCH detected on entry TF (reversal break)
    2. Unmitigated OB exists from the CHoCH impulse move
    3. Price is touching / near the OB zone
    → Highest confidence reversal entry

  Pattern B — BOS + Liquidity Sweep + OB
    1. BOS confirms direction (trend continuation)
    2. Liquidity sweep happened near OB zone (stop hunt)
    3. Unmitigated OB exists in sweep area
    → Trend continuation after stop hunt

  Pattern C — Accumulation Breakout + OB Retest
    1. Tight consolidation range (accumulation)
    2. BOS breaks out of range with strong candle
    3. OB = last candle of accumulation range
    4. Price retests OB after breakout
    → Breakout retest entry

HTF Bias — SCORE ONLY (never blocks a signal):
  Aligned  → +20 score
  Opposing → -10 score
  Ranging  → 0

Score threshold: ≥ 40 fires signal (shown with confidence label)
"""
from typing import Optional
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

from app.analysis.indicators             import add_indicators
from app.analysis.market_structure       import detect_structure
from app.analysis.smc.bos_choch         import detect_bos_choch
from app.analysis.smc.order_block       import detect_order_block, detect_accumulation_ob
from app.analysis.smc.liquidity_sweep   import detect_liquidity_sweep, detect_inducement
from app.analysis.smc.fvg               import detect_fvg

# ── Constants ────────────────────────────────────────────────────────────────

KILLZONES = {
    "London Open":   (7,  11),
    "New York Open": (12, 17),
}

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
         "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

# HTF for each entry TF
HTF_MAP = {"M3": "M15", "M15": "H4"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pip_cfg(symbol: str):
    """Returns (pip_unit, pip_multiplier, min_sl_pips, max_sl_pips)."""
    s = symbol.upper()
    if "JPY" in s:          return 0.01,   100.0, 4,  25
    if "XAU" in s:          return 0.1,     10.0, 5,  40
    if "BTC" in s:          return 1.0,      1.0, 50, 300
    return 0.0001, 10000.0, 3, 20


def _session(hour: int) -> str:
    if 7  <= hour <= 11: return "London"
    if 12 <= hour <= 17: return "New York"
    if 0  <= hour <= 4:  return "Asian"
    return "Off-Session"


def _in_killzone(hour: int) -> tuple:
    for name, (s, e) in KILLZONES.items():
        if s <= hour < e:
            return True, name
    return False, None


def _candles_needed(tf: str, days: int) -> int:
    per_day = {"M1": 1440, "M3": 480, "M5": 288, "M15": 96,
               "M30": 48,  "H1": 24,  "H4": 6,   "D1": 1}
    return min(int(per_day.get(tf.upper(), 96) * days * 1.4), 10000)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "time" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    t = pd.to_datetime(df["time"], errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    df["time"] = t
    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def _load(symbol: str, tf: str, days: int) -> pd.DataFrame:
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
    needed     = _candles_needed(tf, days)

    if token and account_id:
        try:
            from app.mt5.metaapi_client import MetaApiClient
            client = MetaApiClient(token=token, account_id=account_id, region=region)
            df = client.get_candles(symbol, tf, count=needed)
            if not df.empty and len(df) >= 30:
                logger.info("Loaded %d candles %s %s", len(df), symbol, tf)
                return _normalize(df)
        except Exception as e:
            logger.error("MetaApi %s %s: %s", symbol, tf, e)

    return pd.DataFrame()


def _score_signal(pattern: str, htf_aligned: bool, htf_opposing: bool,
                  in_kz: bool, has_inducement: bool, has_fvg: bool) -> int:
    base = {"A_CHOCH": 45, "B_BOS_SWEEP": 38, "C_ACCUM": 36}.get(pattern, 35)
    score = base
    if htf_aligned:   score += 20
    if htf_opposing:  score -= 10
    if in_kz:         score += 15
    if has_inducement: score += 12
    if has_fvg:       score += 8
    return max(score, 0)


def _confidence(score: int) -> str:
    if score >= 75: return "HIGH"
    if score >= 55: return "MEDIUM"
    return "LOW"


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    symbol:         str   = "EURUSD",
    timeframe:      str   = "M15",
    days:           int   = 15,
    session_filter: Optional[str] = None,
    rr_ratio:       float = 3.0,
    min_win_prob:   int   = 40,
    ladder_mode:    bool  = False,
) -> dict:

    tf = timeframe.upper()
    if tf not in ("M3", "M15"):
        tf = "M15"

    # ── Load candles ──────────────────────────────────────────────────────────
    df_raw = _load(symbol, tf, days)
    if df_raw.empty or len(df_raw) < 50:
        return {
            "error": f"No data for {symbol} {tf}. Connect MetaApi in Settings.",
            "symbol": symbol, "timeframe": tf, "days": days,
            "total_signals": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "net_pips": 0.0,
        }

    latest = df_raw["time"].max()
    cutoff = latest - pd.Timedelta(days=days)
    df = df_raw[df_raw["time"] >= cutoff].copy().reset_index(drop=True)
    if len(df) < 50:
        df = df_raw.copy().reset_index(drop=True)
    df = add_indicators(df)

    # ── HTF bias ──────────────────────────────────────────────────────────────
    htf_tf  = HTF_MAP.get(tf, "H4")
    df_htf  = _load(symbol, htf_tf, days)
    if not df_htf.empty:
        df_htf = add_indicators(df_htf)

    def _htf_bias_at(ts):
        if df_htf.empty:
            return "RANGING"
        sl = df_htf[df_htf["time"] <= ts].tail(80)
        if len(sl) < 20:
            return "RANGING"
        return detect_structure(sl)

    pip_unit, pip_mult, min_sl, max_sl = _pip_cfg(symbol)
    total_len = len(df)
    trades    = []
    last_exit = -1
    fc = {"kz": 0, "no_pattern": 0, "no_ob": 0, "low_score": 0}

    step = 1

    for i in range(40, total_len - 65, step):
        if i <= last_exit:
            continue

        candle = df.iloc[i]
        c_time = candle["time"]
        if getattr(c_time, "tzinfo", None) is not None:
            c_time = c_time.tz_convert("UTC").tz_localize(None)

        hour    = c_time.hour
        sess    = _session(hour)
        in_kz, kz_name = _in_killzone(hour)

        # Session filter — block Off-Session, only allow London, NY, Asian
        if session_filter:
            if session_filter.upper() not in sess.upper():
                fc["kz"] += 1; continue
        else:
            if sess == "Off-Session":
                fc["kz"] += 1; continue

        # Candle slices
        sl60 = df.iloc[max(0, i - 59): i + 1]
        sl40 = df.iloc[max(0, i - 39): i + 1]
        sl20 = df.iloc[max(0, i - 19): i + 1]

        # ── Detect BOS / CHoCH ────────────────────────────────────────────────
        bos = detect_bos_choch(sl60, lookback=50)

        pattern   = None
        direction = None
        ob        = None

        # ── Pattern A: CHoCH + OB Retest ─────────────────────────────────────
        if bos["detected"] and bos["type"] == "CHoCH":
            direction = bos["direction"]
            ob = detect_order_block(sl40, direction, lookback=35)
            if ob["detected"]:
                pattern = "A_CHOCH"

        # ── Pattern B: BOS + Sweep + OB ───────────────────────────────────────
        if pattern is None and bos["detected"] and bos["type"] == "BOS":
            direction = bos["direction"]
            sweep = detect_liquidity_sweep(sl40, lookback=25)
            if sweep["detected"] and sweep["direction"] == direction:
                ob = detect_order_block(sl40, direction, lookback=35)
                if ob["detected"]:
                    pattern = "B_BOS_SWEEP"

        # ── Pattern C: Accumulation + OB Retest ───────────────────────────────
        if pattern is None and bos["detected"]:
            direction = bos["direction"]
            ob = detect_accumulation_ob(sl60, direction, lookback=55)
            if ob["detected"]:
                pattern = "C_ACCUM"

        if pattern is None:
            fc["no_pattern"] += 1; continue
        if ob is None or not ob["detected"]:
            fc["no_ob"] += 1; continue

        # ── Score ─────────────────────────────────────────────────────────────
        htf_bias    = _htf_bias_at(c_time)
        htf_aligned  = htf_bias == direction
        htf_opposing = htf_bias not in ("RANGING", direction)

        ind = detect_inducement(sl20, direction)
        fvg = detect_fvg(sl20)
        fvg_ok = fvg["detected"] and fvg["direction"] == direction

        score = _score_signal(pattern, htf_aligned, htf_opposing,
                              in_kz, ind["detected"], fvg_ok)

        if score < min_win_prob:
            fc["low_score"] += 1; continue

        conf = _confidence(score)

        # ── Entry, SL, TP ─────────────────────────────────────────────────────
        atr = float(candle["atr"]) if "atr" in candle.index and not pd.isna(candle["atr"]) \
              else float(np.mean((sl20["high"] - sl20["low"]).values))

        zh, zl = float(ob["zone_high"]), float(ob["zone_low"])
        entry  = round((zh + zl) / 2, 5)

        if direction == "BULLISH":
            sl_dist = abs(entry - zl) + pip_unit * 2
            sl_dist = min(max_sl * pip_unit, max(min_sl * pip_unit, sl_dist))
            sl_price = round(entry - sl_dist, 5)
            tp_price = round(entry + sl_dist * rr_ratio, 5)
            trade_dir = "BUY"
        else:
            sl_dist = abs(zh - entry) + pip_unit * 2
            sl_dist = min(max_sl * pip_unit, max(min_sl * pip_unit, sl_dist))
            sl_price = round(entry + sl_dist, 5)
            tp_price = round(entry - sl_dist * rr_ratio, 5)
            trade_dir = "SELL"

        # ── Forward simulation (next 65 candles) ─────────────────────────────
        future  = df.iloc[i + 1: i + 66]
        outcome = "OPEN"
        pips    = 0.0
        exit_i  = i + len(future)

        for fs, (_, fr) in enumerate(future.iterrows()):
            fh = float(fr["high"])
            fl = float(fr["low"])

            if trade_dir == "BUY":
                if fl <= sl_price:
                    outcome = "LOSS"
                    pips    = -round(sl_dist * pip_mult, 1)
                    exit_i  = i + 1 + fs; break
                if fh >= tp_price:
                    outcome = "WIN"
                    pips    = round(sl_dist * rr_ratio * pip_mult, 1)
                    exit_i  = i + 1 + fs; break
            else:
                if fh >= sl_price:
                    outcome = "LOSS"
                    pips    = -round(sl_dist * pip_mult, 1)
                    exit_i  = i + 1 + fs; break
                if fl <= tp_price:
                    outcome = "WIN"
                    pips    = round(sl_dist * rr_ratio * pip_mult, 1)
                    exit_i  = i + 1 + fs; break

        if outcome == "OPEN":
            continue  # skip unresolved trades

        last_exit = exit_i + 2
        trades.append({
            "time":        str(c_time),
            "direction":   trade_dir,
            "pattern":     pattern,
            "confidence":  conf,
            "score":       score,
            "htf_bias":    htf_bias,
            "entry":       entry,
            "sl":          sl_price,
            "tp":          tp_price,
            "outcome":     outcome,
            "pips":        pips,
            "session":     sess,
            "kz":          kz_name,
            "inducement":  ind["detected"],
            "fvg":         fvg_ok,
        })

    # ── Results ───────────────────────────────────────────────────────────────
    total = len(trades)
    if total == 0:
        dominant = max(fc, key=fc.get) if any(fc.values()) else "unknown"
        reason_map = {
            "kz":         "all candles outside London/NY killzone hours",
            "no_pattern": "no CHoCH, BOS+Sweep, or Accumulation pattern found",
            "no_ob":      "pattern found but no unmitigated Order Block nearby",
            "low_score":  "setups found but score below threshold",
        }
        return {
            "symbol": symbol, "timeframe": tf, "days": days,
            "total_signals": 0, "wins": 0, "losses": 0,
            "win_rate": 0.0, "net_pips": 0.0,
            "filter_counts": fc,
            "message": f"No signals. Main block: {reason_map.get(dominant, dominant)}. Filters={fc}",
        }

    wins     = [t for t in trades if t["outcome"] == "WIN"]
    losses   = [t for t in trades if t["outcome"] == "LOSS"]
    win_rate = round(len(wins) / total * 100, 1)
    net_pips = round(sum(t["pips"] for t in trades), 1)

    # Session breakdown
    sessions = {}
    for sn in ("London", "New York", "Asian", "Off-Session"):
        st = [t for t in trades if t["session"] == sn]
        if st:
            sw = len([t for t in st if t["outcome"] == "WIN"])
            sessions[sn] = {"total": len(st), "wins": sw,
                            "win_rate": round(sw / len(st) * 100, 1)}

    # Pattern breakdown
    patterns = {}
    for pt in ("A_CHOCH", "B_BOS_SWEEP", "C_ACCUM"):
        pt_trades = [t for t in trades if t["pattern"] == pt]
        if pt_trades:
            pw = len([t for t in pt_trades if t["outcome"] == "WIN"])
            patterns[pt] = {"total": len(pt_trades), "wins": pw,
                            "win_rate": round(pw / len(pt_trades) * 100, 1)}

    sorted_t = sorted(trades, key=lambda x: x["pips"], reverse=True)

    return {
        "symbol":        symbol,
        "timeframe":     tf,
        "days":          days,
        "rr_ratio":      rr_ratio,
        "start_date":    str(df["time"].iloc[0].date()),
        "end_date":      str(df["time"].iloc[-1].date()),
        "total_signals": total,
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      win_rate,
        "net_pips":      net_pips,
        "filter_counts": fc,
        "sessions":      sessions,
        "patterns":      patterns,
        "best_trade":    sorted_t[0]  if sorted_t else None,
        "worst_trade":   sorted_t[-1] if sorted_t else None,
        "all_trades":    trades,
        "recent_trades": trades[-5:],
    }
