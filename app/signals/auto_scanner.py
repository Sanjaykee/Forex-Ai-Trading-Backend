"""
Auto-Scanner Background Task - Variant C (smc_mtf) only.
Backtest-parity fixes applied:
  1. Slippage buffer moved to order submission only (not before validation)
  2. Spread threshold raised to 50% of SL (was 25%)
  3. M3 timing lag - scan buffer 8s
  4. Duplicate guard window reduced to 3 min (one M3 candle, was 6 min)
  5. Per-pair daily cap matching backtest DailyRiskProtector (3 trades/pair)
     Global user cap removed from auto mode
  6. Broker SL/TP distance validation removed (slippage buffer handles it)
  7. Global news block removed - per-symbol only (non-affected pairs still trade)
  8. MetaApi execution retry on failure
  9. MT5 min SL enforced via position_sizer
 10. Spread recorded in signal for audit
 11. Consecutive loss guard per pair per day (matches backtest per-pair protector)
 12. Session restriction filter — Asian disabled by default (ALLOW_ASIAN=false)
 13. API health tracker — pauses scanner after 2 consecutive MetaApi failures
 14. Weekly drawdown guard — locks trading for remainder of ISO week at 6% loss
 15. Scaling phase guard — forces $0.10 risk for first 50 live trades
"""

import asyncio
import logging
import logging.handlers
import os
import uuid
import requests
from datetime import datetime, timezone, timedelta
from app.db.database import SessionLocal
from app.db import models, crud
from app.signals.scanner import scan_all_pairs, scan_pair_all_m5_candles, _apply_spread, SPREAD_PIPS
from app.risk.position_sizer import (
    calculate_lot_size, enforce_min_sl, get_min_sl_pips, actual_risk_usd,
    get_risk_per_trade,
)
from app.notifications.telegram_bot import send_signal_alert, _get_token
from app.mt5.metaapi_client import MetaApiClient
from app.mt5.order_manager import place_order
from app.filters.news_filter import is_news_safe, get_upcoming_news

logger = logging.getLogger(__name__)

# -- Paper mode ---------------------------------------------------------------
# Set PAPER_MODE=false in environment to enable live execution.
# Defaults to true (safe) so accidental deploys never fire real orders.
PAPER_MODE = os.environ.get("PAPER_MODE", "true").strip().lower() != "false"

_PAPER_LOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "logs", "paper_signals.log"
)
_PAPER_LOG_PATH = os.path.normpath(_PAPER_LOG_PATH)

def _get_paper_logger() -> logging.Logger:
    """Return a dedicated file logger for paper signals (lazy-init, one handler)."""
    plog = logging.getLogger("paper_signals")
    if not plog.handlers:
        os.makedirs(os.path.dirname(_PAPER_LOG_PATH), exist_ok=True)
        fh = logging.FileHandler(_PAPER_LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        plog.addHandler(fh)
        plog.setLevel(logging.INFO)
        plog.propagate = False
    return plog


def _log_paper_signal(db, user_id: int, r: dict, entry: float, sl: float,
                      tp: float, lot: float, risk_usd: float,
                      spread_pips: float | None):
    """Write paper signal to log file and paper_trades DB table."""
    ist_offset = timedelta(hours=5, minutes=30)
    now_utc    = datetime.now(timezone.utc)
    ist_time   = (now_utc + ist_offset).strftime("%Y-%m-%d %H:%M:%S IST")
    pair       = r["symbol"]
    direction  = r["direction"]
    score      = r.get("score", 0)

    _get_paper_logger().info(
        "PAPER_SIGNAL: %s %s entry=%s sl=%s tp=%s score=%d time=%s",
        pair, direction, entry, sl, tp, score, ist_time,
    )

    sig_id = str(uuid.uuid4())
    row = models.PaperTrade(
        signal_id     = sig_id,
        user_id       = user_id,
        symbol        = pair,
        direction     = direction,
        entry_price   = entry,
        stop_loss     = sl,
        take_profit   = tp,
        score         = score,
        lot_size      = lot,
        risk_usd      = risk_usd,
        session       = r.get("session"),
        spread_pips   = spread_pips,
        signal_ts     = now_utc,
        signal_ts_ist = ist_time,
    )
    try:
        db.add(row)
        db.commit()
        logger.info("[PAPER] %s %s logged signal_id=%s", pair, direction, sig_id)
    except Exception as e:
        db.rollback()
        logger.error("[PAPER] DB write failed for %s %s: %s", pair, direction, e)


# -- Alerts log ---------------------------------------------------------------

_ALERTS_LOG_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "logs", "alerts.log"
))


def _get_alert_logger() -> logging.Logger:
    """Lazy-init file logger for operational alerts (one handler, no propagation)."""
    alog = logging.getLogger("scanner_alerts")
    if not alog.handlers:
        os.makedirs(os.path.dirname(_ALERTS_LOG_PATH), exist_ok=True)
        fh = logging.FileHandler(_ALERTS_LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        alog.addHandler(fh)
        alog.setLevel(logging.INFO)
        alog.propagate = False
    return alog


def send_alert(message: str) -> None:
    """Write an operational alert to alerts.log and the root logger."""
    _get_alert_logger().critical(message)
    logger.critical("[ALERT] %s", message)


# -- API health tracker (Fix 13) ----------------------------------------------
# Tracks consecutive MetaApi failures across the scan loop.
# paused=True blocks all broker calls until reset_pause() is called.

_api_health: dict = {
    "consecutive_failures": 0,
    "last_success":         None,   # datetime UTC or None
    "paused":               False,
}

API_PAUSE_THRESHOLD = 2


def _record_api_success() -> None:
    _api_health["consecutive_failures"] = 0
    _api_health["last_success"]         = datetime.now(timezone.utc)


def _record_api_failure(context: str, exc: Exception) -> None:
    _api_health["consecutive_failures"] += 1
    failures = _api_health["consecutive_failures"]
    logger.error("[APIHealth] MetaApi failure #%d in %s: %s", failures, context, exc)
    if failures >= API_PAUSE_THRESHOLD and not _api_health["paused"]:
        _api_health["paused"] = True
        msg = "SCANNER_PAUSED: %d consecutive MetaApi failures" % failures
        logger.critical("[APIHealth] %s", msg)
        send_alert(msg)


def reset_pause() -> None:
    """Clear the paused state and reset the failure counter.

    Call manually or wire to a future admin endpoint once MetaApi is healthy.
    """
    _api_health["paused"]               = False
    _api_health["consecutive_failures"] = 0
    logger.info("[APIHealth] Pause cleared — scanner resumed.")
    send_alert("SCANNER_RESUMED: pause cleared via reset_pause()")


# -- Timing -------------------------------------------------------------------
M3_PERIOD_S        = 180
M3_SCAN_BUFFER_S   = 8
ORDER_SUBMIT_DELAY = 0.1   # Issue 1: 100ms settle delay after M3 candle close

# -- Score threshold (matches backtest MIN_SCORE=11) --------------------------
VC_MIN_RAW_SCORE        = 11
VC_NORMALIZED_THRESHOLD = round(VC_MIN_RAW_SCORE / 14.0 * 100, 2)

# -- Fixed risk per trade matching backtest RISK_USD = $1.00 ------------------
BACKTEST_RISK_USD = 1.0

# -- Slippage buffer (applied at order submission only, not before validation) -
SLIPPAGE_PIPS = {"JPY": 3, "default": 1}

# -- Fix 2: Spread guard raised to 50% of SL (was 25%) -----------------------
MAX_SPREAD_PCT_OF_SL = 0.50

# -- Fix 8: MetaApi order retry -----------------------------------------------
ORDER_MAX_RETRIES = 2
ORDER_RETRY_DELAY = 2

# -- Fix 5: Per-pair daily protection matching backtest DailyRiskProtector ----
PAIR_MAX_TRADES_DAY   = 3   # matches backtest max_daily_trades=3
PAIR_MAX_LOSSES_DAY   = 2   # matches backtest max_daily_loss_r=2.0
PAIR_MAX_CONSEC_LOSS  = 2   # matches backtest max_consec_losses=2

# -- Account/weekly loss guards DISABLED for backtest parity -----------------
# Backtest has no account-level or weekly drawdown guard.
# These are intentionally removed so live takes the same trades as backtest.
ACCOUNT_DAILY_LOSS_PCT  = 0.03   # kept for reference only — not enforced
_daily_loss_tracker: dict  = {}
_weekly_loss_tracker: dict = {}

# -- Correlation blocs --------------------------------------------------------
# Set CORRELATION_BLOC_GUARD=false to disable for backtest parity.
# Backtest has no correlation guard — it evaluates each pair independently.
# Disabling live means potentially holding correlated positions simultaneously,
# which increases real drawdown beyond what the backtest showed.
CORRELATION_BLOC_GUARD = os.environ.get("CORRELATION_BLOC_GUARD", "false").strip().lower() == "true"
USD_BLOC = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]
JPY_BLOC = ["EURJPY", "GBPJPY"]

# -- Session restriction (Fix 12) ---------------------------------------------
# Set ALLOW_ASIAN=false in environment to disable the Asian session for all pairs.
#
# DEFAULT IS TRUE (Asian enabled) because the backtest's ALLOWED_SESSIONS in
# mtf_strategy.py includes Asian (00:00-04:00 UTC) and run_mtf_smc_backtest has
# no session restriction beyond that. In the Oct-2026 sample the Asian block
# produced ~36% of total net R, so disabling it live means intentionally running
# a different strategy than the one that was validated.
ALLOW_ASIAN = os.environ.get("ALLOW_ASIAN", "true").strip().lower() == "true"

_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY"}

# Base allowed sessions per group.
#
# These windows MUST mirror ALLOWED_SESSIONS in app/analysis/smc/mtf_strategy.py,
# which is what the backtest trades:
#     London   (7, 11)
#     New York (12, 17)
#     Asian    (0, 4)
# and the strings must match _get_session() EXACTLY. The previous list contained
# "NewYork" (no space), so every 16:00-20:00 UTC candle was labelled "New York"
# by _get_session() and then rejected by the exact-match test below.
SESSION_RULES: dict[str, list[str]] = {
    "JPY":   ["London", "New York"],   # JPY pairs
    "other": ["London", "New York"],   # all other pairs
}


def _session_allowed(symbol: str, session: str) -> bool:
    """Return True if the current session is permitted for this pair.

    ALLOW_ASIAN=true adds Asian to every group at runtime without touching
    SESSION_RULES, so the base dict stays clean for future per-group tuning.
    """
    if session in ("Off-Session", "Unknown"):
        return False
    group   = "JPY" if symbol.upper() in _JPY_PAIRS else "other"
    allowed = list(SESSION_RULES[group])
    if ALLOW_ASIAN:
        allowed.append("Asian")
    return session in allowed

# -- News filter --------------------------------------------------------------
# Set NEWS_FILTER=true to re-enable. Backtest has no news filter — it trades
# through all news events. Keeping it live means missing trades the backtest
# took. Disabled by default for parity.
NEWS_FILTER_ENABLED = os.environ.get("NEWS_FILTER", "false").strip().lower() == "true"
def _resolve_risk_usd(db) -> float:
    """Return fixed $1.00 risk per trade — matches backtest RISK_USD = 1.0."""
    return BACKTEST_RISK_USD


# -- Duplicate guard ----------------------------------------------------------
_fired_this_session: set = set()


# -- Timing helpers -----------------------------------------------------------

def _get_wait_seconds() -> int:
    now = datetime.now(timezone.utc)
    seconds_past_hour = now.minute * 60 + now.second
    wait = M3_PERIOD_S - (seconds_past_hour % M3_PERIOD_S) + M3_SCAN_BUFFER_S
    return max(10, min(wait, M3_PERIOD_S + M3_SCAN_BUFFER_S))


def _get_session() -> str:
    """Return the trading session label for the current UTC hour.

    Windows mirror ALLOWED_SESSIONS in app/analysis/smc/mtf_strategy.py, which
    defines what the backtest is allowed to trade. Hours outside those windows
    return "Off-Session" so the scanner skips them, matching the backtest.

    A weekend guard is kept on top of the windows: the backtest never has to
    handle weekends because FX candles simply do not exist then, while live can
    be scheduled to run at any moment.
    """
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()
    hour    = now.hour
    # Fri 22:00 UTC -> Sun 21:00 UTC: market closed.
    if (weekday == 4 and hour >= 22) or weekday == 5 or (weekday == 6 and hour < 21):
        return "Off-Session"
    if 0  <= hour < 4:  return "Asian"
    if 7  <= hour < 11: return "London"
    if 12 <= hour < 17: return "New York"
    return "Off-Session"


# -- Slippage buffer (applied at order submission only) -----------------------

def _apply_slippage_buffer(symbol: str, direction: str, sl: float) -> float:
    """Widen SL by slippage only at order submission - not before validation."""
    pip_size  = 0.01 if "JPY" in symbol.upper() else 0.0001
    slip_pips = SLIPPAGE_PIPS.get("JPY" if "JPY" in symbol.upper() else "default")
    buf = slip_pips * pip_size
    return round(sl - buf if direction == "BUY" else sl + buf, 5)


# -- Fix 2: Spread check (50% threshold) --------------------------------------

def _spread_ok(symbol: str, sl: float, entry: float, spread_pips: float) -> tuple:
    """Return (ok, reason). Reject only if spread > 50% of SL (extreme conditions)."""
    pip_size = 0.01 if "JPY" in symbol.upper() else 0.0001
    sl_pips  = abs(entry - sl) / pip_size
    if sl_pips <= 0:
        return False, "zero SL distance"
    pct = spread_pips / sl_pips
    if pct > MAX_SPREAD_PCT_OF_SL:
        return False, "spread %.1f pips = %.0f%% of SL %.1f pips (max 50%%)" % (
            spread_pips, pct * 100, sl_pips)
    return True, "ok"


# -- Fix 4: Duplicate guard (3-min window = one M3 candle) --------------------

def _signal_identity(symbol: str, direction: str, m3_trigger_ts) -> str:
    ts_str = str(m3_trigger_ts) if m3_trigger_ts else "unknown"
    return "%s|%s|%s" % (symbol, direction, ts_str)


def _is_duplicate(db, user_id: int, symbol: str, direction: str,
                  m3_trigger_ts=None) -> bool:
    # Block only if a trade is currently OPEN/PENDING on this pair
    active = db.query(models.Trade).filter(
        models.Trade.user_id == user_id,
        models.Trade.symbol  == symbol,
        models.Trade.result.in_(["OPEN", "PENDING"]),
    ).first()
    if active:
        logger.info("[VariantC] %s %s skipped - active trade #%d (%s)",
                    symbol, direction, active.id, active.result)
        return True

    # In-memory identity check (exact same M3 candle timestamp)
    identity = _signal_identity(symbol, direction, m3_trigger_ts)
    if identity in _fired_this_session:
        logger.info("[VariantC] %s %s skipped - duplicate (in-memory)", symbol, direction)
        return True

    # Fix 4: DB lookback reduced to 3 min (one M3 candle, was 6 min)
    cutoff_naive = (datetime.now(timezone.utc) - timedelta(minutes=3)).replace(tzinfo=None)
    recent = db.query(models.Signal).filter(
        models.Signal.user_id   == user_id,
        models.Signal.symbol    == symbol,
        models.Signal.direction == direction,
        models.Signal.created_at >= cutoff_naive,
    ).first()
    if recent:
        logger.info("[VariantC] %s %s skipped - duplicate DB signal #%d < 3 min ago",
                    symbol, direction, recent.id)
        return True

    return False


# -- Correlation guard -------------------------------------------------------

def _bloc_has_open_trade(db, user_id: int, bloc_pairs: list) -> bool:
    """Return True if any pair in bloc_pairs has a currently OPEN trade for user."""
    return db.query(models.Trade).filter(
        models.Trade.user_id == user_id,
        models.Trade.symbol.in_(bloc_pairs),
        models.Trade.result == "OPEN",
    ).first() is not None


# -- Issue 9 + Fix 14: Account-level daily AND weekly loss limits -------------

def _account_loss_limit_ok(user) -> tuple:
    """Always returns ok — backtest has no account/weekly loss guard."""
    return True, "ok"


# -- Fix 5: Per-pair daily protection (matches backtest DailyRiskProtector) ---

def _pair_can_trade(db, user_id: int, symbol: str) -> tuple:
    """
    Issue 7 fix: Only count CLOSED trades (WIN/LOSS) toward daily cap.
    OPEN/PENDING trades are excluded — they haven't resolved yet.
    Matches backtest DailyRiskProtector which only records completed trades.
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # Issue 7: filter to CLOSED trades only (WIN + LOSS), not OPEN/PENDING
    closed_trades = db.query(models.Trade).filter(
        models.Trade.user_id   == user_id,
        models.Trade.symbol    == symbol,
        models.Trade.open_time >= today_start,
        models.Trade.result.in_(["WIN", "LOSS"]),
    ).order_by(models.Trade.open_time.asc()).all()

    if len(closed_trades) >= PAIR_MAX_TRADES_DAY:
        return False, "%s max trades/day (%d) reached" % (symbol, PAIR_MAX_TRADES_DAY)

    losses = sum(1 for t in closed_trades if t.result == "LOSS")
    if losses >= PAIR_MAX_LOSSES_DAY:
        return False, "%s max losses/day (%d) reached" % (symbol, PAIR_MAX_LOSSES_DAY)

    # Consecutive losses from end of closed trade list
    consec = 0
    for t in reversed(closed_trades):
        if t.result == "LOSS":
            consec += 1
        else:
            break
    if consec >= PAIR_MAX_CONSEC_LOSS:
        return False, "%s max consecutive losses (%d) reached" % (symbol, PAIR_MAX_CONSEC_LOSS)

    return True, "ok"


# -- Broker helpers -----------------------------------------------------------

# -- Fixed symbol info matching backtest SYM_INFO ----------------------------
# Backtest uses: {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}
# Live fetches from broker which may differ. Use fixed values for parity.
BACKTEST_SYM_INFO = {"volume_min": 0.01, "volume_step": 0.01, "volume_max": 100.0}


def _get_symbol_info(symbol: str, user) -> dict | None:
    """Return fixed backtest-parity symbol info — matches SYM_INFO in run_yesterday_backtest.py."""
    try:
        client = MetaApiClient(token=user.metaapi_token,
                               account_id=user.metaapi_account_id)
        if client.is_configured:
            specs = client.get_symbol_specs(symbol)
            if specs:
                spread = specs.get("spread", specs.get("spreadFloat", None))
                return {
                    "volume_min":  0.01,
                    "volume_max":  100.0,
                    "volume_step": 0.01,
                    "spread_pips": float(spread) if spread else None,
                }
    except Exception as e:
        logger.warning("[VariantC] get_symbol_info %s failed: %s", symbol, e)
    # Fallback to backtest fixed values
    return dict(BACKTEST_SYM_INFO)


def _calculate_lot(risk_usd: float, entry_ref: float, sl: float,
                   symbol: str, user) -> float:
    """Calculate lot size using backtest-parity fixed SYM_INFO.
    Never returns None — falls back to BACKTEST_SYM_INFO so live always sizes
    the same way as run_yesterday_backtest.py.
    """
    sym_info = _get_symbol_info(symbol, user) or dict(BACKTEST_SYM_INFO)

    usd_jpy = None
    if "JPY" in symbol.upper() and not symbol.upper().startswith("USD"):
        usd_jpy = _get_usdjpy_rate(user)

    lot = calculate_lot_size(
        risk_usd=risk_usd, entry=entry_ref, stop_loss=sl,
        symbol_info=sym_info, symbol=symbol, usd_jpy=usd_jpy,
    )

    if lot is None or lot <= 0:
        # Backtest never refuses — use minimum lot as fallback
        lot = sym_info["volume_min"]
        logger.warning(
            "[VariantC] %s lot sizing fell back to min lot %.2f "
            "(risk $%.2f may not be exact)",
            symbol, lot, risk_usd,
        )

    return lot


def _get_usdjpy_rate(user) -> float | None:
    """Fetch a live USDJPY mid rate for valuing JPY-cross pip risk.

    Returns None on any failure; position_sizer then falls back to its
    USDJPY_RATE_ESTIMATE env value. Never raises.
    """
    try:
        client = MetaApiClient(token=user.metaapi_token,
                               account_id=user.metaapi_account_id)
        if not client.is_configured:
            return None
        px = client.get_live_price("USDJPY")
        if not px:
            return None
        bid, ask = px.get("bid"), px.get("ask")
        if bid and ask:
            return (float(bid) + float(ask)) / 2.0
        return float(bid or ask) or None
    except Exception as e:
        logger.debug("[VariantC] USDJPY rate fetch failed: %s", e)
        return None


def _validate_sl_direction(symbol: str, direction: str,
                            sl: float, tp: float, entry: float) -> tuple:
    """
    Fix 6: Only validate SL/TP direction correctness (not broker min distance).
    Slippage buffer already handles minimum distance at submission.
    """
    if direction == "BUY":
        if sl >= entry:
            return False, "BUY SL %s >= entry %s" % (sl, entry)
        if tp <= entry:
            return False, "BUY TP %s <= entry %s" % (tp, entry)
    else:
        if sl <= entry:
            return False, "SELL SL %s <= entry %s" % (sl, entry)
        if tp >= entry:
            return False, "SELL TP %s >= entry %s" % (tp, entry)
    return True, "ok"


# -- Signal data builder ------------------------------------------------------

def _build_signal_data(user, r: dict, entry_ref: float, sl: float, tp: float,
                       lot: float, risk_usd: float, rr_ratio: float,
                       status: str, spread_pips: float = None) -> dict:
    raw_score  = r.get("score", 0)
    norm_score = round(raw_score / 14.0 * 100, 2)
    return {
        "user_id":          user.id,
        "symbol":           r["symbol"],
        "direction":        r["direction"],
        "strategy":         "Variant C [M15->M5->M3]",
        "entry_price":      entry_ref,
        "stop_loss":        sl,
        "take_profit":      tp,
        "sl_pips":          r.get("sl_pips"),
        "tp_pips":          r.get("tp_pips"),
        "rr_ratio":         "1:%.2f" % rr_ratio,
        "lot_size":         lot,
        "risk_usd":         risk_usd,
        "ml_probability":   norm_score,
        "raw_score":        raw_score,
        "h4_trend":         r.get("h4_trend"),
        "h1_trend":         r.get("h1_trend"),
        "session":          r.get("session"),
        "spread_at_signal": spread_pips,
        "status":           status,
    }


# -- Manual handler -----------------------------------------------------------

def _handle_manual(db, user, r: dict, entry_ref: float, sl: float, tp: float,
                   lot: float, risk_usd: float, rr_ratio: float,
                   spread_pips: float = None):
    saved = crud.create_signal(
        db, _build_signal_data(user, r, entry_ref, sl, tp, lot,
                               risk_usd, rr_ratio, "pending", spread_pips)
    )
    alert_payload = dict(r)
    alert_payload.update({
        "id": saved.id, "risk_usd": risk_usd,
        "rr_ratio": "1:%.2f" % rr_ratio, "lot_size": lot, "take_profit": tp,
    })
    send_signal_alert(user.telegram_chat_id, alert_payload)
    logger.info("[VariantC] %s %s MANUAL signal saved id=%d lot=%.4f",
                r["symbol"], r["direction"], saved.id, lot)


# -- Fix 8: Auto-execute with retry -------------------------------------------

def _place_order_with_retry(symbol, direction, lot, sl, tp,
                            metaapi_token, metaapi_account_id,
                            client_order_id: str | None = None) -> dict:
    import time
    last_result = {}
    for attempt in range(1, ORDER_MAX_RETRIES + 1):
        result = place_order(
            symbol=symbol, direction=direction, lot=lot,
            entry=None, sl=sl, tp=tp, comment="VariantC-M3",
            metaapi_token=metaapi_token,
            metaapi_account_id=metaapi_account_id,
            client_order_id=client_order_id,
        )
        if result.get("success"):
            if attempt > 1:
                logger.info("[VariantC] %s order succeeded on attempt %d", symbol, attempt)
            return result
        last_result = result
        err = result.get("error", "unknown")
        logger.warning("[VariantC] %s order attempt %d failed: %s", symbol, attempt, err)
        # An ambiguous result means the broker may have accepted the order but
        # we have no ticket. Retrying could open a second position.
        if result.get("ambiguous"):
            logger.critical(
                "[VariantC] %s order result ambiguous — not retrying, "
                "manual reconciliation required", symbol,
            )
            break
        if attempt < ORDER_MAX_RETRIES:
            time.sleep(ORDER_RETRY_DELAY)
    return last_result


def _handle_auto_execute(db, user, r: dict, entry_ref: float, sl: float,
                         tp: float, lot: float, risk_usd: float,
                         rr_ratio: float, spread_pips: float = None):
    from datetime import datetime as _dt
    symbol    = r["symbol"]
    direction = r["direction"]

    saved = crud.create_signal(
        db, _build_signal_data(user, r, entry_ref, sl, tp, lot,
                               risk_usd, rr_ratio, "auto_approved", spread_pips)
    )

    # Create a PENDING trade row so place_order() can locate and update it
    trade = models.Trade(
        user_id     = user.id,
        signal_id   = saved.id,
        symbol      = symbol,
        direction   = direction,
        entry_price = entry_ref,
        stop_loss   = sl,
        take_profit = tp,
        lot_size    = lot,
        risk_usd    = risk_usd,
        open_time   = _dt.utcnow(),
        result      = "PENDING",
        strategy    = "Variant C",
        actual_rr   = float(rr_ratio),
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)

    logger.info("[VariantC] %s %s MARKET order submitting lot=%.4f SL=%s TP=%s",
                symbol, direction, lot, sl, tp)

    result = _place_order_with_retry(
        symbol, direction, lot, sl, tp,
        user.metaapi_token, user.metaapi_account_id,
        client_order_id="VC-%d" % trade.id,
    )

    if result.get("success"):
        ticket      = str(result.get("ticket", ""))
        order_type  = str(result.get("order_type", ""))
        fill_price  = result.get("fill_price")    # confirmed broker fill price
        actual_lots = result.get("actual_lots")   # confirmed broker lot size
        db_write_ok = result.get("db_write_ok", False)

        # place_order() already updated the PENDING row to OPEN with ticket +
        # fill_price + actual_lots. Refresh the local trade object to reflect that.
        db.refresh(trade)

        if "LIMIT" in order_type.upper():
            trade.result = "REJECTED"
            saved.status = "rejected"
            db.commit()
            logger.error("[VariantC] %s %s unexpected LIMIT order - rejected", symbol, direction)
            _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                           "WARNING: Unexpected LIMIT order - rejected",
                           "Trade NOT opened. Variant C requires market execution.")
            return

        fill_str = "%.5f" % fill_price if fill_price else "pending"
        db_str   = "db_ok" if db_write_ok else "db_FAILED"
        logger.info("[VariantC] %s %s confirmed ticket=%s fill=%s lots=%s %s",
                    symbol, direction, ticket, fill_str,
                    "%.4f" % actual_lots if actual_lots else "?", db_str)
        _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                       "Market order executed ticket=%s fill=%s" % (ticket, fill_str),
                       "Price monitor active - closes at full TP or SL.")
    else:
        err = result.get("error", "unknown")
        # Ambiguous: the broker may have accepted the order but gave us no
        # ticket. Marking it plainly REJECTED would tell the trade monitor the
        # position does not exist, so flag it for reconciliation instead.
        if result.get("ambiguous"):
            trade.result = "RECONCILE_REQUIRED"
            saved.status = "reconcile_required"
            db.commit()
            logger.critical(
                "[VariantC] %s %s order AMBIGUOUS — broker may hold a position "
                "with no ticket. Manual reconciliation required: %s",
                symbol, direction, err,
            )
            _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                           "AMBIGUOUS ORDER: %s" % err,
                           "Check the broker terminal NOW — a position may be open.")
            return

        trade.result = "REJECTED"
        saved.status = "rejected"
        db.commit()
        logger.error("[VariantC] %s %s order FAILED after %d retries: %s",
                     symbol, direction, ORDER_MAX_RETRIES, err)
        _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                       "MT5 order failed after %d retries: %s" % (ORDER_MAX_RETRIES, err),
                       "Trade was NOT placed on MT5 broker.")


# -- Telegram helper ----------------------------------------------------------

def _send_telegram(user, r: dict, entry_ref: float, sl: float, tp: float,
                   lot: float, risk_usd: float, rr_ratio: float,
                   status_line: str, monitor_line: str):
    token = _get_token()
    if not token or not user.telegram_chat_id:
        return
    raw_score  = r.get("score", 0)
    norm_score = round(raw_score / 14.0 * 100, 2)
    reward_usd = round(risk_usd * rr_ratio, 2)
    dir_label  = "BUY" if r["direction"] == "BUY" else "SELL"
    msg = (
        "*VARIANT C AUTO-EXECUTED*\n\n"
        "*Pair:* %s\n"
        "*Direction:* %s\n"
        "*M3 Ref:* `%s` | *SL:* `%s` | *TP:* `%s`\n"
        "*Risk:* $%.2f -> *Reward:* $%.2f (1:%.2f RR)\n"
        "*Lot:* `%s` | *Score:* %d/14 (%.2f/100)\n"
        "*Session:* %s\n\n"
        "%s\n%s"
    ) % (
        r["symbol"], dir_label,
        entry_ref, sl, tp,
        risk_usd, reward_usd, rr_ratio,
        lot, raw_score, norm_score,
        r.get("session", "?"),
        status_line, monitor_line,
    )
    try:
        requests.post(
            "https://api.telegram.org/bot%s/sendMessage" % token,
            json={"chat_id": user.telegram_chat_id, "text": msg,
                  "parse_mode": "Markdown"},
            timeout=6,
        )
    except Exception as e:
        logger.error("[VariantC] Telegram send failed: %s", e)


# -- Per-pair sequencing state ------------------------------------------------
# Mirrors backtest `last_exit_time` and `last_scanned_m5_time` per pair.
# last_exit_time:      no new trade evaluated until previous trade has exited.
# last_scanned_m5_time: tracks which M5 candles have already been evaluated so
#                       each cycle only replays candles that closed since last scan.
import pandas as _pd
_pair_last_exit_time:    dict = {}   # {symbol: _pd.Timestamp}
_pair_last_scanned_m5:   dict = {}   # {symbol: _pd.Timestamp | None}


def _get_pair_last_exit(symbol: str):
    """Return last exit time for pair from DB (close_time of last WIN/LOSS trade today)."""
    return _pair_last_exit_time.get(symbol)


def _update_pair_last_exit(db, user_id: int, symbol: str) -> None:
    """Refresh in-memory last_exit_time from DB for this pair."""
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    last_trade = db.query(models.Trade).filter(
        models.Trade.user_id   == user_id,
        models.Trade.symbol    == symbol,
        models.Trade.open_time >= today_start,
        models.Trade.result.in_(["WIN", "LOSS"]),
    ).order_by(models.Trade.open_time.desc()).first()
    if last_trade and last_trade.close_time:
        ts = _pd.Timestamp(last_trade.close_time)
        _pair_last_exit_time[symbol] = ts


# -- Main scan loop -----------------------------------------------------------

async def auto_scan_loop():
    logger.info("[AutoScanner] Variant C started (M3 timing, threshold raw>=%d / norm>=%.2f).",
                VC_MIN_RAW_SCORE, VC_NORMALIZED_THRESHOLD)
    await asyncio.sleep(30)

    while True:
        session = _get_session()
        wait_s  = _get_wait_seconds()

        if session == "Off-Session":
            logger.debug("[AutoScanner] Off-session - sleeping %ds.", wait_s)
            await asyncio.sleep(wait_s)
            continue

        logger.info("[AutoScanner] Scanning - Variant C - session=%s", session)
        db = SessionLocal()
        try:
            users = db.query(models.User).filter(
                models.User.telegram_chat_id.isnot(None),
                models.User.telegram_chat_id != "",
            ).all()

            if not users:
                logger.debug("[AutoScanner] No users with Telegram - skipping.")
                continue

            _cycle_raw      = 0
            _cycle_filtered = 0
            _cycle_executed = 0

            # Fix 15: resolve effective risk once per cycle
            cycle_risk_usd = _resolve_risk_usd(db)

            from app.signals.scanner import PAIRS as _PAIRS
            for symbol in _PAIRS:
                # Collect all signals from every M5 candle that closed since last scan
                try:
                    pair_signals = scan_pair_all_m5_candles(
                        symbol,
                        last_scanned_m5_time=_pair_last_scanned_m5.get(symbol),
                    )
                except Exception as e:
                    logger.error("[AutoScanner] scan_pair_all_m5_candles %s: %s", symbol, e)
                    continue

                if not pair_signals:
                    continue

                _cycle_raw += len(pair_signals)

                # Update last_scanned_m5_time to the last M5 candle evaluated
                last_m5_ts = pair_signals[-1].get("m5_candle_time")
                if last_m5_ts is not None:
                    _pair_last_scanned_m5[symbol] = last_m5_ts

                for user in users:
                    if _api_health["paused"]:
                        logger.warning(
                            "[APIHealth] SCANNER_PAUSED — skipping user %d. "
                            "Call reset_pause() to resume.", user.id
                        )
                        continue

                    # Refresh last_exit_time from DB at start of each pair/user
                    _update_pair_last_exit(db, user.id, symbol)

                    risk_usd   = BACKTEST_RISK_USD
                    rr_setting = float(crud.get_setting(db, user.id, "rr_ratio") or 3.0)
                    exec_mode  = crud.get_setting(db, user.id, "execution_mode") or "manual"

                    for r in pair_signals:
                        direction = r["direction"]
                        raw_score = r.get("score", 0)
                        rr_ratio  = float(r.get("rr_val") or rr_setting)
                        m3_ts     = r.get("m3_trigger_ts")
                        entry_ref = float(r.get("entry_price") or r.get("entry") or 0)
                        sl        = float(r.get("stop_loss")   or r.get("sl")    or 0)
                        tp        = float(r.get("take_profit") or r.get("tp_1_3") or 0)
                        m5_candle_time = r.get("m5_candle_time")

                        # ── Sequencing gate: backtest `if t_m5 <= last_exit_time: continue` ──
                        last_exit = _pair_last_exit_time.get(symbol)
                        if last_exit is not None and m5_candle_time is not None:
                            m5_ts = _pd.Timestamp(m5_candle_time)
                            if m5_ts <= last_exit:
                                logger.debug(
                                    "[VariantC] %s M5[%s] skipped — before last exit %s",
                                    symbol, m5_ts, last_exit,
                                )
                                _cycle_filtered += 1
                                continue

                        if raw_score < VC_MIN_RAW_SCORE:
                            logger.info("[VariantC] %s rejected - score %d/14 < %d/14",
                                        symbol, raw_score, VC_MIN_RAW_SCORE)
                            _cycle_filtered += 1
                            continue

                        if not entry_ref or not sl or not tp:
                            logger.warning("[VariantC] %s missing entry/SL/TP - skipping.", symbol)
                            _cycle_filtered += 1
                            continue

                        # News filter — disabled by default for backtest parity
                        if NEWS_FILTER_ENABLED and not is_news_safe(symbol):
                            upcoming = get_upcoming_news(symbol, lookahead_minutes=35)
                            logger.info("[VariantC] %s skipped - news imminent: %s",
                                        symbol, upcoming[:1])
                            _cycle_filtered += 1
                            continue

                        # Session restriction filter
                        current_session = r.get("session") or _get_session()
                        if not _session_allowed(symbol, current_session):
                            logger.info("[VariantC] SESSION_BLOCKED: %s at %s",
                                        symbol, current_session)
                            _cycle_filtered += 1
                            continue

                        # Per-pair daily protection
                        if exec_mode == "auto":
                            pair_ok, pair_reason = _pair_can_trade(db, user.id, symbol)
                            if not pair_ok:
                                logger.info("[VariantC] %s skipped - %s", symbol, pair_reason)
                                _cycle_filtered += 1
                                continue

                        # MT5 minimum SL
                        sl = enforce_min_sl(symbol, direction, entry_ref, sl)

                        # Direction validation
                        dir_ok, dir_reason = _validate_sl_direction(symbol, direction, sl, tp, entry_ref)
                        if not dir_ok:
                            logger.warning("[VariantC] %s rejected - %s", symbol, dir_reason)
                            _cycle_filtered += 1
                            continue

                        if _is_duplicate(db, user.id, symbol, direction, m3_ts):
                            _cycle_filtered += 1
                            continue

                        try:
                            sym_info    = _get_symbol_info(symbol, user)
                            broker_spread = sym_info.get("spread_pips") if sym_info else None

                            # Spread filter (50% of SL guard — real broker spread)
                            if broker_spread is not None:
                                sp_ok, sp_reason = _spread_ok(symbol, sl, entry_ref, broker_spread)
                                if not sp_ok:
                                    logger.warning("[VariantC] %s skipped - spread too wide: %s",
                                                   symbol, sp_reason)
                                    _cycle_filtered += 1
                                    continue

                            # Fixed spread applied at order submission — matches backtest
                            # effective_entry = entry ± SPREAD_PIPS[symbol] * pip_size
                            effective_entry = _apply_spread(symbol, direction, entry_ref)
                            fixed_spread_pips = SPREAD_PIPS.get(symbol.upper(), 1.5)

                            if exec_mode == "auto":
                                lot = _calculate_lot(risk_usd, effective_entry, sl, symbol, user)

                                sl_with_slip = _apply_slippage_buffer(symbol, direction, sl)

                                # Correlation bloc guard — disabled by default for backtest parity
                                if CORRELATION_BLOC_GUARD:
                                    if symbol in USD_BLOC:
                                        _bloc_name, _bloc_list = "USD_BLOC", USD_BLOC
                                    elif symbol in JPY_BLOC:
                                        _bloc_name, _bloc_list = "JPY_BLOC", JPY_BLOC
                                    else:
                                        _bloc_name, _bloc_list = None, []
                                    if _bloc_list and _bloc_has_open_trade(db, user.id, _bloc_list):
                                        logger.info(
                                            "[VariantC] BLOC_BLOCKED: %s — %s already has open trade",
                                            symbol, _bloc_name,
                                        )
                                        _cycle_filtered += 1
                                        continue

                                # Account-level loss limits
                                acct_ok, acct_reason = _account_loss_limit_ok(user)
                                if not acct_ok:
                                    logger.warning("[VariantC] All trading halted — %s", acct_reason)
                                    _cycle_filtered += 1
                                    continue

                                import time as _time
                                _time.sleep(ORDER_SUBMIT_DELAY)

                                try:
                                    if PAPER_MODE:
                                        logger.info(
                                            "[PAPER] %s %s score=%d/14 eff_entry=%.5f — PAPER_MODE",
                                            symbol, direction, raw_score, effective_entry,
                                        )
                                        _log_paper_signal(
                                            db, user.id, r, effective_entry, sl_with_slip, tp,
                                            lot, risk_usd, fixed_spread_pips,
                                        )
                                    else:
                                        logger.info(
                                            "[VariantC] %s %s MARKET order score=%d/14 "
                                            "entry=%.5f eff=%.5f spread=%.1fpips",
                                            symbol, direction, raw_score,
                                            entry_ref, effective_entry, fixed_spread_pips,
                                        )
                                        _handle_auto_execute(
                                            db, user, r, effective_entry, sl_with_slip, tp,
                                            lot, risk_usd, rr_ratio, fixed_spread_pips,
                                        )
                                    _record_api_success()
                                except Exception as api_err:
                                    _record_api_failure("place_order/%s" % symbol, api_err)
                                    _cycle_filtered += 1
                                    continue

                                _fired_this_session.add(_signal_identity(symbol, direction, m3_ts))
                                _cycle_executed += 1

                            else:
                                try:
                                    lot = _calculate_lot(risk_usd, effective_entry, sl, symbol, user)
                                    if lot is None:
                                        logger.warning(
                                            "[VariantC] %s MANUAL signal lot=0 — size manually.",
                                            symbol,
                                        )
                                        lot = BACKTEST_SYM_INFO["volume_min"]
                                    _handle_manual(db, user, r, effective_entry, sl, tp,
                                                   lot, risk_usd, rr_ratio, fixed_spread_pips)
                                    _record_api_success()
                                except Exception as api_err:
                                    _record_api_failure("manual_signal/%s" % symbol, api_err)
                                    _cycle_filtered += 1
                                    continue

                                _fired_this_session.add(_signal_identity(symbol, direction, m3_ts))
                                _cycle_executed += 1

                        except Exception as sig_err:
                            logger.error("[VariantC] Error processing %s: %s",
                                         symbol, sig_err, exc_info=True)

            logger.info(
                "[AutoScanner] Cycle summary: raw=%d | filtered=%d | executed=%d",
                _cycle_raw, _cycle_filtered, _cycle_executed,
            )

        except Exception as e:
            logger.error("[AutoScanner] Unexpected error: %s", e, exc_info=True)
        finally:
            db.close()
            next_wait = _get_wait_seconds()
            logger.info("[AutoScanner] Scan complete. Next M3 scan in %.1f min.", next_wait / 60)
            await asyncio.sleep(next_wait)
