"""
Auto-Scanner Background Task — Variant C (smc_mtf) only.

Timing: aligned to M3 candle closes (:00, :03, :06, ... :57 UTC) + M3_SCAN_BUFFER_S.
Strategy: M15 bias -> M5 confirmation -> M3 trigger -> market order execution.
Score: raw 0-14 from calculate_a_plus_score(); normalized = (raw/14)*100.
Threshold: raw >= 11 (normalized >= 78.57) for auto-execution.
Entry: market order (entry=None to place_order). M3 close stored as signal reference.
Duplicate guard: symbol + direction + M3 trigger timestamp (stable identity).
"""

import asyncio
import logging
import requests
from datetime import datetime, timezone, timedelta
from app.db.database import SessionLocal
from app.db import models, crud
from app.signals.scanner import scan_all_pairs
from app.risk.position_sizer import calculate_lot_size
from app.notifications.telegram_bot import send_signal_alert, _get_token
from app.mt5.metaapi_client import MetaApiClient
from app.mt5.order_manager import place_order

logger = logging.getLogger(__name__)

# ── Timing ────────────────────────────────────────────────────────────────────
M3_PERIOD_S      = 180   # M3 candle = 3 minutes
M3_SCAN_BUFFER_S = 5     # wait 5 s after candle close before scanning

# ── Score threshold (Variant C, backtested at min_score=11/14) ────────────────
VC_MIN_RAW_SCORE        = 11
VC_NORMALIZED_THRESHOLD = round(VC_MIN_RAW_SCORE / 14.0 * 100, 2)   # 78.57

# ── Duplicate-guard in-memory set (symbol|direction|m3_ts) ───────────────────
# Cleared on each startup; DB check is the authoritative guard.
_fired_this_session: set = set()


# ── Timing helpers ────────────────────────────────────────────────────────────

def _get_wait_seconds() -> int:
    """Seconds until the next M3 candle close (:00, :03, :06, ...) + buffer."""
    now = datetime.now(timezone.utc)
    seconds_past_hour = now.minute * 60 + now.second
    wait = M3_PERIOD_S - (seconds_past_hour % M3_PERIOD_S) + M3_SCAN_BUFFER_S
    return max(10, min(wait, M3_PERIOD_S + M3_SCAN_BUFFER_S))


def _get_session() -> str:
    """Return active session name based on UTC hour."""
    now     = datetime.now(timezone.utc)
    weekday = now.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    hour    = now.hour
    if (weekday == 4 and hour >= 22) or weekday == 5 or (weekday == 6 and hour < 21):
        return "Off-Session"
    if 7  <= hour < 12: return "London"
    if 12 <= hour < 16: return "London/NY"
    if 16 <= hour < 21: return "New York"
    return "Asian"


# ── Duplicate protection ──────────────────────────────────────────────────────

def _signal_identity(symbol: str, direction: str, m3_trigger_ts) -> str:
    """Stable string identity for a Variant C setup."""
    ts_str = str(m3_trigger_ts) if m3_trigger_ts else "unknown"
    return f"{symbol}|{direction}|{ts_str}"


def _is_duplicate(db, user_id: int, symbol: str, direction: str,
                  m3_trigger_ts=None) -> bool:
    """
    Return True (block) if:
    1. An OPEN or PENDING trade already exists for this symbol.
    2. The exact same M3 trigger timestamp was already processed this session.
    3. A signal with the same identity was recorded in the DB within the last
       6 minutes (covers scanner restarts within the same M3 candle window).
    """
    # 1. Active trade guard
    active = db.query(models.Trade).filter(
        models.Trade.user_id == user_id,
        models.Trade.symbol  == symbol,
        models.Trade.result.in_(["OPEN", "PENDING"]),
    ).first()
    if active:
        logger.info("[VariantC] %s %s skipped — active trade #%d (%s)",
                    symbol, direction, active.id, active.result)
        return True

    # 2. In-memory session guard (fastest path)
    identity = _signal_identity(symbol, direction, m3_trigger_ts)
    if identity in _fired_this_session:
        logger.info("[VariantC] %s %s skipped — duplicate setup (in-memory: %s)",
                    symbol, direction, identity)
        return True

    # 3. DB recency guard — same symbol+direction within last 6 min
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=6)
    # SQLAlchemy may store created_at as tz-aware or tz-naive depending on DB.
    # Use a tz-naive comparison to be safe.
    cutoff_naive = cutoff.replace(tzinfo=None)
    recent = db.query(models.Signal).filter(
        models.Signal.user_id   == user_id,
        models.Signal.symbol    == symbol,
        models.Signal.direction == direction,
        models.Signal.created_at >= cutoff_naive,
    ).first()
    if recent:
        logger.info("[VariantC] %s %s skipped — duplicate setup (DB signal #%d < 6 min ago)",
                    symbol, direction, recent.id)
        return True

    return False


# ── Broker-aware lot sizing ───────────────────────────────────────────────────

def _get_symbol_info(symbol: str, user) -> dict | None:
    """
    Fetch volume_min / volume_max / volume_step from broker via MetaApi.
    Returns None if unavailable — caller must decide whether to skip or use fallback.
    """
    try:
        client = MetaApiClient(
            token=user.metaapi_token,
            account_id=user.metaapi_account_id,
        )
        if not client.is_configured:
            return None
        specs = client.get_symbol_specs(symbol)
        if not specs:
            return None
        # MetaApi symbol specs include volumeMin / volumeMax / volumeStep
        vol_min  = specs.get("volumeMin",  specs.get("volume_min",  None))
        vol_max  = specs.get("volumeMax",  specs.get("volume_max",  None))
        vol_step = specs.get("volumeStep", specs.get("volume_step", None))
        if vol_min is None or vol_step is None:
            return None
        return {
            "volume_min":  float(vol_min),
            "volume_max":  float(vol_max) if vol_max else 100.0,
            "volume_step": float(vol_step),
        }
    except Exception as e:
        logger.warning("[VariantC] get_symbol_info %s failed: %s", symbol, e)
        return None


def _calculate_lot(risk_usd: float, entry_ref: float, sl: float,
                   symbol: str, user) -> float | None:
    """
    Calculate lot size using broker symbol specs when available.
    Returns None if specs unavailable (caller skips auto-execution).
    """
    sym_info = _get_symbol_info(symbol, user)
    if sym_info is None:
        logger.warning(
            "[VariantC] %s lot sizing skipped — broker symbol specs unavailable. "
            "Auto-execution requires verified broker constraints.", symbol
        )
        return None

    vol_min  = sym_info["volume_min"]
    vol_max  = sym_info["volume_max"]
    vol_step = sym_info["volume_step"]

    lot = calculate_lot_size(
        risk_usd=risk_usd,
        entry=entry_ref,
        stop_loss=sl,
        symbol_info={"volume_min": vol_min, "volume_step": vol_step},
        symbol=symbol,
    )

    # Clamp to broker limits
    lot = max(vol_min, min(vol_max, lot))
    # Round to nearest step
    import math
    steps = round(lot / vol_step)
    lot   = round(steps * vol_step, 8)
    lot   = max(vol_min, lot)

    if lot > vol_max:
        logger.warning("[VariantC] %s calculated lot %.4f exceeds broker max %.4f — clamping",
                       symbol, lot, vol_max)
        lot = vol_max

    return lot


# ── SL/TP pre-validation ──────────────────────────────────────────────────────

def _validate_sl_tp(symbol: str, direction: str, sl: float, tp: float,
                    user) -> tuple[bool, str]:
    """
    Fetch live price and broker min-stop-distance.
    Validate that SL and TP are on the correct side of the current market price
    and satisfy the broker's minimum stop distance.
    Returns (ok: bool, reason: str).
    Does NOT silently move SL/TP.
    """
    try:
        client = MetaApiClient(
            token=user.metaapi_token,
            account_id=user.metaapi_account_id,
        )
        if not client.is_configured:
            return False, "MetaApi not configured"

        price_data = client.get_live_price(symbol)
        if not price_data:
            return False, "live price unavailable"

        ask = float(price_data.get("ask", 0))
        bid = float(price_data.get("bid", 0))
        if ask <= 0 or bid <= 0:
            return False, f"invalid live price ask={ask} bid={bid}"

        specs       = client.get_symbol_specs(symbol)
        broker_min  = float(specs.get("min_stop_distance", 0)) if specs else 0
        pip_unit    = 0.01 if "JPY" in symbol.upper() else 0.0001
        fallback_min = pip_unit * 3   # 3-pip fallback if broker doesn't report

        min_dist = broker_min if broker_min > 0 else fallback_min
        exec_price = ask if direction == "BUY" else bid

        if direction == "BUY":
            if sl >= exec_price:
                return False, f"BUY SL {sl} >= exec_price {exec_price}"
            if tp <= exec_price:
                return False, f"BUY TP {tp} <= exec_price {exec_price}"
            if (exec_price - sl) < min_dist:
                return False, (f"BUY SL distance {exec_price - sl:.5f} < "
                               f"broker min {min_dist:.5f}")
            if (tp - exec_price) < min_dist:
                return False, (f"BUY TP distance {tp - exec_price:.5f} < "
                               f"broker min {min_dist:.5f}")
        else:  # SELL
            if sl <= exec_price:
                return False, f"SELL SL {sl} <= exec_price {exec_price}"
            if tp >= exec_price:
                return False, f"SELL TP {tp} >= exec_price {exec_price}"
            if (sl - exec_price) < min_dist:
                return False, (f"SELL SL distance {sl - exec_price:.5f} < "
                               f"broker min {min_dist:.5f}")
            if (exec_price - tp) < min_dist:
                return False, (f"SELL TP distance {exec_price - tp:.5f} < "
                               f"broker min {min_dist:.5f}")

        return True, "ok"

    except Exception as e:
        return False, f"validation error: {e}"


# ── Signal data builder ───────────────────────────────────────────────────────

def _build_signal_data(user, r: dict, entry_ref: float, sl: float, tp: float,
                       lot: float, risk_usd: float, rr_ratio: float,
                       status: str) -> dict:
    raw_score  = r.get("score", 0)
    norm_score = round(raw_score / 14.0 * 100, 2)
    return {
        "user_id":          user.id,
        "symbol":           r["symbol"],
        "direction":        r["direction"],
        "strategy":         "Variant C [M15->M5->M3]",
        "entry_price":      entry_ref,          # M3 close — reference price only
        "stop_loss":        sl,
        "take_profit":      tp,
        "sl_pips":          r.get("sl_pips"),
        "tp_pips":          r.get("tp_pips"),
        "rr_ratio":         f"1:{rr_ratio:.2f}",
        "lot_size":         lot,
        "risk_usd":         risk_usd,
        "ml_probability":   norm_score,         # normalized score stored here
        "raw_score":        raw_score,
        "h4_trend":         r.get("h4_trend"),
        "h1_trend":         r.get("h1_trend"),
        "session":          r.get("session"),
        "spread_at_signal": None,
        "status":           status,
    }


# ── Manual handler ────────────────────────────────────────────────────────────

def _handle_manual(db, user, r: dict, entry_ref: float, sl: float, tp: float,
                   lot: float, risk_usd: float, rr_ratio: float):
    saved = crud.create_signal(
        db, _build_signal_data(user, r, entry_ref, sl, tp, lot,
                               risk_usd, rr_ratio, "pending")
    )
    alert_payload = dict(r)
    alert_payload.update({
        "id": saved.id, "risk_usd": risk_usd,
        "rr_ratio": f"1:{rr_ratio:.2f}", "lot_size": lot, "take_profit": tp,
    })
    send_signal_alert(user.telegram_chat_id, alert_payload)
    logger.info("[VariantC] %s %s MANUAL signal saved id=%d lot=%.4f",
                r["symbol"], r["direction"], saved.id, lot)


# ── Auto-execute handler ──────────────────────────────────────────────────────

def _handle_auto_execute(db, user, r: dict, entry_ref: float, sl: float,
                         tp: float, lot: float, risk_usd: float,
                         rr_ratio: float):
    """
    Save signal, place MARKET order (entry=None), record actual fill price,
    send Telegram confirmation.
    Trade is only marked OPEN after broker confirms execution.
    """
    from datetime import datetime as _dt

    symbol    = r["symbol"]
    direction = r["direction"]

    saved = crud.create_signal(
        db, _build_signal_data(user, r, entry_ref, sl, tp, lot,
                               risk_usd, rr_ratio, "auto_approved")
    )

    # Create trade row in PENDING state — only promoted to OPEN on broker confirm
    trade = models.Trade(
        user_id      = user.id,
        signal_id    = saved.id,
        symbol       = symbol,
        direction    = direction,
        entry_price  = entry_ref,    # M3 reference price
        stop_loss    = sl,
        take_profit  = tp,
        lot_size     = lot,
        risk_usd     = risk_usd,
        open_time    = _dt.utcnow(),
        result       = "PENDING",    # stays PENDING until broker confirms
        strategy     = "Variant C",
        actual_rr    = float(rr_ratio),
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)

    # Place MARKET order — entry=None forces market execution in MetaApiClient
    logger.info("[VariantC] %s %s MARKET order submitting lot=%.4f SL=%s TP=%s",
                symbol, direction, lot, sl, tp)

    result = place_order(
        symbol             = symbol,
        direction          = direction,
        lot                = lot,
        entry              = None,       # None = market order, not limit
        sl                 = sl,
        tp                 = tp,
        comment            = "VariantC-M3",
        metaapi_token      = user.metaapi_token,
        metaapi_account_id = user.metaapi_account_id,
    )

    if result.get("success"):
        ticket      = str(result.get("ticket", ""))
        order_type  = str(result.get("order_type", ""))
        fill_data   = result.get("data", {}) or {}

        trade.mt5_ticket = ticket

        # Attempt to capture actual fill price from broker response
        actual_fill = (
            fill_data.get("openPrice")
            or fill_data.get("currentPrice")
            or fill_data.get("price")
        )
        if actual_fill:
            try:
                trade.actual_fill_price = float(actual_fill)
            except (TypeError, ValueError):
                pass

        # Only mark OPEN if broker confirmed a market fill (not a limit)
        if "LIMIT" in order_type.upper():
            # Should not happen for Variant C — log and reject
            trade.result  = "REJECTED"
            saved.status  = "rejected"
            db.commit()
            logger.error(
                "[VariantC] %s %s order returned LIMIT type unexpectedly — "
                "rejected. Variant C requires market execution.", symbol, direction
            )
            _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                           f"⚠️ Unexpected LIMIT order returned — rejected",
                           "❌ Trade NOT opened. Variant C requires market execution.")
            return

        trade.result = "OPEN"
        db.commit()

        fill_str = f"{trade.actual_fill_price:.5f}" if trade.actual_fill_price else "pending"
        logger.info("[VariantC] %s %s MARKET order confirmed ticket=%s actual_fill=%s",
                    symbol, direction, ticket, fill_str)
        _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                       f"✅ Market order executed ticket={ticket} fill={fill_str}",
                       "📊 Price monitor active — closes at full TP or SL.")

    else:
        err = result.get("error", "unknown")
        trade.result  = "REJECTED"
        saved.status  = "rejected"
        db.commit()
        logger.error("[VariantC] %s %s order FAILED: %s", symbol, direction, err)
        _send_telegram(user, r, entry_ref, sl, tp, lot, risk_usd, rr_ratio,
                       f"⚠️ MT5 order failed: {err}",
                       "❌ Trade was NOT placed on MT5 broker.")


# ── Telegram helper ───────────────────────────────────────────────────────────

def _send_telegram(user, r: dict, entry_ref: float, sl: float, tp: float,
                   lot: float, risk_usd: float, rr_ratio: float,
                   status_line: str, monitor_line: str):
    token = _get_token()
    if not token or not user.telegram_chat_id:
        return
    raw_score  = r.get("score", 0)
    norm_score = round(raw_score / 14.0 * 100, 2)
    emoji      = "📈" if r["direction"] == "BUY" else "📉"
    reward_usd = round(risk_usd * rr_ratio, 2)
    msg = (
        "⚡ *VARIANT C AUTO-EXECUTED*\n\n"
        f"*Pair:* {r['symbol']}\n"
        f"*Direction:* {emoji} *{r['direction']}*\n"
        f"*M3 Ref:* `{entry_ref}` | *SL:* `{sl}` | *TP:* `{tp}`\n"
        f"*Risk:* ${risk_usd:.2f} → *Reward:* ${reward_usd:.2f} (1:{rr_ratio:.2f} RR)\n"
        f"*Lot:* `{lot}` | *Score:* {raw_score}/14 ({norm_score:.2f}/100)\n"
        f"*Session:* {r.get('session', '?')}\n\n"
        f"{status_line}\n{monitor_line}"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": user.telegram_chat_id, "text": msg,
                  "parse_mode": "Markdown"},
            timeout=6,
        )
    except Exception as e:
        logger.error("[VariantC] Telegram send failed: %s", e)


# ── Main scan loop ────────────────────────────────────────────────────────────

async def auto_scan_loop():
    """
    Background task: Variant C auto-scanner aligned to M3 candle closes.
    Wakes every ~3 minutes at :00, :03, :06, ... UTC + M3_SCAN_BUFFER_S.
    """
    logger.info("[AutoScanner] Variant C background task started "
                "(M3 timing, threshold raw>=%d / normalized>=%.2f).",
                VC_MIN_RAW_SCORE, VC_NORMALIZED_THRESHOLD)
    await asyncio.sleep(30)   # startup delay

    while True:
        session = _get_session()
        wait_s  = _get_wait_seconds()

        if session == "Off-Session":
            logger.debug("[AutoScanner] Off-session — sleeping %ds.", wait_s)
            await asyncio.sleep(wait_s)
            continue

        logger.info("[AutoScanner] Scanning — Variant C — session=%s", session)
        db = SessionLocal()
        try:
            users = db.query(models.User).filter(
                models.User.telegram_chat_id.isnot(None),
                models.User.telegram_chat_id != "",
            ).all()

            if not users:
                logger.debug("[AutoScanner] No users with Telegram — skipping.")
                continue

            # Variant C only — strategy="smc_mtf"
            try:
                results = scan_all_pairs(strategy="smc_mtf")
            except Exception as e:
                logger.error("[AutoScanner] scan_all_pairs failed: %s", e)
                continue

            if not results:
                logger.info("[AutoScanner] No Variant C signals found this cycle.")
                continue

            logger.info("[AutoScanner] %d Variant C signal(s) — processing for %d user(s).",
                        len(results), len(users))

            for user in users:
                risk_usd       = float(crud.get_setting(db, user.id, "max_risk_usd")     or 1.0)
                rr_ratio       = float(crud.get_setting(db, user.id, "rr_ratio")         or 3.0)
                max_trades_day = int(float(crud.get_setting(db, user.id, "max_trades_day") or 999))
                max_losses_day = int(float(crud.get_setting(db, user.id, "max_losses_day") or 999))
                exec_mode      = crud.get_setting(db, user.id, "execution_mode") or "manual"

                # Daily guard
                if exec_mode == "auto":
                    today_start = datetime.utcnow().replace(
                        hour=0, minute=0, second=0, microsecond=0)
                    trades_today = db.query(models.Trade).filter(
                        models.Trade.user_id   == user.id,
                        models.Trade.open_time >= today_start,
                    ).count()
                    losses_today = db.query(models.Trade).filter(
                        models.Trade.user_id   == user.id,
                        models.Trade.open_time >= today_start,
                        models.Trade.result    == "LOSS",
                    ).count()
                    if trades_today >= max_trades_day:
                        logger.info("[AutoScanner] User %d hit max trades/day (%d).",
                                    user.id, max_trades_day)
                        continue
                    if losses_today >= max_losses_day:
                        logger.info("[AutoScanner] User %d hit max losses/day (%d).",
                                    user.id, max_losses_day)
                        continue

                for r in results:
                    symbol       = r["symbol"]
                    direction    = r["direction"]
                    raw_score    = r.get("score", 0)
                    norm_score   = round(raw_score / 14.0 * 100, 2)
                    m3_ts        = r.get("m3_trigger_ts")
                    entry_ref    = float(r.get("entry_price") or r.get("entry") or 0)
                    sl           = float(r.get("stop_loss")   or r.get("sl")    or 0)
                    tp           = float(r.get("take_profit") or r.get("tp_1_3") or 0)

                    logger.info("[VariantC] %s M3 closed — evaluating entry_ref=%.5f",
                                symbol, entry_ref)
                    logger.info("[VariantC] %s %s setup raw_score=%d/14 normalized=%.2f",
                                symbol, direction, raw_score, norm_score)

                    # Score gate — raw >= 11 (normalized >= 78.57)
                    if raw_score < VC_MIN_RAW_SCORE:
                        logger.info(
                            "[VariantC] %s rejected — score %.2f < %.2f "
                            "(raw %d/14 < %d/14)",
                            symbol, norm_score, VC_NORMALIZED_THRESHOLD,
                            raw_score, VC_MIN_RAW_SCORE,
                        )
                        continue

                    logger.info("[VariantC] %s %s RR=%s", symbol, direction,
                                r.get("rr_ratio", "?"))

                    if not entry_ref or not sl or not tp:
                        logger.warning("[VariantC] %s missing entry/SL/TP — skipping.", symbol)
                        continue

                    # Duplicate guard
                    if _is_duplicate(db, user.id, symbol, direction, m3_ts):
                        continue

                    try:
                        if exec_mode == "auto":
                            # Broker lot sizing (requires broker specs)
                            lot = _calculate_lot(risk_usd, entry_ref, sl, symbol, user)
                            if lot is None:
                                logger.warning(
                                    "[VariantC] %s auto-execution skipped — "
                                    "broker symbol specs unavailable.", symbol)
                                continue

                            # SL/TP validation against live price
                            sl_tp_ok, sl_tp_reason = _validate_sl_tp(
                                symbol, direction, sl, tp, user)
                            if not sl_tp_ok:
                                logger.warning(
                                    "[VariantC] %s rejected — invalid SL/TP: %s",
                                    symbol, sl_tp_reason)
                                continue

                            logger.info("[VariantC] %s %s signal accepted — "
                                        "executing MARKET order", symbol, direction)
                            _handle_auto_execute(db, user, r, entry_ref, sl, tp,
                                                 lot, risk_usd, rr_ratio)

                            # Mark identity as fired to prevent same-candle re-fire
                            _fired_this_session.add(
                                _signal_identity(symbol, direction, m3_ts))

                        else:
                            # Manual mode — lot sizing uses broker specs if available,
                            # falls back to safe default for display purposes only
                            sym_info = _get_symbol_info(symbol, user)
                            if sym_info is None:
                                sym_info = {"volume_min": 0.01, "volume_step": 0.01}
                                logger.debug("[VariantC] %s using fallback symbol_info "
                                             "for manual signal display.", symbol)
                            lot = calculate_lot_size(
                                risk_usd=risk_usd,
                                entry=entry_ref,
                                stop_loss=sl,
                                symbol_info=sym_info,
                                symbol=symbol,
                            )
                            _handle_manual(db, user, r, entry_ref, sl, tp,
                                           lot, risk_usd, rr_ratio)
                            _fired_this_session.add(
                                _signal_identity(symbol, direction, m3_ts))

                    except Exception as sig_err:
                        logger.error("[VariantC] Error processing %s: %s",
                                     symbol, sig_err, exc_info=True)

        except Exception as e:
            logger.error("[AutoScanner] Unexpected error: %s", e, exc_info=True)
        finally:
            db.close()
            next_wait = _get_wait_seconds()
            logger.info("[AutoScanner] Scan complete. Next M3 scan in %.1f min.",
                        next_wait / 60)
            await asyncio.sleep(next_wait)
