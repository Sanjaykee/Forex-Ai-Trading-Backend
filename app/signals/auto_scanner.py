"""
Auto-Scanner Background Task
Scans all pairs every 15 minutes synchronized with M15 candle closes (:00, :15, :30, :45).
Sends Telegram alerts for new signals and executes trades when criteria are met.
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

logger = logging.getLogger(__name__)

# Scan interval per session (seconds) — aligned with M5/M15 candle closes (every 5 min)
SCAN_INTERVAL_ACTIVE     = 300   # 5 min during London / NY
SCAN_INTERVAL_ASIAN      = 300   # 5 min during Asian
SCAN_INTERVAL_OFFSESSION = 300   # 5 min off-session

# Signal dedup window — don't re-alert same symbol+direction within this period
SIGNAL_COOLDOWN_MINUTES = 120


def _get_wait_seconds() -> int:
    """Calculate seconds until the next M5 candle close (:00, :05, :10, ... :55) + 5s buffer."""
    now = datetime.now(timezone.utc)
    seconds_past_hour = now.minute * 60 + now.second
    wait = 300 - (seconds_past_hour % 300) + 5
    return max(15, min(wait, 305))


def _get_session() -> str:
    """Return active session name (Asian, London, London/NY, New York, or Off-Session on weekends)."""
    now = datetime.now(timezone.utc)
    weekday = now.weekday()  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
    hour = now.hour

    # Weekend market close: Friday 22:00 UTC to Sunday 21:00 UTC
    if (weekday == 4 and hour >= 22) or (weekday == 5) or (weekday == 6 and hour < 21):
        return "Off-Session"

    if 7 <= hour < 12:
        return "London"
    elif 12 <= hour < 16:
        return "London/NY"
    elif 16 <= hour < 21:
        return "New York"
    else:
        return "Asian"


def _is_duplicate(db, user_id: int, symbol: str, direction: str) -> bool:
    """
    Return True if symbol already has an active OPEN or PENDING trade,
    or has an unacted pending manual signal awaiting user approval.
    Matches backtest logic: as soon as a trade exits/cancels, the pair is immediately unlocked!
    """
    # 1. Never open a second trade if there is ALREADY an active/pending trade for this symbol
    active_trade = db.query(models.Trade).filter(
        models.Trade.user_id == user_id,
        models.Trade.symbol  == symbol,
        models.Trade.result.in_(["OPEN", "PENDING"]),
    ).first()
    if active_trade:
        logger.info(f"[AutoScanner] Duplicate blocked: Trade #{active_trade.id} for {symbol} is already {active_trade.result}")
        return True

    # 2. For manual approval mode: don't spam if there is already a recent pending signal awaiting user decision
    now_tz = datetime.now().astimezone()
    cutoff_pending = now_tz - timedelta(minutes=15)
    pending_signal = db.query(models.Signal).filter(
        models.Signal.user_id   == user_id,
        models.Signal.symbol    == symbol,
        models.Signal.direction == direction,
        models.Signal.status    == "pending",
        models.Signal.created_at >= cutoff_pending,
    ).first()
    if pending_signal:
        logger.info(f"[AutoScanner] Duplicate blocked: Pending Signal #{pending_signal.id} for {symbol} {direction} awaiting user approval")
        return True

    # 3. Prevent rapid duplicate firing in the exact same candle (< 4 minutes)
    candle_cutoff = now_tz - timedelta(minutes=4)
    same_candle = db.query(models.Signal).filter(
        models.Signal.user_id   == user_id,
        models.Signal.symbol    == symbol,
        models.Signal.direction == direction,
        models.Signal.created_at >= candle_cutoff,
    ).first()
    if same_candle:
        logger.info(f"[AutoScanner] Duplicate blocked: Signal #{same_candle.id} for {symbol} {direction} fired in the same candle (<4m ago)")
        return True

    return False


async def auto_scan_loop():
    """Background task: scans all pairs and sends Telegram alerts automatically."""
    logger.info("[AutoScanner] Background auto-scan task started.")

    # Small startup delay so DB/connections are ready
    await asyncio.sleep(30)

    while True:
        session = _get_session()
        wait_s = _get_wait_seconds()

        if session == "Off-Session":
            logger.debug(f"[AutoScanner] Off-session — skipping scan, sleeping {wait_s}s.")
            await asyncio.sleep(wait_s)
            continue

        logger.info(f"[AutoScanner] Scanning — session={session}")
        db = SessionLocal()
        try:
            # Fetch all users who have Telegram configured
            users = db.query(models.User).filter(
                models.User.telegram_chat_id.isnot(None),
                models.User.telegram_chat_id != "",
            ).all()

            if not users:
                logger.debug("[AutoScanner] No users with Telegram configured — skipping scan.")
                continue

            # Dual Mode: scan both M15 and M5 for high-probability setups
            try:
                results = scan_all_pairs(timeframes=("M15", "M5"), min_confidence="LOW")
            except Exception as e:
                logger.error(f"[AutoScanner] scan_all_pairs failed: {e}")
                continue

            if not results:
                logger.info("[AutoScanner] No signals found this cycle.")
                continue

            logger.info(f"[AutoScanner] {len(results)} signal(s) found — processing for {len(users)} user(s).")

            for user in users:
                # Load this user's risk + execution settings
                risk_usd       = float(crud.get_setting(db, user.id, "max_risk_usd")     or 1.0)
                rr_ratio       = float(crud.get_setting(db, user.id, "rr_ratio")         or 3.0)
                max_trades_day = int(float(crud.get_setting(db, user.id, "max_trades_day") or 999))
                max_losses_day = int(float(crud.get_setting(db, user.id, "max_losses_day") or 999))
                exec_mode      = crud.get_setting(db, user.id, "execution_mode") or "manual"
                rr_key         = f"tp_1_{int(rr_ratio)}" if int(rr_ratio) in (2, 3, 5) else "tp_1_3"

                # Daily trade/loss guard for auto-execute
                if exec_mode == "auto":
                    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                    trades_today = db.query(models.Trade).filter(
                        models.Trade.user_id  == user.id,
                        models.Trade.open_time >= today_start,
                    ).count()
                    losses_today = db.query(models.Trade).filter(
                        models.Trade.user_id  == user.id,
                        models.Trade.open_time >= today_start,
                        models.Trade.result   == "LOSS",
                    ).count()
                    if trades_today >= max_trades_day:
                        logger.info(f"[AutoScanner] User {user.id} hit max trades/day ({max_trades_day}) — skipping auto-execute.")
                        continue
                    if losses_today >= max_losses_day:
                        logger.info(f"[AutoScanner] User {user.id} hit max losses/day ({max_losses_day}) — skipping auto-execute.")
                        continue

                for r in results:
                    symbol    = r["symbol"]
                    direction = r["direction"]
                    score     = r.get("score", 0)

                    # Auto-execute fires on score >= 50
                    if exec_mode == "auto" and score < 50:
                        logger.info(f"[AutoScanner] Skipping {symbol} (score={score}) — auto-execute requires score >= 50")
                        continue

                    # Skip if duplicate within cooldown window
                    if _is_duplicate(db, user.id, symbol, direction):
                        continue

                    try:
                        entry = float(r.get("entry_price") or r.get("entry") or 0)
                        sl    = float(r.get("stop_loss")   or r.get("sl")    or 0)
                        tp    = float(r.get(rr_key) or r.get("tp_1_3") or 0)

                        if not entry or not sl or not tp:
                            continue

                        lot = calculate_lot_size(
                            risk_usd=risk_usd,
                            entry=entry,
                            stop_loss=sl,
                            symbol_info={"volume_min": 0.01, "volume_step": 0.01},
                            symbol=symbol,
                        )

                        if exec_mode == "auto":
                            _handle_auto_execute(db, user, r, entry, sl, tp, lot, risk_usd, rr_ratio)
                        else:
                            _handle_manual(db, user, r, entry, sl, tp, lot, risk_usd, rr_ratio)

                    except Exception as sig_err:
                        logger.error(f"[AutoScanner] Error processing signal {symbol}: {sig_err}")

        except Exception as e:
            logger.error(f"[AutoScanner] Unexpected error in scan loop: {e}", exc_info=True)
        finally:
            db.close()
            next_wait = _get_wait_seconds()
            logger.info(f"[AutoScanner] Scan complete. Next Dual M5/M15 scan in {round(next_wait / 60, 1)} min (at candle close).")
            await asyncio.sleep(next_wait)


def _build_signal_data(user, r, entry, sl, tp, lot, risk_usd, rr_ratio, status):
    tf_tag = f" [{r.get('timeframe', 'M15')}]"
    return {
        "user_id":          user.id,
        "symbol":           r["symbol"],
        "direction":        r["direction"],
        "strategy":         f"{r.get('pattern_name', 'SMC')}{tf_tag}",
        "entry_price":      entry,
        "stop_loss":        sl,
        "take_profit":      tp,
        "sl_pips":          r.get("sl_pips"),
        "tp_pips":          None,
        "rr_ratio":         f"1:{int(rr_ratio)}",
        "lot_size":         lot,
        "risk_usd":         risk_usd,
        "h4_trend":         r.get("htf_bias"),
        "h1_trend":         None,
        "session":          r.get("session"),
        "spread_at_signal": None,
        "status":           status,
    }


def _handle_manual(db, user, r, entry, sl, tp, lot, risk_usd, rr_ratio):
    """Save signal as pending and send Telegram with Accept/Reject buttons."""
    saved = crud.create_signal(db, _build_signal_data(user, r, entry, sl, tp, lot, risk_usd, rr_ratio, "pending"))
    alert_payload = dict(r)
    alert_payload["id"]          = saved.id
    alert_payload["risk_usd"]    = risk_usd
    alert_payload["rr_ratio"]    = f"1:{int(rr_ratio)}"
    alert_payload["lot_size"]    = lot
    alert_payload["take_profit"] = tp
    send_signal_alert(user.telegram_chat_id, alert_payload)
    logger.info(f"[AutoScanner] MANUAL signal sent -> {r['symbol']} ({r.get('timeframe', 'M15')}) {r['direction']} | user={user.id} | lot={lot}")


def _handle_auto_execute(db, user, r, entry, sl, tp, lot, risk_usd, rr_ratio):
    """Save signal, execute order immediately on MT5, send Telegram confirmation."""
    from app.mt5.order_manager import place_order
    from datetime import datetime as _dt

    saved  = crud.create_signal(db, _build_signal_data(user, r, entry, sl, tp, lot, risk_usd, rr_ratio, "auto_approved"))
    rr_val = float(rr_ratio)

    trade = models.Trade(
        user_id     = user.id,
        signal_id   = saved.id,
        symbol      = r["symbol"],
        direction   = r["direction"],
        entry_price = entry,
        stop_loss   = sl,
        take_profit = tp,
        lot_size    = lot,
        risk_usd    = risk_usd,
        open_time   = _dt.utcnow(),
        result      = "OPEN",
        strategy    = r.get("pattern_name"),
        actual_rr   = rr_val,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)

    result = place_order(
        symbol             = trade.symbol,
        direction          = trade.direction,
        lot                = trade.lot_size,
        entry              = trade.entry_price,
        sl                 = trade.stop_loss,
        tp                 = trade.take_profit,
        comment            = f"ForexAI-{r.get('timeframe', 'M15')}",
        metaapi_token      = user.metaapi_token,
        metaapi_account_id = user.metaapi_account_id,
    )

    if result.get("success"):
        trade.mt5_ticket = str(result.get("ticket", ""))
        order_type = str(result.get("order_type", ""))
        if "LIMIT" in order_type:
            trade.result = "PENDING"
            status_line = f"⏳ Pending limit order placed on MT5 (ticket: {trade.mt5_ticket})"
            monitor_line = "📊 _Waiting for price pullback to entry zone._"
            logger.info(f"[AutoScanner] AUTO-EXECUTED PENDING LIMIT {r['symbol']} {r['direction']} ticket={trade.mt5_ticket}")
        else:
            trade.result = "OPEN"
            status_line = f"✅ Market order executed on MT5 (ticket: {trade.mt5_ticket})"
            monitor_line = "📊 _Price monitor active — closes at full TP or SL._"
            logger.info(f"[AutoScanner] AUTO-EXECUTED MARKET {r['symbol']} {r['direction']} ticket={trade.mt5_ticket}")
        db.commit()
    else:
        trade.result = "REJECTED"
        saved.status = "rejected"
        db.commit()
        status_line = "⚠️ MT5 order failed: " + str(result.get("error", "unknown"))
        monitor_line = "❌ _Trade was NOT placed on MT5 broker. No position opened._"
        logger.error(f"[AutoScanner] AUTO-EXECUTE FAILED {r['symbol']}: {result.get('error')}")

    token = _get_token()
    if token and user.telegram_chat_id:
        direction_emoji = "📈" if r["direction"] == "BUY" else "📉"
        reward_usd      = round(risk_usd * rr_val, 2)
        tf_label        = r.get("timeframe", "M15")
        msg = (
            "⚡ *AUTO-EXECUTED*\n\n"
            f"*Pair:* {r['symbol']} *({tf_label})*\n"
            f"*Direction:* {direction_emoji} *{r['direction']}*\n"
            f"*Entry:* `{entry}` | *SL:* `{sl}` | *TP:* `{tp}`\n"
            f"*Risk:* ${risk_usd:.2f} → *Reward:* ${reward_usd:.2f} (1:{int(rr_ratio)} RR)\n"
            f"*Lot:* `{lot}` | *Score:* {r.get('score', 0)}%\n"
            f"*Pattern:* {r.get('pattern_name', 'SMC')}\n\n"
            f"{status_line}\n"
            f"{monitor_line}"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": user.telegram_chat_id, "text": msg, "parse_mode": "Markdown"},
                timeout=6,
            )
        except Exception as e:
            logger.error(f"[AutoScanner] Telegram confirmation failed: {e}")
