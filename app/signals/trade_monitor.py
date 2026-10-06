"""
Real-Time Trade & Profit Ladder Monitor
Runs continuously in the background, monitoring all active trades against live market ticks.
Automatically ratchets Stop Loss to Breakeven at TP1, closes trades at TP targets or SL,
updates the database performance records, and sends instant milestone alerts to Telegram.
"""

import asyncio
import logging
from datetime import datetime
from app.db.database import SessionLocal
from app.db import models
from app.mt5.source_router import get_fetcher
from app.notifications.telegram_bot import send_tp1_alert, send_trade_closed_alert

logger = logging.getLogger(__name__)

async def monitor_open_trades_loop():
    """Background task running every 8 seconds to evaluate open trades."""
    logger.info("[TradeMonitor] Background price monitor started.")
    while True:
        try:
            await check_open_trades()
        except Exception as e:
            logger.error(f"[TradeMonitor] Unexpected error in monitor loop: {e}", exc_info=True)
        await asyncio.sleep(8)

async def check_open_trades():
    db = SessionLocal()
    try:
        # Fetch trades with OPEN or TP1_HIT status
        active_trades = db.query(models.Trade).filter(
            models.Trade.result.in_(["OPEN", "TP1_HIT"])
        ).all()

        if not active_trades:
            return

        _, get_live_price, _, _ = get_fetcher("auto")

        for trade in active_trades:
            try:
                symbol = trade.symbol
                price_data = get_live_price(symbol)
                if not price_data:
                    continue

                bid = float(price_data.get("bid", 0.0))
                ask = float(price_data.get("ask", 0.0))
                if bid <= 0 or ask <= 0:
                    continue

                direction = (trade.direction or "BUY").upper()
                current_price = bid if direction == "BUY" else ask
                entry = float(trade.entry_price or current_price)
                sl = float(trade.stop_loss or 0.0)
                tp = float(trade.take_profit or 0.0)

                # Pip calculation
                pip_size = 0.01 if ("JPY" in symbol or entry > 50) else 0.0001
                sl_distance = abs(entry - sl) if sl > 0 else (atr_fallback := 25 * pip_size)

                # TP1 target = 1:1.0 RR or half distance to TP
                if direction == "BUY":
                    tp1_target = entry + sl_distance
                else:
                    tp1_target = entry - sl_distance

                # Fetch user for Telegram alerts
                user = db.query(models.User).filter(models.User.id == trade.user_id).first()
                chat_id = user.telegram_chat_id if user else None

                # ==============================================================
                # 1. CHECK TAKE PROFIT TARGET (FULL WIN)
                # ==============================================================
                hit_tp = (current_price >= tp) if direction == "BUY" else (current_price <= tp)
                if tp > 0 and hit_tp:
                    actual_rr = trade.actual_rr or 3.0
                    profit_usd = round(trade.risk_usd * actual_rr, 2)
                    pips = round(abs(current_price - entry) / pip_size, 1)

                    trade.close_price = current_price
                    trade.close_time = datetime.utcnow()
                    trade.profit_usd = profit_usd
                    trade.result = "WIN"
                    db.commit()

                    logger.info(f"[TradeMonitor] Trade #{trade.id} ({symbol}) HIT TP: +${profit_usd} USD (+{pips} pips)")
                    if chat_id:
                        send_trade_closed_alert(
                            chat_id=chat_id,
                            trade={
                                "symbol": symbol,
                                "direction": direction,
                                "entry": entry,
                                "close_price": current_price,
                                "profit_usd": profit_usd,
                                "pips": pips,
                                "result": "WIN",
                                "ticket": trade.mt5_ticket or str(trade.id),
                            }
                        )
                    continue

                # ==============================================================
                # 2. CHECK STOP LOSS
                # ==============================================================
                hit_sl = (current_price <= sl) if direction == "BUY" else (current_price >= sl)
                if sl > 0 and hit_sl:
                    is_breakeven = (trade.result == "TP1_HIT")
                    profit_usd = 0.0 if is_breakeven else round(-abs(trade.risk_usd), 2)
                    result_status = "BREAKEVEN" if is_breakeven else "LOSS"
                    pips = round(abs(current_price - entry) / pip_size, 1)

                    trade.close_price = current_price
                    trade.close_time = datetime.utcnow()
                    trade.profit_usd = profit_usd
                    trade.result = result_status
                    db.commit()

                    logger.info(f"[TradeMonitor] Trade #{trade.id} ({symbol}) closed at {result_status}: ${profit_usd}")
                    if chat_id:
                        send_trade_closed_alert(
                            chat_id=chat_id,
                            trade={
                                "symbol": symbol,
                                "direction": direction,
                                "entry": entry,
                                "close_price": current_price,
                                "profit_usd": profit_usd,
                                "pips": pips,
                                "result": result_status,
                                "ticket": trade.mt5_ticket or str(trade.id),
                            }
                        )
                    continue

                # ==============================================================
                # 3. CHECK MILESTONE 1 (TP1) -> RATCHET SL TO BREAKEVEN
                # ==============================================================
                if trade.result == "OPEN":
                    hit_tp1 = (current_price >= tp1_target) if direction == "BUY" else (current_price <= tp1_target)
                    if hit_tp1:
                        # Move Stop Loss to Entry price (Risk-free Breakeven)
                        trade.stop_loss = entry
                        trade.result = "TP1_HIT"
                        db.commit()

                        pips = round(abs(current_price - entry) / pip_size, 1)
                        logger.info(f"[TradeMonitor] Trade #{trade.id} ({symbol}) REACHED TP1 (+{pips} pips). SL moved to Breakeven.")
                        if chat_id:
                            send_tp1_alert(
                                chat_id=chat_id,
                                trade={
                                    "symbol": symbol,
                                    "direction": direction,
                                    "entry": entry,
                                    "current_price": current_price,
                                    "breakeven_sl": entry,
                                    "pips": pips,
                                    "ticket": trade.mt5_ticket or str(trade.id),
                                }
                            )

            except Exception as trade_err:
                logger.error(f"[TradeMonitor] Error processing trade #{trade.id}: {trade_err}")

    finally:
        db.close()


def activate_trade_from_signal(db, user_id: int, signal_id: int) -> dict:
    """Activates an approved signal into a live tracked trade."""
    signal = db.query(models.Signal).filter(models.Signal.id == signal_id).first()
    if not signal:
        return {"success": False, "message": "Signal not found"}

    existing_trade = db.query(models.Trade).filter(models.Trade.signal_id == signal_id).first()
    if existing_trade:
        return {"success": True, "message": "Trade is already active and monitored", "trade_id": existing_trade.id}

    rr_val = 3.0
    if signal.rr_ratio:
        try:
            rr_val = float(str(signal.rr_ratio).split(":")[-1])
        except Exception:
            rr_val = 3.0

    trade = models.Trade(
        user_id=user_id,
        signal_id=signal.id,
        symbol=signal.symbol,
        direction=signal.direction,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        take_profit=signal.take_profit,
        lot_size=signal.lot_size or 0.01,
        risk_usd=signal.risk_usd or 1.0,
        open_time=datetime.utcnow(),
        result="OPEN",
        strategy=signal.strategy,
        actual_rr=rr_val,
    )
    db.add(trade)
    signal.status = "approved"
    db.commit()
    db.refresh(trade)

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user and user.telegram_chat_id:
        direction_emoji = "📈" if trade.direction == "BUY" else "📉"
        msg = f"""
🚀 *TRADE ENTERED & ACTIVATED!*

*Pair:* {trade.symbol}
*Direction:* {direction_emoji} *{trade.direction}*
*Entry:* `{trade.entry_price}`
*Stop Loss:* `{trade.stop_loss}`
*Take Profit:* `{trade.take_profit}`
*Risk:* ${trade.risk_usd:.2f} | *Target RR:* 1:{rr_val}

🛡️ *Background Price Monitor Active:*
When price hits TP1 (+25 pips), Stop Loss will automatically ratchet to Breakeven (100% Risk-Free)!
"""
        from app.notifications.telegram_bot import _get_token
        import requests
        token = _get_token()
        if token:
            try:
                requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
                    "chat_id": user.telegram_chat_id,
                    "text": msg.strip(),
                    "parse_mode": "Markdown"
                }, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send trade activation alert: {e}")

    return {"success": True, "message": "Trade activated and monitored", "trade_id": trade.id}
