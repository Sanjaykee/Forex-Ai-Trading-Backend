"""
Real-Time Trade Monitor
Runs continuously in the background, monitoring all active trades against live market ticks.
Closes trades at full TP (WIN) or original SL (LOSS). Pure 1:RR — no breakeven ratchet, no partial closes.
"""

import asyncio
import logging
from datetime import datetime
from app.db.database import SessionLocal
from app.db import models
from app.mt5.source_router import get_fetcher
from app.notifications.telegram_bot import send_trade_closed_alert

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
        active_trades = db.query(models.Trade).filter(
            models.Trade.result.in_(["OPEN", "PENDING"])
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
                sl    = float(trade.stop_loss or 0.0)
                tp    = float(trade.take_profit or 0.0)

                # Correct pip size per instrument
                s = symbol.upper()
                if "XAU" in s or "GOLD" in s:
                    pip_size = 0.01
                elif "XAG" in s or "SILVER" in s:
                    pip_size = 0.001
                elif "OIL" in s or "WTI" in s or "BRENT" in s:
                    pip_size = 0.01
                elif "BTC" in s or "ETH" in s:
                    pip_size = 1.0
                elif "JPY" in s:
                    pip_size = 0.01
                else:
                    pip_size = 0.0001

                # If this is a pending limit order waiting to fill:
                if trade.result == "PENDING":
                    # Order only fills when price touches or crosses entry (pullback to zone)
                    filled = (ask <= entry) if direction == "BUY" else (bid >= entry)
                    if filled:
                        trade.result = "OPEN"
                        db.commit()
                        logger.info(f"[TradeMonitor] Pending limit order #{trade.id} ({symbol}) filled at entry {entry} -> Trade is now OPEN")
                    else:
                        # If price ran straight to TP without pulling back to entry:
                        ran_away = (bid >= tp) if direction == "BUY" else (ask <= tp)
                        if ran_away and tp > 0:
                            trade.result = "MISSED"
                            trade.close_time = datetime.utcnow()
                            db.commit()
                            logger.info(f"[TradeMonitor] Pending limit order #{trade.id} ({symbol}) MISSED: Price reached TP without pulling back to entry.")
                            if trade.mt5_ticket:
                                from app.mt5.metaapi_client import metaapi_client
                                metaapi_client.cancel_order(trade.mt5_ticket)
                    continue

                # Fetch user for Telegram alerts
                user = db.query(models.User).filter(models.User.id == trade.user_id).first()
                chat_id = user.telegram_chat_id if user else None

                # ==============================================================
                # 1. CHECK TAKE PROFIT TARGET (FULL WIN)
                # ==============================================================
                hit_tp = (current_price >= tp) if direction == "BUY" else (current_price <= tp)
                if tp > 0 and hit_tp:
                    actual_rr = trade.actual_rr or 3.0
                    profit_usd = round(float(trade.risk_usd or 1.0) * actual_rr, 2)
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
                    profit_usd = round(-abs(float(trade.risk_usd or 1.0)), 2)
                    result_status = "LOSS"
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

    # Execute the actual trade on MT5 using this user's own MetaApi credentials
    from app.mt5.order_manager import place_order
    result = place_order(
        symbol=trade.symbol,
        direction=trade.direction,
        lot=trade.lot_size,
        entry=trade.entry_price,
        sl=trade.stop_loss,
        tp=trade.take_profit,
        comment="ForexAI",
        metaapi_token=user.metaapi_token if user else None,
        metaapi_account_id=user.metaapi_account_id if user else None,
    )
    from app.notifications.telegram_bot import _get_token
    import requests

    if result.get("success"):
        trade.mt5_ticket = str(result.get("ticket", ""))
        order_type = str(result.get("order_type", ""))
        if "LIMIT" in order_type:
            trade.result = "PENDING"
        else:
            trade.result = "OPEN"
        db.commit()
        logger.info(f"[TradeMonitor] MT5 order placed: ticket={trade.mt5_ticket} for {trade.symbol} {trade.direction} (result={trade.result})")

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

📊 *Price monitor active — trade will close at full TP or SL.*"""
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

    else:
        error_msg = result.get("error", "Unknown error")
        logger.error(f"[TradeMonitor] MT5 order FAILED for trade #{trade.id}: {error_msg}")
        trade.result = "REJECTED"
        signal.status = "rejected"
        db.commit()

        if user and user.telegram_chat_id:
            msg = f"""
❌ *ORDER REJECTED — NOT PLACED*

*Pair:* {trade.symbol}
*Direction:* {trade.direction}
*Entry:* `{trade.entry_price}`

*Reason:* {error_msg}

⚠️ No trade was opened on MT5. Signal has been marked rejected."""
            token = _get_token()
            if token:
                try:
                    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
                        "chat_id": user.telegram_chat_id,
                        "text": msg.strip(),
                        "parse_mode": "Markdown"
                    }, timeout=5)
                except Exception as e:
                    logger.error(f"Failed to send order rejection alert: {e}")

        return {"success": False, "message": error_msg, "trade_id": trade.id}
