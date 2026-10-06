import logging
from fastapi import APIRouter, Request, Depends
from sqlalchemy.orm import Session
import requests
from app.db.database import get_db
from app.db import models, crud
from app.signals.trade_monitor import activate_trade_from_signal
from app.notifications.telegram_bot import (
    _get_token,
    answer_callback_query,
    edit_telegram_message,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telegram", tags=["telegram"])

@router.post("/webhook")
async def telegram_webhook(request: Request, db: Session = Depends(get_db)):
    """Receives callback queries from Telegram inline buttons (APPROVE / REJECT) and text commands."""
    try:
        data = await request.json()
    except Exception:
        return {"ok": False, "error": "Invalid JSON"}

    # ==============================================================
    # 1. HANDLE INLINE BUTTON CLICKS (CALLBACK QUERIES)
    # ==============================================================
    if "callback_query" in data:
        cb = data["callback_query"]
        cb_id = cb.get("id")
        cb_data = cb.get("data", "")
        chat_id = str(cb.get("message", {}).get("chat", {}).get("id") or cb.get("from", {}).get("id"))
        msg_id = cb.get("message", {}).get("message_id")
        orig_text = cb.get("message", {}).get("text", "")

        user = db.query(models.User).filter(models.User.telegram_chat_id == chat_id).first()
        if not user:
            # Fallback to single active user if configured
            user = db.query(models.User).first()

        if not user:
            answer_callback_query(cb_id, "User not found in database.")
            return {"ok": True}

        if cb_data.startswith("approve_"):
            try:
                signal_id = int(cb_data.split("_")[1])
                res = activate_trade_from_signal(db, user.id, signal_id)
                if res.get("success"):
                    answer_callback_query(cb_id, "✅ Trade Approved! Price monitor active.")
                    new_text = orig_text + "\n\n🟢 *STATUS: APPROVED & ACTIVE (Risk-Free Price Monitor Running)*"
                    edit_telegram_message(chat_id, msg_id, new_text)
                else:
                    answer_callback_query(cb_id, f"⚠️ {res.get('message', 'Failed to activate')}")
            except Exception as e:
                logger.error(f"[TelegramWebhook] Error approving signal: {e}")
                answer_callback_query(cb_id, f"Error: {str(e)}")

        elif cb_data.startswith("reject_"):
            try:
                signal_id = int(cb_data.split("_")[1])
                crud.update_signal_status(db, signal_id, "rejected")
                answer_callback_query(cb_id, "❌ Trade Rejected.")
                new_text = orig_text + "\n\n🔴 *STATUS: REJECTED BY TRADER*"
                edit_telegram_message(chat_id, msg_id, new_text)
            except Exception as e:
                logger.error(f"[TelegramWebhook] Error rejecting signal: {e}")
                answer_callback_query(cb_id, f"Error: {str(e)}")

        return {"ok": True}

    # ==============================================================
    # 2. HANDLE TEXT COMMANDS (/start, /trades, /status)
    # ==============================================================
    if "message" in data:
        msg = data["message"]
        chat_id = str(msg.get("chat", {}).get("id"))
        text = msg.get("text", "").strip()
        token = _get_token()

        if text.startswith("/start"):
            welcome = (
                f"👋 *Welcome to Forex AI Trading Bot!*\n\n"
                f"Your Telegram Chat ID is: `{chat_id}`\n\n"
                f"👉 Copy this Chat ID and paste it in your Web App **Settings** page.\n"
                f"Once saved, you'll receive real-time signals with **[APPROVE]** and **[REJECT]** buttons right here!"
            )
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
                "chat_id": chat_id,
                "text": welcome,
                "parse_mode": "Markdown"
            }, timeout=5)

        elif text.startswith("/status") or text.startswith("/trades"):
            user = db.query(models.User).filter(models.User.telegram_chat_id == chat_id).first()
            if not user:
                user = db.query(models.User).first()
            if user:
                active_trades = db.query(models.Trade).filter(
                    models.Trade.user_id == user.id,
                    models.Trade.result.in_(["OPEN", "TP1_HIT"])
                ).all()
                if active_trades:
                    lines = [f"• #{t.id} *{t.symbol}* {t.direction} (Status: {t.result})" for t in active_trades]
                    status_msg = f"📊 *Active Monitored Trades ({len(active_trades)}):*\n" + "\n".join(lines)
                else:
                    status_msg = "ℹ️ No trades currently open. Scan in your Web UI to generate new signals!"
            else:
                status_msg = "User not registered in database."

            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={
                "chat_id": chat_id,
                "text": status_msg,
                "parse_mode": "Markdown"
            }, timeout=5)

    return {"ok": True}


@router.get("/setup-webhook")
def setup_webhook():
    """Configures Telegram to send inline button events directly to our Render backend."""
    token = _get_token()
    if not token or token == "your_telegram_bot_token":
        return {"success": False, "error": "TELEGRAM_BOT_TOKEN not configured"}

    webhook_url = "https://forex-ai-trading-backend.onrender.com/telegram/webhook"
    telegram_api = f"https://api.telegram.org/bot{token}/setWebhook"
    try:
        resp = requests.post(telegram_api, json={"url": webhook_url}, timeout=8)
        data = resp.json()
        return {
            "success": data.get("ok", False),
            "webhook_url": webhook_url,
            "telegram_response": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
