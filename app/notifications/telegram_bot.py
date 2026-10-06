import os
from pathlib import Path
import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)

def _get_token() -> str:
    """Dynamically resolve Telegram Bot token from settings, env, or disk."""
    token = getattr(settings, "TELEGRAM_BOT_TOKEN", "")
    if token and token != "your_telegram_bot_token":
        return token
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if env_token and env_token != "your_telegram_bot_token":
        return env_token
    for p in [Path("backend/.env"), Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith("TELEGRAM_BOT_TOKEN="):
                            val = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                            if val and val != "your_telegram_bot_token":
                                return val
            except Exception:
                pass
    return "8885177423:AAFhpQFKhqCvc50OhtwvYywlu0vLs8BVcmk"

def send_signal_alert(chat_id: str, signal: dict) -> bool:
    token = _get_token()
    if not chat_id or not token or token == "your_telegram_bot_token":
        logger.debug("Telegram alert skipped: Token or chat_id not configured")
        return False

    direction_emoji = "📈" if signal.get("direction") == "BUY" else "📉"
    confirmations   = "\n".join([f"✅ {c}" for c in signal.get("confirmations", [])]) or "✅ SMC Precision Setup"

    # Format dynamic ladder if present
    ladder_text = ""
    if "dynamic_ladder" in signal:
        dl = signal["dynamic_ladder"]
        m1 = dl.get("milestone_1", {})
        m2 = dl.get("milestone_2", {})
        m3 = dl.get("milestone_3", {})
        ladder_text = (
            f"\n*🎯 Dynamic Ladder Execution:*\n"
            f"• *M1 ({m1.get('target_rr', '1:3')}):* `{m1.get('target_price', signal.get('take_profit'))}` (Close 50%, SL to {m1.get('ratchet_sl_rr', '+2.5 RR')})\n"
            f"• *M2 ({m2.get('target_rr', '1:4')}):* `{m2.get('target_price', '')}` (Close 25%, SL to {m2.get('ratchet_sl_rr', '+3.5 RR')})\n"
            f"• *M3 ({m3.get('target_rr', '1:5')}):* `{m3.get('target_price', '')}` (Runner Trailing)\n"
        )

    risk_val = signal.get("risk_usd", 1.0)
    rr_str = str(signal.get("rr_ratio", "1:3.0"))
    try:
        rr_mult = float(rr_str.split(":")[-1])
    except Exception:
        rr_mult = 3.0
    reward_val = risk_val * rr_mult

    message = f"""
🚨 *FOREX AI SIGNAL*

*Pair:* {signal.get('symbol')}
*Direction:* {direction_emoji} *{signal.get('direction')}*
*Strategy:* {signal.get('strategy', 'SMC Precision Sniper')}

*Entry:*       `{signal.get('entry_price')}`
*Stop Loss:*   `{signal.get('stop_loss')}` ({signal.get('sl_pips', 'N/A')} pips)
*Take Profit:* `{signal.get('take_profit')}` ({signal.get('tp_pips', 'N/A')} pips)

*Risk:*   ${risk_val:.2f}
*Reward:* ${reward_val:.2f} ({rr_str} RR)
*Lot:*    `{signal.get('lot_size', 0.01)}`
{ladder_text}
*Confirmations:*
{confirmations}

*Session:* {signal.get('session', 'Active')} | *Spread:* {signal.get('spread', 'N/A')} pips

⚠️ _Manual approval required on your MT5 mobile app before entering_
"""
    payload = {
        "chat_id":    chat_id,
        "text":       message.strip(),
        "parse_mode": "Markdown"
    }
    signal_id = signal.get("id")
    if signal_id:
        payload["reply_markup"] = {
            "inline_keyboard": [
                [
                    {"text": "✅ APPROVE TRADE", "callback_data": f"approve_{signal_id}"},
                    {"text": "❌ REJECT", "callback_data": f"reject_{signal_id}"}
                ]
            ]
        }

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json=payload, timeout=6)
        if resp.status_code == 200:
            logger.info(f"Telegram signal alert sent successfully to {chat_id}")
            return True
        else:
            logger.warning(f"Telegram API responded with {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False

def send_tp1_alert(chat_id: str, trade: dict) -> bool:
    """Alert user that Milestone 1 (TP1) has been reached and SL is moved to Breakeven."""
    token = _get_token()
    if not chat_id or not token:
        return False

    direction_emoji = "📈" if trade.get("direction") == "BUY" else "📉"
    text = f"""
🎯 *MILESTONE 1 (TP1) REACHED!*

*Pair:* {trade.get('symbol')}
*Direction:* {direction_emoji} *{trade.get('direction')}*
*Entry:* `{trade.get('entry')}`
*Live Price:* `{trade.get('current_price')}` (+{trade.get('pips')} pips)

🛡️ *Action:* Stop Loss moved to Breakeven (`{trade.get('breakeven_sl')}`)!
🔒 *This trade is now 100% RISK-FREE.*
🏃 Trailing runner moving towards TP2 & TP3...
"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": chat_id,
            "text": text.strip(),
            "parse_mode": "Markdown"
        }, timeout=6)
        return True
    except Exception as e:
        logger.error(f"Failed to send TP1 alert: {e}")
        return False

def send_trade_closed_alert(chat_id: str, trade: dict) -> bool:
    """Alert user when a trade finishes (WIN at TP, Breakeven, or Stop Loss)."""
    token = _get_token()
    if not chat_id or not token:
        return False

    result = trade.get("result", "WIN")
    if result == "WIN":
        icon = "🎉 🏆"
        title = "TRADE CLOSED IN FULL PROFIT!"
        outcome_text = f"💰 *Profit:* `+${trade.get('profit_usd'):.2f} USD` (+{trade.get('pips')} pips)"
    elif result == "BREAKEVEN":
        icon = "🛡️"
        title = "TRADE CLOSED AT BREAKEVEN"
        outcome_text = "🔒 *Profit:* `$0.00 USD` — Capital preserved 100% risk-free!"
    else:
        icon = "🛑"
        title = "STOP LOSS HIT"
        outcome_text = f"📉 *Loss:* `-${abs(trade.get('profit_usd', 1.0)):.2f} USD` (Risk strictly limited)"

    direction_emoji = "📈" if trade.get("direction") == "BUY" else "📉"
    text = f"""
{icon} *{title}*

*Pair:* {trade.get('symbol')}
*Direction:* {direction_emoji} *{trade.get('direction')}*
*Entry:* `{trade.get('entry')}`
*Exit:*  `{trade.get('close_price')}`
{outcome_text}

📊 *Updated on Dashboard:*
Open your web dashboard at `/trades` and `/performance` to view updated statistics.
"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": chat_id,
            "text": text.strip(),
            "parse_mode": "Markdown"
        }, timeout=6)
        return True
    except Exception as e:
        logger.error(f"Failed to send trade closed alert: {e}")
        return False

def edit_telegram_message(chat_id: str, message_id: int, text: str) -> bool:
    token = _get_token()
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/editMessageText"
    try:
        requests.post(url, json={
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=5)
        return True
    except Exception:
        return False

def answer_callback_query(callback_query_id: str, text: str) -> bool:
    token = _get_token()
    if not token:
        return False
    url = f"https://api.telegram.org/bot{token}/answerCallbackQuery"
    try:
        requests.post(url, json={
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": False
        }, timeout=5)
        return True
    except Exception:
        return False

def send_no_trade_alert(chat_id: str, symbol: str, reason: str):
    token = _get_token()
    if not chat_id or not token or token == "your_telegram_bot_token":
        return
    message = f"⏳ *{symbol}* — NO TRADE\n_{reason}_"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "Markdown"
        }, timeout=5)
    except Exception:
        pass

def test_telegram_bot(chat_id: str) -> dict:
    """Send a test message to verify Telegram bot credentials."""
    token = _get_token()
    if not token or token == "your_telegram_bot_token":
        return {"success": False, "message": "TELEGRAM_BOT_TOKEN is not configured in backend/.env"}
    if not chat_id:
        return {"success": False, "message": "Telegram Chat ID is missing. Enter your Chat ID in Settings."}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": "🤖 *Forex AI Telegram Test Alert*\n\n✅ Your Telegram Bot is connected and working perfectly! You will receive live trade alerts here.",
            "parse_mode": "Markdown"
        }, timeout=6)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return {"success": True, "message": "Test alert sent to your Telegram!"}
        else:
            return {"success": False, "message": f"Telegram Error: {data.get('description', 'Unknown error')}"}
    except Exception as e:
        return {"success": False, "message": f"Connection failed: {str(e)}"}
