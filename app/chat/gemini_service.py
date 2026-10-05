"""
Gemini 1.5 Flash AI Service with Function Calling (Tools).
Integrates Google Gemini to power the ForexAI conversational trading assistant.
"""
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Dict, Any, Optional
import google.generativeai as genai
from app.config import settings
from app.backtest.engine import run_backtest

logger = logging.getLogger(__name__)


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from settings, environment, or database."""
    key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if key and key.strip():
        return key.strip()
    try:
        from app.db.database import SessionLocal
        from app.db import crud
        from app.db.models import User
        db = SessionLocal()
        u = db.query(User).filter(User.mt5_login != None).first()
        if u:
            db_key = crud.get_setting(db, u.id, "gemini_api_key")
            if db_key and db_key.strip():
                db.close()
                return db_key.strip()
        db.close()
    except Exception:
        pass
    return ""


def run_backtest_tool(symbol: str = "EURUSD", timeframe: str = "M3", days: int = 7, session_filter: str = "", min_win_prob: int = 60) -> dict:
    """
    Run an SMC (Smart Money Concepts) backtest on historical Forex candles with Scanner Parity (Multi-Timeframe Hierarchy and >=60% Win Probability filter).
    
    Args:
        symbol: The currency pair to test (e.g. 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD').
        timeframe: Candle timeframe to test ('M1', 'M3', 'M5', 'M15', 'M30', 'H1', 'H4').
        days: Number of historical days to backtest (e.g. 7 for 1 week, 30 for 1 month).
        session_filter: Optional trading session filter ('London', 'New York', 'Asian', or empty for all).
        min_win_prob: Minimum win probability threshold to evaluate (default 60).
    """
    s_filter = session_filter if session_filter and session_filter.strip() else None
    return run_backtest(symbol=symbol, timeframe=timeframe, days=days, session_filter=s_filter, min_win_prob=min_win_prob)


def get_live_market_price_tool(symbol: str = "EURUSD") -> dict:
    """
    Get the current live market price, bid, ask, and spread for a currency pair directly from the broker.
    
    Args:
        symbol: The currency pair to check (e.g. 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD').
    """
    from app.mt5.source_router import get_fetcher
    _, get_live_price_fn, _, _ = get_fetcher("mt5")
    price = get_live_price_fn(symbol)
    if not price:
        _, get_live_price_fn, _, _ = get_fetcher("yfinance")
        price = get_live_price_fn(symbol)
    
    if not price:
        return {"symbol": symbol, "status": "No price available currently."}
        
    return {
        "symbol": symbol,
        "bid": price.get("bid"),
        "ask": price.get("ask"),
        "spread_points": price.get("spread"),
        "status": "Live Quote",
    }


SYSTEM_INSTRUCTION = """
You are ForexAI Analyst, an expert institutional Smart Money Concepts (SMC) trading assistant.
You have access to two real-time tools:
1. `get_live_market_price_tool`: Returns live bid, ask, and spread for any currency pair.
2. `run_backtest_tool`: Analyzes real MT5 broker candles using Scanner Parity entry rules (Multi-Timeframe Hierarchy: M1/M3/M5 checks H1+M15, M15 checks H4+H1, and filters for setups with >=60% Win Probability).

Guidelines:
1. When a user asks for current prices, live quotes, or bid/ask rates, call `get_live_market_price_tool`.
2. When a user asks about win rates, signal entries, timeframes (like M3, M5, M15), or performance over a period (e.g. last 1 week), call `run_backtest_tool`.
3. Format responses cleanly using GitHub Markdown:
   - For live prices: Show Symbol, Bid, Ask, Spread, and quick session context.
   - For backtests: Highlight Setup Quality (Scanner Parity >=60% Win Prob), Timeframe Hierarchy used, Total Signals, Wins, Losses, Win Rate %, Net Pips, Risk:Reward (1:2), and Session breakdowns.
4. If the user asks conceptual questions (e.g. "What is an Order Block?", "Explain BOS vs CHoCH"), answer clearly with practical trading examples.
5. Keep answers professional, data-driven, and concise.
"""


def chat_with_gemini(message: str) -> Optional[Dict[str, Any]]:
    """Send message to Gemini 2.5 Flash with function calling."""
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)

        available_tools = [run_backtest_tool, get_live_market_price_tool]

        # Use gemini-2.5-flash or gemini-flash-latest
        model_name = "gemini-2.5-flash"
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                tools=available_tools,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(message)
        except Exception:
            model = genai.GenerativeModel(
                model_name="gemini-flash-latest",
                tools=available_tools,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(message)

        if response and response.text:
            return {
                "reply": response.text,
                "source": "gemini-2.5-flash",
            }
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None

    return None
