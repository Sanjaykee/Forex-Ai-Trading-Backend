"""
Gemini 1.5 Flash AI Service with Function Calling (Tools).
Integrates Google Gemini to power the ForexAI conversational trading assistant.
"""
import os
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from typing import Dict, Any, Optional
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from app.config import settings
from app.backtest.engine import run_backtest

logger = logging.getLogger(__name__)


def get_gemini_api_key() -> str:
    """Retrieve Gemini API key from settings, environment, or database."""
    candidates = [settings.GEMINI_API_KEY, os.getenv("GEMINI_API_KEY", "")]
    try:
        from app.db.database import SessionLocal
        from app.db import crud
        from app.db.models import User
        db = SessionLocal()
        u = db.query(User).filter(User.mt5_login != None).first()
        if u:
            db_key = crud.get_setting(db, u.id, "gemini_api_key")
            if db_key:
                candidates.append(db_key)
        db.close()
    except Exception:
        pass

    for k in candidates:
        if k and k.strip() and k.strip().startswith("AIza"):
            return k.strip()
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
2. `run_backtest_tool`: Analyzes real MT5 broker candles using Scanner Parity entry rules.

The user message may contain [REAL MARKET DATA] blocks with live backtest results and loss analysis.
Always use this real data in your answer — never make up numbers.

## Response Rules:

1. LIVE PRICE questions → call `get_live_market_price_tool`, show Bid/Ask/Spread + session context.

2. BACKTEST questions → use the [BACKTEST ...] data already injected. Show:
   - Total Signals, Wins, Losses, Win Rate %, Net Pips
   - Session breakdown (which session performed best/worst)
   - Top confirmations that led to wins

3. LOW WIN RATE (below 65%) → when [LOSS PATTERN ANALYSIS] is present, diagnose and suggest fixes:
   - If WorstSession has >50% of losses → suggest blocking that session
   - If MarketEntryLossPct > 60% → suggest requiring OB or FVG entry only
   - If AvgConfirmationsOnLoss < 2.5 → suggest raising min confirmations to 3
   - If AvgWinProbOnLoss < 68% → suggest raising win probability filter to 70%
   - If PureLosses is high → suggest tightening SL or reducing lot size
   Always give the SPECIFIC fix with the exact parameter to change.
   Format as numbered list: Problem → Root Cause → Fix → Expected Impact.

4. SMC concept questions → explain clearly with practical examples.

5. Keep answers concise, data-driven, and professional.
"""


def chat_with_gemini(message: str) -> Optional[Dict[str, Any]]:
    """Send message to Gemini with real market data already injected in message context."""
    if not genai:
        return None
    api_key = get_gemini_api_key()
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        available_tools = [run_backtest_tool, get_live_market_price_tool]

        for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
            try:
                model = genai.GenerativeModel(
                    model_name=model_name,
                    tools=available_tools,
                    system_instruction=SYSTEM_INSTRUCTION,
                )
                chat = model.start_chat(enable_automatic_function_calling=True)
                response = chat.send_message(message)
                if response and response.text:
                    return {"reply": response.text, "source": model_name}
                break
            except Exception as model_err:
                logger.warning(f"Model {model_name} failed: {model_err}")
                continue
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")

    return None
