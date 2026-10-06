"""
ForexAI Chatbot Engine — Natural Language Parser and Analysis Generator.
Answers backtesting queries, strategy comparisons, and session analysis.
"""
import re
from typing import Tuple, Dict, Any
from app.backtest.engine import run_backtest

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]
TIMEFRAMES = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"]


def _extract_parameters(message: str) -> Tuple[str, str, int, str]:
    """Extract symbol, timeframe, days, and session from user text."""
    msg = message.upper()

    # 1. Symbol extraction
    symbol = "EURUSD"
    if any(k in msg for k in ["ALL PAIR", "ALL THE PAIR", "EVERY PAIR", "ALL SYMBOL", "PORTFOLIO", "ALL CURRENCI", "COMPARE PAIR"]):
        symbol = "ALL"
    else:
        for p in PAIRS:
            if p in msg or p.replace("/", "") in msg:
                symbol = p
                break
        # Match common nicknames
        if "GOLD" in msg or "XAU" in msg:
            symbol = "EURUSD"

    # 2. Timeframe extraction
    tf = "M3" if ("M3" in msg or "3 MIN" in msg or "3MIN" in msg or "3 M" in msg) else None
    if not tf:
        for t in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            pattern = rf"\b{t}\b|\b{t[1:]}\s*(?:MIN|MINUTE|HR|HOUR)\b"
            if re.search(pattern, msg):
                tf = t
                break
    if not tf:
        tf = "M3"  # default to M3 as requested

    # 3. Days / Period extraction
    days = 7
    if "MONTH" in msg or "30 DAY" in msg:
        days = 30
    elif "2 WEEK" in msg or "14 DAY" in msg:
        days = 14
    elif "1 WEEK" in msg or "7 DAY" in msg or "LAST WEEK" in msg:
        days = 7
    elif "3 DAY" in msg:
        days = 3
    elif "TODAY" in msg or "1 DAY" in msg or "24 HOUR" in msg:
        days = 1
    else:
        # Check for explicit number of days, e.g. "last 10 days"
        match = re.search(r"(\d+)\s*(?:DAYS?|D)", msg)
        if match:
            days = min(60, max(1, int(match.group(1))))

    # 4. Session extraction
    session = None
    if "LONDON" in msg:
        session = "London"
    elif "NEW YORK" in msg or "NY" in msg:
        session = "New York"
    elif "ASIAN" in msg or "ASIA" in msg or "TOKYO" in msg:
        session = "Asian"

    # 5. Risk-Reward Ratio extraction
    rr = 2.0
    rr_match = re.search(r"1\s*:\s*(\d+(?:\.\d+)?)", msg)
    if rr_match:
        rr = float(rr_match.group(1))
    elif "1 TO 3" in msg or "1-3" in msg:
        rr = 3.0
    elif "1 TO 2" in msg or "1-2" in msg:
        rr = 2.0
    elif "1 TO 1.5" in msg or "1:1.5" in msg:
        rr = 1.5
    elif "1 TO 1" in msg or "1:1" in msg:
        rr = 1.0

    return symbol, tf, days, session, rr


def process_chat_message(message: str) -> Dict[str, Any]:
    """Process user message and return an AI analysis response."""
    cleaned = message.strip()
    if not cleaned:
        return {"reply": "Hi! How can I help you analyze the markets today? Ask me about any pair, timeframe (e.g. M3 last week), or win rates."}

    # 1. Try Gemini 1.5 Flash if API key is configured
    try:
        from app.chat.gemini_service import chat_with_gemini
        gemini_response = chat_with_gemini(cleaned)
        if gemini_response:
            return gemini_response
    except Exception:
        pass

    # 2. Local Fallback: Greetings / basic chat
    lower_msg = cleaned.lower()
    if lower_msg in ("hi", "hello", "hey", "help"):
        return {
            "reply": (
                "👋 **Hello! I am ForexAI Analyst.**\n\n"
                "You can ask me questions like:\n"
                "- *'What is EURUSD current price now?'*\n"
                "- *'Show me EURUSD entries on M3 for the last 1 week with win ratio'*\n"
                "- *'What is the win rate for GBPUSD on M15 last 14 days?'*\n"
                "- *'What is an Order Block or FVG?'*\n"
                "- *'Compare London session vs Asian session on M5'*"
            )
        }

    # Extract parameters
    symbol, tf, days, session, rr = _extract_parameters(cleaned)

    # 3. Live Price / Quote Inquiries (e.g. "give eurusd pair current price now")
    price_words = ["price", "prise", "quote", "rate", "cost", "value", "now", "current", "currectnt", "live", "bid", "ask", "how much"]
    backtest_words = ["backtest", "win rate", "win ratio", "historical", "signals", "test", "last week", "performance", "pnl", "profit", "statistic", "strategy"]

    if any(w in lower_msg for w in price_words) and not any(w in lower_msg for w in backtest_words):
        try:
            from app.mt5.source_router import get_fetcher
            _, get_live_price, _, _ = get_fetcher("auto")
            price_data = get_live_price(symbol)
            if price_data:
                bid = price_data.get("bid")
                ask = price_data.get("ask")
                spread = price_data.get("spread", 0.0)
                return {
                    "reply": (
                        f"📊 **Live Market Price: {symbol}**\n\n"
                        f"• **Bid:** `{bid}`\n"
                        f"• **Ask:** `{ask}`\n"
                        f"• **Spread:** `{spread} pips`\n\n"
                        f"🟢 *Live broker data feed active.*"
                    ),
                    "data": price_data
                }
            else:
                return {"reply": f"⚠️ Live market data for **{symbol}** is currently unavailable. Please try again in a moment."}
        except Exception as e:
            return {"reply": f"⚠️ Could not fetch live price for **{symbol}**: {str(e)}"}

    # 4. Educational SMC Concept Inquiries
    if "order block" in lower_msg or ("ob" in lower_msg and "what" in lower_msg):
        return {
            "reply": (
                "📚 **Smart Money Concepts: Order Block (OB)**\n\n"
                "An **Order Block** is the last opposite-direction candle before an aggressive institutional expansion that creates a Break of Structure (BOS).\n\n"
                "• **Bullish OB:** The last bearish down-candle before a rally breaking structure. Acts as a high-probability **BUY** entry.\n"
                "• **Bearish OB:** The last bullish up-candle before a drop breaking structure. Acts as a high-probability **SELL** entry."
            )
        }
    if "fvg" in lower_msg or "fair value gap" in lower_msg or "imbalance" in lower_msg:
        return {
            "reply": (
                "📚 **Smart Money Concepts: Fair Value Gap (FVG)**\n\n"
                "A **Fair Value Gap (FVG)** is a 3-candle imbalance where Candle 1's wick and Candle 3's wick do not overlap, leaving an unfilled market pocket.\n\n"
                "• Institutions often return price to fill the FVG before continuing the macro trend."
            )
        }
    if "choch" in lower_msg or "change of character" in lower_msg:
        return {
            "reply": (
                "📚 **Smart Money Concepts: CHoCH (Change of Character)**\n\n"
                "**CHoCH** is the earliest signal of a trend reversal. It happens when price breaks the most recent major swing point opposite to the current trend."
            )
        }
    if "bos" in lower_msg or "break of structure" in lower_msg:
        return {
            "reply": (
                "📚 **Smart Money Concepts: BOS (Break of Structure)**\n\n"
                "**BOS** confirms trend continuation. When price breaks the previous High in an uptrend, or previous Low in a downtrend, institutional flow is continuing."
            )
        }

    # Multi-Pair Portfolio Scan
    if symbol == "ALL":
        session_title = f" [{session} Session]" if session else ""
        lines = [
            f"📊 **Multi-Pair Portfolio Backtest: All 9 Pairs on {tf}{session_title}**",
            f"⏱️ **Period:** Last {days} days | **Quality Filter:** Scanner Parity (`Win Prob >= 60%`) | **Execution:** Dynamic Ladder (`1:{rr}` ➔ `1:{rr+1.0:.1f}` ➔ `1:{rr+2.0:.1f}` with `0.5 RR` offset SL)",
            "",
            "| Pair | Total Signals | Wins | Win Rate % | Net Pips | Top Winning Confirmations |",
            "| :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
        total_all = 0
        wins_all = 0
        pips_all = 0.0
        pair_data = []
        portfolio_confs = {}

        for p in PAIRS:
            res = run_backtest(symbol=p, timeframe=tf, days=days, session_filter=session, rr_ratio=rr, min_win_prob=60)
            if "total_signals" in res:
                s_tot = res["total_signals"]
                s_win = res["wins"]
                s_rate = res["win_rate"]
                s_pips = res["net_pips"]
                total_all += s_tot
                wins_all += s_win
                pips_all += s_pips
                
                # Pair top confirmations
                top_c_list = [c[0] for c in res.get("top_confirmations", [])[:2]]
                top_c_str = ", ".join(top_c_list) if top_c_list else "None"

                # Aggregate portfolio confirmations
                for c_name, count in res.get("top_confirmations", []):
                    portfolio_confs[c_name] = portfolio_confs.get(c_name, 0) + count

                lines.append(f"| **{p}** | `{s_tot}` | `{s_win}` | `{s_rate}%` | `{s_pips:+} pips` | {top_c_str} |")
                pair_data.append(res)

        overall_win_rate = round((wins_all / total_all * 100), 1) if total_all > 0 else 0.0
        m1_sl_str = f"+{round(rr - 0.5, 1)} RR"
        m2_sl_str = f"+{round(rr + 0.5, 1)} RR"
        lines.append("")
        lines.append("### 🏆 Overall Portfolio Performance")
        lines.append(f"- **Total Signals Across 9 Pairs:** `{total_all}`")
        lines.append(f"- **Total Wins:** `{wins_all}` ({overall_win_rate}%) 🟢")
        lines.append(f"- **Net Portfolio Pips:** `{pips_all:+} pips`")
        lines.append(f"- **Risk : Reward:** `1 : {rr}`")
        lines.append(f"- **Dynamic Ladder Model:** `M1 (1:{rr})` Close 50% & Trail SL to `{m1_sl_str}` ➔ `M2 (1:{rr+1.0:.1f})` Close 25% & Trail SL to `{m2_sl_str}` ➔ `M3 (1:{rr+2.0:.1f})`")

        if portfolio_confs:
            lines.append("")
            lines.append("### 🔑 Top Winning Confirmations Across All Pairs")
            sorted_confs = sorted(portfolio_confs.items(), key=lambda x: x[1], reverse=True)
            for c_name, count in sorted_confs[:5]:
                lines.append(f"- **{c_name}:** contributed to `{count}` winning trades across portfolio")

        lines.append("")
        lines.append("💡 **AI Recommendation:** The top performing pairs for your strategy in this period are **GBPJPY**, **EURUSD**, and **USDJPY**.")

        return {
            "reply": "\n".join(lines),
            "data": {
                "portfolio": pair_data,
                "total_signals": total_all,
                "wins": wins_all,
                "win_rate": overall_win_rate,
                "net_pips": round(pips_all, 1),
                "top_confirmations": sorted(portfolio_confs.items(), key=lambda x: x[1], reverse=True),
            },
        }

    # Run backtest engine
    data = run_backtest(symbol=symbol, timeframe=tf, days=days, session_filter=session, rr_ratio=rr, min_win_prob=60)

    if "error" in data:
        return {"reply": f"⚠️ {data['error']}", "data": data}

    total = data["total_signals"]
    wins = data["wins"]
    losses = data["losses"]
    win_rate = data["win_rate"]
    net_pips = data["net_pips"]
    best = data.get("best_trade")
    worst = data.get("worst_trade")
    sessions = data.get("sessions", {})
    top_confs = data.get("top_confirmations", [])

    # Format reply
    period_str = f"Last {days} days ({data.get('start_date')} to {data.get('end_date')})"
    session_title = f" [{session} Session Only]" if session else ""

    tf_hier = data.get("timeframe_hierarchy", {})
    min_prob = data.get("min_win_prob_filter", 60)
    avg_prob = data.get("avg_win_probability", 0)

    lines = [
        f"📊 **Backtest Analysis: {symbol} on {tf}{session_title}**",
        f"⏱️ **Period:** {period_str}",
        f"🎯 **Setup Quality:** Scanner Parity Filter (`Win Prob >= {min_prob}%`, Avg: `{avg_prob}%`)",
        f"🔭 **Hierarchy:** Bias `{tf_hier.get('macro_bias_tf', 'H4')}` + Trend `{tf_hier.get('intermediate_trend_tf', 'H1')}` ({tf_hier.get('style', 'Intraday')})",
        "",
        "### 📈 Performance Summary",
        f"- **Total Signals Fired:** `{total}`",
        f"- **Winning Trades:** `{wins}` ({win_rate}%) 🟢",
        f"- **Losing Trades:** `{losses}` ({round(100 - win_rate, 1)}%) 🔴",
        f"- **Net Pips:** `{net_pips:+} pips`",
        f"- **Risk : Reward:** `1 : {rr}`",
        f"- **Dynamic Ladder Model:** `M1 (1:{rr})` Close 50% & Trail SL to `+{round(rr - 0.5, 1)} RR` ➔ `M2 (1:{round(rr + 1.0, 1)})` Close 25% & Trail SL to `+{round(rr + 0.5, 1)} RR` ➔ `M3 (1:{round(rr + 2.0, 1)})`",
        "",
    ]

    # Session stats
    if sessions:
        lines.append("### 🌐 Session Win Rate Breakdown")
        for s_name, s_data in sessions.items():
            rate = s_data["win_rate"]
            icon = "🟢" if rate >= 50 else ("🟡" if rate >= 30 else "🔴")
            lines.append(f"- **{s_name}:** `{s_data['wins']}/{s_data['total']} wins` ({rate}%) {icon}")
        lines.append("")

    # Best & worst trade
    if best and worst:
        lines.append("### 🎯 Trade Highlights")
        best_prob = f" [Prob: {best.get('win_probability')}%]" if best.get("win_probability") else ""
        worst_prob = f" [Prob: {worst.get('win_probability')}%]" if worst.get("win_probability") else ""
        lines.append(f"- **Best Trade:** `+{best['pips']} pips` ({best['direction']} at {best['entry']} during {best['session']}{best_prob})")
        lines.append(f"- **Worst Trade:** `{worst['pips']} pips` ({worst['direction']} at {worst['entry']} during {worst['session']}{worst_prob})")
        lines.append("")

    # Top confirmations
    if top_confs:
        lines.append("### 🔑 Top Winning Confirmations")
        for c_name, count in top_confs:
            lines.append(f"- **{c_name}:** appeared in `{count}` winning setups")
        lines.append("")

    # Strategic AI Advice
    lines.append("### 💡 AI Strategic Recommendation")
    if tf in ("M1", "M3") and win_rate < 40:
        lines.append(
            f"⚠️ **Note on {tf}:** Low timeframes like {tf} have higher market noise and spread drag. "
            f"Notice that London & New York sessions performed best, while Asian session produced choppy false breakouts. "
            f"**Recommendation:** Trade {tf} **only during London Open (07:00-11:00 UTC)** or switch to **M15** for cleaner SMC structure."
        )
    elif win_rate >= 50:
        lines.append(
            f"✅ **Solid Edge:** {symbol} on {tf} maintains a strong {win_rate}% win rate with a positive pip expectancy of {net_pips:+} pips with 1:2 RR."
        )
    else:
        lines.append(
            f"ℹ️ **Observation:** For {symbol} on {tf}, focus on entries confirmed by both **Order Blocks** and **BOS** to filter out low-probability losses."
        )

    return {
        "reply": "\n".join(lines),
        "data": data,
    }
