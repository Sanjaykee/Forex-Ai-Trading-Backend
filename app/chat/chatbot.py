"""
ForexAI Chatbot — Gemini-first dynamic AI assistant.
Every message goes to Gemini with real market data injected as context.
Local fallback only runs backtest engine directly — no static keyword answers.
"""
import re
import logging
from typing import Dict, Any, Tuple, Optional
from app.backtest.engine import run_backtest

logger = logging.getLogger(__name__)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

# Symbols supported for live price (MetaApi/MT5 symbol names)
EXTRA_SYMBOLS = {
    # Gold
    "GOLD": "XAUUSD", "XAU": "XAUUSD", "XAUUSD": "XAUUSD",
    # Silver
    "SILVER": "XAGUSD", "XAG": "XAGUSD", "XAGUSD": "XAGUSD",
    # Oil
    "OIL": "USOIL", "CRUDE": "USOIL", "WTI": "USOIL", "USOIL": "USOIL",
    # Indices
    "US30": "US30", "DOW": "US30", "NAS": "NAS100", "NAS100": "NAS100",
    "SPX": "SPX500", "SP500": "SPX500",
    # Bitcoin
    "BTC": "BTCUSD", "BITCOIN": "BTCUSD", "BTCUSD": "BTCUSD",
}

# Only these pairs have local CSV sample data — MetaApi can fetch ANY symbol when connected
BACKTEST_SUPPORTED_LOCAL = set(PAIRS)


def _extract_parameters(message: str) -> Tuple[str, str, int, Optional[str], float]:
    msg = message.upper()

    symbol = None

    # 1a. Check for ALL pairs keyword
    if any(k in msg for k in [
        "ALLPAIR", "ALLPAIRS", "ALL PAIR", "ALL PAIRS", "ALL THE PAIR", "ALL THE PAIRS",
        "EVERY PAIR", "EVERY PAIRS", "EVERYPAIR", "ALL SYMBOL", "ALL SYMBOLS",
        "ALL FOREX", "PORTFOLIO", "COMPARE PAIR", "COMPARE PAIRS", "ALL THE SYMBOL"
    ]):
        symbol = "ALL"

    # 1b. Check known extra symbols (Gold, Silver, Oil, BTC etc)
    if not symbol:
        for keyword, mt5_symbol in EXTRA_SYMBOLS.items():
            if keyword in msg:
                symbol = mt5_symbol
                break

    # 1c. Check standard forex pairs
    if not symbol:
        for p in PAIRS:
            if p in msg:
                symbol = p
                break

    # 1d. Detect any raw MT5-style symbol typed directly (e.g. NAS100, US30, ETHUSD, XBRUSD)
    if not symbol:
        raw_match = re.search(r'\b([A-Z]{2,6}(?:USD|EUR|JPY|GBP|CHF|CAD|AUD|NZD|100|200|500|30|50)?)\b', msg)
        if raw_match:
            candidate = raw_match.group(1)
            # Ignore common English words that match the pattern
            ignore = {"TODAY", "LAST", "DAYS", "WEEK", "MONTH", "SHOW", "GIVE", "TELL",
                      "WHAT", "WITH", "FROM", "THIS", "THAT", "HAVE", "WILL", "BEST"}
            if candidate not in ignore and len(candidate) >= 4:
                symbol = candidate

    # 1e. Final fallback
    if not symbol:
        symbol = "EURUSD"

    tf = None
    if "M3" in msg or "3 MIN" in msg or "3MIN" in msg:
        tf = "M3"
    if not tf:
        for t in ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]:
            if re.search(rf"\b{t}\b", msg):
                tf = t
                break
    tf = tf or "M15"

    days = 7
    if "MONTH" in msg or "30 DAY" in msg:
        days = 30
    elif "2 WEEK" in msg or "14 DAY" in msg:
        days = 14
    elif "3 DAY" in msg:
        days = 3
    elif "TODAY" in msg or "1 DAY" in msg:
        days = 1
    else:
        m = re.search(r"(\d+)\s*(?:DAYS?|D)\b", msg)
        if m:
            days = min(60, max(1, int(m.group(1))))

    session = None
    if "LONDON" in msg:
        session = "London"
    elif "NEW YORK" in msg or "NY " in msg:
        session = "New York"
    elif "ASIAN" in msg or "TOKYO" in msg:
        session = "Asian"

    rr = 2.0
    m = re.search(r"1\s*:\s*(\d+(?:\.\d+)?)", msg)
    if m:
        rr = float(m.group(1))

    return symbol, tf, days, session, rr


def _analyse_losses(bt: dict) -> str:
    """Analyse losing trade patterns and return a structured diagnosis string for Gemini."""
    all_trades = bt.get("all_trades", bt.get("recent_trades", []))
    losing = [t for t in all_trades if t.get("outcome") == "LOSS"]
    winning = [t for t in all_trades if t.get("outcome") == "WIN"]

    if not losing:
        return ""

    # Session pattern
    loss_sessions = [t.get("session", "Unknown") for t in losing]
    win_sessions  = [t.get("session", "Unknown") for t in winning]
    session_counts = {}
    for s in loss_sessions:
        session_counts[s] = session_counts.get(s, 0) + 1
    worst_session = max(session_counts, key=session_counts.get) if session_counts else "Unknown"
    worst_session_pct = round(session_counts.get(worst_session, 0) / len(losing) * 100) if losing else 0

    # Entry type pattern
    loss_entry_types = [t.get("entry_type", "market") for t in losing]
    win_entry_types  = [t.get("entry_type", "market") for t in winning]
    market_loss_pct  = round(loss_entry_types.count("market") / len(losing) * 100) if losing else 0
    market_win_pct   = round(win_entry_types.count("market") / len(winning) * 100) if winning else 0

    # Confirmation count pattern
    loss_conf_counts = [len(t.get("confirmations", [])) for t in losing]
    win_conf_counts  = [len(t.get("confirmations", [])) for t in winning]
    avg_loss_confs   = round(sum(loss_conf_counts) / len(loss_conf_counts), 1) if loss_conf_counts else 0
    avg_win_confs    = round(sum(win_conf_counts)  / len(win_conf_counts),  1) if win_conf_counts  else 0

    # Win probability pattern
    loss_probs = [t.get("win_probability", 0) for t in losing]
    win_probs  = [t.get("win_probability", 0) for t in winning]
    avg_loss_prob = round(sum(loss_probs) / len(loss_probs), 1) if loss_probs else 0
    avg_win_prob  = round(sum(win_probs)  / len(win_probs),  1) if win_probs  else 0

    # Ladder milestone pattern (did losses hit any milestone before stopping out?)
    loss_milestones = [t.get("ladder_milestone", "None") for t in losing]
    pure_losses = loss_milestones.count("None")  # stopped out before any milestone

    return (
        f"[LOSS PATTERN ANALYSIS] "
        f"TotalLosses={len(losing)} TotalWins={len(winning)} "
        f"WorstSession={worst_session}({worst_session_pct}%ofLosses) "
        f"MarketEntryLossPct={market_loss_pct}% MarketEntryWinPct={market_win_pct}% "
        f"AvgConfirmationsOnLoss={avg_loss_confs} AvgConfirmationsOnWin={avg_win_confs} "
        f"AvgWinProbOnLoss={avg_loss_prob}% AvgWinProbOnWin={avg_win_prob}% "
        f"PureLosses(noMilestone)={pure_losses}/{len(losing)}"
    )


def _get_live_price(symbol: str) -> dict:
    try:
        from app.mt5.source_router import get_fetcher
        _, get_price, _, _ = get_fetcher("auto")
        return get_price(symbol) or {}
    except Exception:
        return {}


def _build_context_for_gemini(message: str) -> str:
    """Fetch real market data and inject it into the message as context for Gemini."""
    symbol, tf, days, session, rr = _extract_parameters(message)
    context_parts = []

    # Inject live price
    price_symbol = symbol if symbol != "ALL" else None
    if price_symbol:
        price = _get_live_price(price_symbol)
        if price:
            context_parts.append(
                f"[LIVE PRICE {price_symbol}] Bid={price.get('bid')} Ask={price.get('ask')} Spread={price.get('spread')} pips"
            )

    # Inject backtest data
    if symbol == "ALL":
        # Fetch real data for all pairs and inject as context
        for p in PAIRS:
            try:
                bt = run_backtest(symbol=p, timeframe=tf, days=days, session_filter=session, rr_ratio=rr)
                if "total_signals" in bt and bt["total_signals"] > 0:
                    context_parts.append(
                        f"[BACKTEST {p} {tf} last {days} days] "
                        f"Signals={bt['total_signals']} Wins={bt['wins']} Losses={bt['losses']} "
                        f"WinRate={bt['win_rate']}% NetPips={bt['net_pips']}"
                    )
            except Exception as err:
                logger.warning(f"Error backtesting {p} in context builder: {err}")
    elif symbol in BACKTEST_SUPPORTED_LOCAL:
        bt = run_backtest(symbol=symbol, timeframe=tf, days=days, session_filter=session, rr_ratio=rr)
        if "total_signals" in bt and bt["total_signals"] > 0:
            context_parts.append(
                f"[BACKTEST {symbol} {tf} last {days} days] "
                f"Signals={bt['total_signals']} Wins={bt['wins']} Losses={bt['losses']} "
                f"WinRate={bt['win_rate']}% NetPips={bt['net_pips']} "
                f"AvgWinProb={bt.get('avg_win_probability', 0)}% "
                f"Sessions={bt.get('sessions', {})} "
                f"TopConfirmations={bt.get('top_confirmations', [])}"
            )
            # Inject loss pattern analysis when win rate is below 65%
            if bt['win_rate'] < 65:
                loss_analysis = _analyse_losses(bt)
                if loss_analysis:
                    context_parts.append(loss_analysis)
        elif "error" in bt:
            context_parts.append(f"[BACKTEST NOTE] {bt['error']}")
    else:
        context_parts.append(
            f"[NOTE] {symbol} is not in the backtest database (only Forex pairs supported for backtest). "
            f"Live price may be available if MetaApi is connected."
        )

    if context_parts:
        return f"[REAL MARKET DATA]\n" + "\n".join(context_parts) + f"\n\n[USER QUESTION]\n{message}"
    return message


def process_chat_message(message: str) -> Dict[str, Any]:
    cleaned = message.strip()
    if not cleaned:
        return {"reply": "Ask me anything — live prices, backtest results, win rates, or SMC concepts like Order Blocks and FVG."}

    # Step 1 — Try Gemini with real data injected as context
    try:
        from app.chat.gemini_service import chat_with_gemini, get_gemini_api_key
        if get_gemini_api_key():
            enriched_message = _build_context_for_gemini(cleaned)
            gemini_response = chat_with_gemini(enriched_message)
            if gemini_response:
                return gemini_response
    except Exception as e:
        logger.warning(f"Gemini failed: {e}")

    # Step 2 — Gemini unavailable: run backtest/price engine directly and return real data
    symbol, tf, days, session, rr = _extract_parameters(cleaned)
    lower = cleaned.lower()

    # Live price request
    if any(w in lower for w in ["price", "quote", "bid", "ask", "live", "now", "current"]) \
            and not any(w in lower for w in ["backtest", "win", "loss", "signal", "last", "days", "week", "month"]):
        s = symbol if symbol != "ALL" else "EURUSD"
        price = _get_live_price(s)
        if price:
            return {
                "reply": (
                    f"📊 **{s} Live Price**\n\n"
                    f"- Bid: `{price.get('bid')}`\n"
                    f"- Ask: `{price.get('ask')}`\n"
                    f"- Spread: `{price.get('spread')} pips`"
                ),
                "data": price
            }
        return {"reply": f"⚠️ Could not fetch live price for {s}. Check MetaApi connection."}

    # All pairs portfolio backtest
    if symbol == "ALL":
        lines = [
            f"📊 **Portfolio Backtest — All Pairs on {tf}, Last {days} days**\n",
            "| Pair | Signals | Wins | Win Rate | Net Pips |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        total_signals, total_wins, total_pips = 0, 0, 0.0
        for p in PAIRS:
            try:
                r = run_backtest(symbol=p, timeframe=tf, days=days, session_filter=session, rr_ratio=rr)
                if "total_signals" in r:
                    total_signals += r["total_signals"]
                    total_wins += r["wins"]
                    total_pips += r["net_pips"]
                    lines.append(f"| **{p}** | {r['total_signals']} | {r['wins']} | {r['win_rate']}% | {r['net_pips']:+} pips |")
            except Exception as err:
                logger.warning(f"Error backtesting {p}: {err}")
        overall_wr = round(total_wins / total_signals * 100, 1) if total_signals > 0 else 0
        lines += ["", f"**Total:** {total_signals} signals | {total_wins} wins | **{overall_wr}% win rate** | {total_pips:+.1f} pips"]
        return {"reply": "\n".join(lines)}

    # Single pair backtest — works for ANY symbol when MetaApi is connected
    if symbol not in BACKTEST_SUPPORTED_LOCAL:
        # Try backtest anyway via MetaApi live candles
        data = run_backtest(symbol=symbol, timeframe=tf, days=days, session_filter=session, rr_ratio=rr)
        if "error" in data:
            # MetaApi not connected or symbol not available — show live price + clear message
            price = _get_live_price(symbol)
            if price:
                return {
                    "reply": (
                        f"📊 **{symbol} Live Price**\n\n"
                        f"- Bid: `{price.get('bid')}`\n"
                        f"- Ask: `{price.get('ask')}`\n"
                        f"- Spread: `{price.get('spread')} pips`\n\n"
                        f"⚠️ Backtest for {symbol} requires MetaApi to be connected in Settings."
                    ),
                    "data": price
                }
            return {"reply": f"⚠️ {symbol} backtest requires MetaApi connection. Connect in Settings first."}
    else:
        data = run_backtest(symbol=symbol, timeframe=tf, days=days, session_filter=session, rr_ratio=rr)

    if "error" in data:
        price = _get_live_price(symbol)
        if price:
            return {
                "reply": (
                    f"📊 **{symbol} Live Price**\n\n"
                    f"- Bid: `{price.get('bid')}`\n"
                    f"- Ask: `{price.get('ask')}`\n"
                    f"- Spread: `{price.get('spread')} pips`\n\n"
                    f"⚠️ Backtest not available for {symbol} — no historical candle data found.\n"
                    f"For backtest, use one of: {', '.join(PAIRS)}"
                ),
                "data": price
            }
        return {
            "reply": (
                f"⚠️ **{symbol} not found.**\n\n"
                f"Neither live price nor candle data is available for `{symbol}`.\n"
                f"Check the symbol name — it must match exactly what your broker uses in MT5.\n\n"
                f"Supported Forex pairs for backtest:\n"
                + "\n".join(f"- {p}" for p in PAIRS)
            )
        }

    total = data["total_signals"]
    if total == 0:
        return {
            "reply": (
                f"📊 **{symbol} {tf} — Last {days} days**\n\n"
                f"No signals met the ≥60% win probability filter in this period.\n"
                f"Try a longer period (e.g. last 30 days) or a different timeframe."
            ),
            "data": data
        }

    wins, losses = data["wins"], data["losses"]
    win_rate, net_pips = data["win_rate"], data["net_pips"]
    tf_hier = data.get("timeframe_hierarchy", {})
    sessions = data.get("sessions", {})
    top_confs = data.get("top_confirmations", [])
    best = data.get("best_trade")
    worst = data.get("worst_trade")

    lines = [
        f"📊 **{symbol} Backtest — {tf} | Last {days} days**",
        f"🔭 Hierarchy: {tf_hier.get('macro_bias_tf','H4')} bias → {tf_hier.get('intermediate_trend_tf','H1')} trend ({tf_hier.get('style','Intraday')})",
        "",
        f"- **Signals:** `{total}`",
        f"- **Wins:** `{wins}` ({win_rate}%) 🟢",
        f"- **Losses:** `{losses}` ({round(100-win_rate,1)}%) 🔴",
        f"- **Net Pips:** `{net_pips:+}`",
        f"- **RR:** `1:{rr}`",
    ]

    if sessions:
        lines.append("\n**Sessions:**")
        for s_name, s_data in sessions.items():
            icon = "🟢" if s_data["win_rate"] >= 50 else "🔴"
            lines.append(f"- {s_name}: {s_data['wins']}/{s_data['total']} ({s_data['win_rate']}%) {icon}")

    if top_confs:
        lines.append("\n**Top Confirmations:**")
        for c_name, count in top_confs:
            lines.append(f"- {c_name}: {count} wins")

    if best:
        lines.append(f"\n🏆 Best: `+{best['pips']} pips` ({best['direction']} @ {best['entry']}, {best['session']})")
    if worst:
        lines.append(f"💔 Worst: `{worst['pips']} pips` ({worst['direction']} @ {worst['entry']}, {worst['session']})")

    # Dynamic recommendation based on actual data
    if win_rate >= 65:
        lines.append(f"\n✅ **Strong edge** — {symbol} {tf} has {win_rate}% win rate over {days} days.")
    elif win_rate >= 45:
        lines.append(f"\n⚠️ **Moderate edge** — Focus on {top_confs[0][0] if top_confs else 'Order Block'} setups only.")
        # Add loss diagnosis in local fallback
        loss_diag = _analyse_losses(data)
        if loss_diag:
            # Parse key metrics from the diagnosis string for readable output
            lines.append("\n**🔍 Loss Pattern Detected:**")
            if "WorstSession" in loss_diag:
                import re as _re
                ws = _re.search(r"WorstSession=(\w+\s?\w*)\(", loss_diag)
                wp = _re.search(r"\((\d+)%ofLosses\)", loss_diag)
                me = _re.search(r"MarketEntryLossPct=(\d+)%", loss_diag)
                ac = _re.search(r"AvgConfirmationsOnLoss=([\d.]+)", loss_diag)
                ap = _re.search(r"AvgWinProbOnLoss=([\d.]+)%", loss_diag)
                if ws and wp and int(wp.group(1)) > 50:
                    lines.append(f"- {ws.group(1)} session = {wp.group(1)}% of losses → consider blocking it")
                if me and int(me.group(1)) > 60:
                    lines.append(f"- {me.group(1)}% of losses are market entries → require OB/FVG entry only")
                if ac and float(ac.group(1)) < 2.5:
                    lines.append(f"- Avg {ac.group(1)} confirmations on losses → raise minimum to 3")
                if ap and float(ap.group(1)) < 68:
                    lines.append(f"- Avg win probability on losses = {ap.group(1)}% → raise filter to 70%")
    else:
        lines.append(f"\n🔴 **Weak edge** — Avoid {tf} on {symbol} this period. Try H1 or H4 instead.")

    return {"reply": "\n".join(lines), "data": data}
