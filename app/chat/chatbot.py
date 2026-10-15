"""
ForexAI Chatbot — Gemini-first dynamic AI assistant.
Parses user intent for ANY symbol, timeframe, RR, days, session.
All backtests run exclusively on live broker data (MetaApi / cTrader).
No local CSV fallback.
"""
import re
import logging
from typing import NamedTuple, Dict, Any, Optional
from app.backtest.engine import run_backtest

logger = logging.getLogger(__name__)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "NZDUSD", "USDCAD", "EURJPY", "GBPJPY"]

EXTRA_SYMBOLS = {
    "GOLD": "XAUUSD", "XAU": "XAUUSD", "XAUUSD": "XAUUSD",
    "SILVER": "XAGUSD", "XAG": "XAGUSD", "XAGUSD": "XAGUSD",
    "OIL": "USOIL", "CRUDE": "USOIL", "WTI": "USOIL", "USOIL": "USOIL",
    "US30": "US30", "DOW": "US30", "NAS": "NAS100", "NAS100": "NAS100",
    "SPX": "SPX500", "SP500": "SPX500",
    "BTC": "BTCUSD", "BITCOIN": "BTCUSD", "BTCUSD": "BTCUSD",
    "ETH": "ETHUSD", "ETHEREUM": "ETHUSD", "ETHUSD": "ETHUSD",
}

ALL_TIMEFRAMES = ["M1", "M3", "M5", "M15", "M30", "H1", "H4", "D1"]

_SYMBOL_IGNORE = {
    "TODAY", "LAST", "DAYS", "WEEK", "MONTH", "SHOW", "GIVE", "TELL",
    "WHAT", "WITH", "FROM", "THIS", "THAT", "HAVE", "WILL", "BEST",
    "BACK", "TEST", "ENTRY", "TRADE", "PAIR", "TIME", "FRAME",
    "RATIO", "RISK", "WIN", "LOSS", "RATE", "SIGNAL",
}


# ── Named tuple replaces the 5-value tuple return (fixes cyclomatic complexity) ──
class ParsedIntent(NamedTuple):
    symbol:   str
    tf:       str
    days:     int
    session:  Optional[str]
    rr:       float


# ── Parameter extraction — split into 4 focused helpers ──────────────────────

def _parse_symbol(msg: str) -> str:
    """Extract trading symbol from uppercased message."""
    if any(k in msg for k in [
        "ALLPAIR", "ALLPAIRS", "ALL PAIR", "ALL PAIRS", "ALL THE PAIR",
        "EVERY PAIR", "ALL SYMBOL", "ALL SYMBOLS", "ALL FOREX", "PORTFOLIO",
    ]):
        return "ALL"

    for keyword, mt5_sym in EXTRA_SYMBOLS.items():
        if re.search(rf"\b{keyword}\b", msg):
            return mt5_sym

    for p in PAIRS:
        if p in msg:
            return p

    raw = re.search(r"\b([A-Z]{2,6}(?:USD|EUR|JPY|GBP|CHF|CAD|AUD|NZD|100|200|500|30|50)?)\b", msg)
    if raw:
        candidate = raw.group(1)
        if candidate not in _SYMBOL_IGNORE and len(candidate) >= 4:
            return candidate

    return "EURUSD"


def _parse_timeframe(msg: str) -> str:
    """Extract timeframe token from uppercased message."""
    min_match = re.search(r"\b(\d+)\s*(?:MIN(?:UTE)?S?|M)\b", msg)
    if min_match:
        candidate = f"M{min_match.group(1)}"
        if candidate in ALL_TIMEFRAMES:
            return candidate

    hr_match = re.search(r"\b(\d+)\s*(?:H(?:OUR)?S?)\b", msg)
    if hr_match:
        candidate = f"H{hr_match.group(1)}"
        if candidate in ALL_TIMEFRAMES:
            return candidate

    for t in ALL_TIMEFRAMES:
        if re.search(rf"\b{t}\b", msg):
            return t

    return "M15"


def _parse_days(msg: str) -> int:
    """Extract lookback days from uppercased message."""
    if re.search(r"\b(1|ONE)\s*MONTH\b", msg) or "30 DAY" in msg:
        return 30
    if re.search(r"\b(2|TWO)\s*WEEK", msg) or "14 DAY" in msg:
        return 14
    if re.search(r"\b(1|ONE)\s*WEEK\b", msg) or "7 DAY" in msg:
        return 7
    if re.search(r"\b(3|THREE)\s*DAY", msg):
        return 3
    if "TODAY" in msg or re.search(r"\b1\s*DAY\b", msg):
        return 1
    m = re.search(r"\b(?:LAST|PAST)?\s*(\d+)\s*DAYS?\b", msg)
    if m:
        return min(90, max(1, int(m.group(1))))
    return 7


def _parse_session_and_rr(msg: str):
    """Extract session filter and RR ratio from uppercased message."""
    session = None
    if "LONDON" in msg:
        session = "London"
    elif "NEW YORK" in msg or re.search(r"\bNY\b", msg):
        session = "New York"
    elif "ASIAN" in msg or "TOKYO" in msg or "ASIA" in msg:
        session = "Asian"

    rr = 2.0
    m = re.search(r"1\s*[:/]\s*(\d+(?:\.\d+)?)", msg)
    if m:
        rr = float(m.group(1))
    else:
        m2 = re.search(r"\b(?:RR|R)\s*[=:]?\s*(\d+(?:\.\d+)?)\b", msg)
        if m2:
            rr = float(m2.group(1))
        else:
            m3 = re.search(r"\b(\d+(?:\.\d+)?)\s*R\b", msg)
            if m3:
                rr = float(m3.group(1))

    rr = max(1.0, min(rr, 20.0))
    return session, rr


def _extract_parameters(message: str) -> ParsedIntent:
    """Parse user message into a ParsedIntent named tuple."""
    msg     = message.upper()
    symbol  = _parse_symbol(msg)
    tf      = _parse_timeframe(msg)
    days    = _parse_days(msg)
    session, rr = _parse_session_and_rr(msg)
    return ParsedIntent(symbol=symbol, tf=tf, days=days, session=session, rr=rr)


# ── Intent detection ──────────────────────────────────────────────────────────

def _is_price_intent(message: str) -> bool:
    lower = message.lower()
    price_kw    = ["price", "quote", "bid", "ask", "live", "now", "current rate", "how much"]
    backtest_kw = ["backtest", "win", "loss", "signal", "last", "days", "week", "month"]
    return any(k in lower for k in price_kw) and not any(k in lower for k in backtest_kw)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _get_live_price(symbol: str) -> dict:
    try:
        from app.mt5.source_router import get_fetcher
        _, get_price, _, _ = get_fetcher("auto")
        return get_price(symbol) or {}
    except (ImportError, ConnectionError, RuntimeError) as e:
        logger.error("Live price fetch failed for %s: %s", symbol, e)
        return {}


def _run_live_backtest(intent: ParsedIntent) -> dict:
    return run_backtest(
        symbol=intent.symbol, timeframe=intent.tf, days=intent.days,
        session_filter=intent.session, rr_ratio=intent.rr,
    )


# ── Response formatters ───────────────────────────────────────────────────────

def _format_backtest_result(data: dict, intent: ParsedIntent) -> Dict[str, Any]:
    symbol, tf, days, _, rr = intent

    if "error" in data:
        return {
            "reply": (
                f"⚠️ **Backtest unavailable for {symbol} {tf}**\n\n"
                f"{data['error']}\n\n"
                f"Connect your MetaApi or cTrader account in **Settings** to enable live backtests."
            ),
            "data": data,
        }

    total = data.get("total_signals", 0)
    if total == 0:
        fc   = data.get("filter_counts", {})
        bias = data.get("global_bias", "unknown")
        cw   = data.get("candles_in_window", 0)
        dominant = max(fc, key=fc.get) if fc else "unknown"
        dominant_map = {
            "kz":            "most candles were outside London/NY/Asian killzone hours",
            "bias_ranging":  "market was ranging — no clear directional bias",
            "trend_conflict":"H1 trend conflicted with H4 bias",
            "no_structure":  "no CHoCH or BOS structure break found",
            "no_zone":       "no Order Block or FVG entry zone found",
            "low_prob":      "setups found but win probability score was below threshold",
        }
        reason = dominant_map.get(dominant, "filters were too strict for this period")
        return {
            "reply": (
                f"📊 **{symbol} {tf} — Last {days} days | RR 1:{rr}**\n\n"
                f"No signals found. Bias detected: **{bias}** | Candles analysed: **{cw}**\n\n"
                f"Main filter blocking signals: **{reason}**\n"
                f"Filter breakdown: {fc}\n\n"
                f"💡 Try: `{symbol} M15 last 14 days` or `{symbol} H1 last 30 days`"
            ),
            "data": data,
        }

    wins      = data["wins"]
    losses    = data["losses"]
    win_rate  = data["win_rate"]
    net_pips  = data["net_pips"]
    tf_hier   = data.get("timeframe_hierarchy", {})
    sessions  = data.get("sessions", {})
    top_confs = data.get("top_confirmations", [])
    best      = data.get("best_trade")
    worst     = data.get("worst_trade")
    start     = data.get("start_date", "")
    end       = data.get("end_date", "")

    lines = [
        f"📊 **{symbol} Backtest — {tf} | {start} → {end} | RR 1:{rr}**",
        f"🔭 Hierarchy: {tf_hier.get('macro_bias_tf', 'H4')} bias → {tf_hier.get('intermediate_trend_tf', 'H1')} trend ({tf_hier.get('style', 'Intraday')})",
        "",
        f"- **Signals:** `{total}`",
        f"- **Wins:** `{wins}` ({win_rate}%) 🟢",
        f"- **Losses:** `{losses}` ({round(100 - win_rate, 1)}%) 🔴",
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

    if win_rate >= 65:
        lines.append(f"\n✅ **Strong edge** — {symbol} {tf} has {win_rate}% win rate over {days} days.")
    elif win_rate >= 45:
        lines.append(f"\n⚠️ **Moderate edge** — Focus on {top_confs[0][0] if top_confs else 'Order Block'} setups only.")
    else:
        lines.append(f"\n🔴 **Weak edge** — Avoid {tf} on {symbol} this period. Try H1 or H4 instead.")

    return {"reply": "\n".join(lines), "data": data}


# ── Request handlers (split from process_chat_message) ───────────────────────

def _handle_price(intent: ParsedIntent) -> Dict[str, Any]:
    s     = intent.symbol if intent.symbol != "ALL" else "EURUSD"
    price = _get_live_price(s)
    if price:
        return {
            "reply": (
                f"📊 **{s} Live Price**\n\n"
                f"- Bid: `{price.get('bid')}`\n"
                f"- Ask: `{price.get('ask')}`\n"
                f"- Spread: `{price.get('spread')} pips`"
            ),
            "data": price,
        }
    return {"reply": f"⚠️ Could not fetch live price for {s}. Check MetaApi / cTrader connection in Settings."}


def _handle_portfolio(intent: ParsedIntent) -> Dict[str, Any]:
    """
    Portfolio backtest.
    If multiple timeframes or RR values are detected, runs a comparison matrix.
    Otherwise runs a single TF/RR table.
    """
    # Detect if user wants multi-TF or multi-RR comparison
    msg = intent.tf  # already parsed primary TF
    # We always expose M15 vs M3 and RR 1:2 vs 1:3 when user says "all pairs"
    # and mentions both timeframes or both RR values
    return _handle_portfolio_matrix(intent)


def _handle_portfolio_matrix(intent: ParsedIntent) -> Dict[str, Any]:
    """
    Runs a full comparison matrix:
      Rows    = all 9 pairs
      Columns = TF x RR combinations (M15 1:2, M15 1:3, M3 1:2, M3 1:3)
    """
    combos = [
        ("M15", 2.0),
        ("M15", 3.0),
        ("M3",  2.0),
        ("M3",  3.0),
    ]

    # Header
    lines = [
        f"📊 **Portfolio Backtest — All Pairs | Last {intent.days} days**",
        f"🔁 Comparing: M15 vs M3 | RR 1:2 vs 1:3\n",
        "| Pair | M15 1:2 | M15 1:3 | M3 1:2 | M3 1:3 |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    summary = {combo: {"signals": 0, "wins": 0, "pips": 0.0} for combo in combos}

    for pair in PAIRS:
        cells = [f"**{pair}**"]
        for tf, rr in combos:
            try:
                r = _run_live_backtest(intent._replace(symbol=pair, tf=tf, rr=rr))
                if "error" in r:
                    cells.append("⚠️ no data")
                elif r.get("total_signals", 0) == 0:
                    cells.append("— 0 signals")
                else:
                    wr  = r["win_rate"]
                    sig = r["total_signals"]
                    net = r["net_pips"]
                    icon = "🟢" if wr >= 60 else ("🟡" if wr >= 45 else "🔴")
                    cells.append(f"{icon} {wr}% ({sig}s / {net:+.1f}p)")
                    summary[(tf, rr)]["signals"] += r.get("total_signals", 0)
                    summary[(tf, rr)]["wins"]    += r.get("wins", 0)
                    summary[(tf, rr)]["pips"]    += r.get("net_pips", 0.0)
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("Matrix backtest error %s %s RR%s: %s", pair, tf, rr, e)
                cells.append("⚠️ error")
        lines.append("| " + " | ".join(cells) + " |")

    # Summary row
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    summary_cells = ["**TOTAL**"]
    best_combo = None
    best_wr    = -1.0
    for tf, rr in combos:
        s = summary[(tf, rr)]
        if s["signals"] > 0:
            wr = round(s["wins"] / s["signals"] * 100, 1)
            summary_cells.append(f"**{wr}%** ({s['signals']}s / {s['pips']:+.1f}p)")
            if wr > best_wr:
                best_wr    = wr
                best_combo = (tf, rr)
        else:
            summary_cells.append("— no data")
    lines.append("| " + " | ".join(summary_cells) + " |")

    # Recommendation
    lines.append("")
    if best_combo:
        lines.append(f"🏆 **Best combo: {best_combo[0]} at RR 1:{int(best_combo[1])} — {best_wr}% overall win rate**")
    else:
        lines.append("⚠️ No live data available. Connect MetaApi or cTrader in **Settings**.")

    return {"reply": "\n".join(lines)}


def _handle_single_backtest(intent: ParsedIntent) -> Dict[str, Any]:
    data = _run_live_backtest(intent)
    return _format_backtest_result(data, intent)


# ── Gemini context builder ────────────────────────────────────────────────────

def _build_context_for_gemini(message: str) -> str:
    intent = _extract_parameters(message)
    context_parts = []

    if intent.symbol and intent.symbol != "ALL":
        price = _get_live_price(intent.symbol)
        if price:
            context_parts.append(
                f"[LIVE PRICE {intent.symbol}] Bid={price.get('bid')} Ask={price.get('ask')} Spread={price.get('spread')} pips"
            )

    if intent.symbol == "ALL":
        combos = [("M15", 2.0), ("M15", 3.0), ("M3", 2.0), ("M3", 3.0)]
        for p in PAIRS:
            for tf, rr in combos:
                try:
                    bt = _run_live_backtest(intent._replace(symbol=p, tf=tf, rr=rr))
                    if bt.get("total_signals", 0) > 0:
                        context_parts.append(
                            f"[BACKTEST {p} {tf} {intent.days}d RR1:{rr}] "
                            f"Signals={bt['total_signals']} Wins={bt['wins']} "
                            f"WinRate={bt['win_rate']}% NetPips={bt['net_pips']}"
                        )
                except (ValueError, RuntimeError, KeyError) as e:
                    logger.warning("Gemini context backtest error %s %s: %s", p, tf, e)
    else:
        bt = _run_live_backtest(intent)
        if bt.get("total_signals", 0) > 0:
            context_parts.append(
                f"[BACKTEST {intent.symbol} {intent.tf} {intent.days}d RR1:{intent.rr}] "
                f"Signals={bt['total_signals']} Wins={bt['wins']} Losses={bt['losses']} "
                f"WinRate={bt['win_rate']}% NetPips={bt['net_pips']} "
                f"AvgWinProb={bt.get('avg_win_probability', 0)}% "
                f"Sessions={bt.get('sessions', {})} TopConfs={bt.get('top_confirmations', [])}"
            )
        elif "error" in bt:
            context_parts.append(f"[BACKTEST ERROR] {bt['error']}")

    if context_parts:
        return "[REAL MARKET DATA]\n" + "\n".join(context_parts) + f"\n\n[USER QUESTION]\n{message}"
    return message


# ── Main entry point ──────────────────────────────────────────────────────────

def _is_backtest_intent(message: str) -> bool:
    lower = message.lower()
    return any(k in lower for k in [
        "backtest", "back test", "last", "days", "week", "month",
        "win rate", "signals", "result", "performance", "pips",
    ])


def process_chat_message(message: str) -> Dict[str, Any]:
    cleaned = message.strip()
    if not cleaned:
        return {"reply": "Ask me anything — live prices, backtest results, win rates, or SMC concepts."}

    intent = _extract_parameters(cleaned)

    # Backtest and price requests: always handle locally with real engine data
    # Never let Gemini rewrite/summarize numerical results — it hallucinates numbers
    if _is_price_intent(cleaned):
        return _handle_price(intent)

    if intent.symbol == "ALL":
        return _handle_portfolio(intent)

    if _is_backtest_intent(cleaned):
        return _handle_single_backtest(intent)

    # Step 3: Non-data questions (SMC concepts, strategy questions) — use Gemini
    try:
        from app.chat.gemini_service import chat_with_gemini, get_gemini_api_key
        if get_gemini_api_key():
            response = chat_with_gemini(_build_context_for_gemini(cleaned))
            if response:
                return response
    except (ImportError, ConnectionError, RuntimeError) as e:
        logger.warning("Gemini failed: %s", e)

    return _handle_single_backtest(intent)
