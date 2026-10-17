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
    Portfolio backtest — all 9 pairs on the user's exact TF and RR.
    Simple table: one column per pair, user's TF and RR only.
    """
    tf   = intent.tf
    rr   = intent.rr
    days = intent.days

    day_label = "Today" if days == 1 else f"Last {days} days"
    lines = [
        f"📊 **Portfolio Backtest — All Pairs | {day_label} | {tf} | RR 1:{rr}**\n",
        "| Pair | Signals | Wins | Losses | WR% | Net Pips |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    total_sig = total_w = total_l = 0
    total_pips = 0.0

    for pair in PAIRS:
        try:
            r = _run_live_backtest(intent._replace(symbol=pair))
            if "error" in r or r.get("total_signals", 0) == 0:
                lines.append(f"| **{pair}** | 0 | — | — | — | — |")
            else:
                sig  = r["total_signals"]
                w    = r["wins"]
                l    = r["losses"]
                wr   = r["win_rate"]
                net  = r["net_pips"]
                icon = "🟢" if wr >= 60 else ("🟡" if wr >= 45 else "🔴")
                lines.append(f"| **{pair}** | {sig} | {w} | {l} | {icon} {wr}% | {net:+.1f}p |")
                total_sig  += sig
                total_w    += w
                total_l    += l
                total_pips += net
        except Exception as e:
            logger.warning("Portfolio backtest error %s: %s", pair, e)
            lines.append(f"| **{pair}** | ⚠️ error | — | — | — | — |")

    overall_wr = round(total_w / total_sig * 100, 1) if total_sig else 0
    icon_total = "🟢" if overall_wr >= 60 else ("🟡" if overall_wr >= 45 else "🔴")
    lines.append(f"| **TOTAL** | **{total_sig}** | **{total_w}** | **{total_l}** | {icon_total} **{overall_wr}%** | **{total_pips:+.1f}p** |")

    lines.append("")
    if total_sig == 0:
        lines.append(f"⚠️ No signals found on {tf} today. Market may be outside killzone hours or no clean pattern formed.")
    elif overall_wr >= 60:
        lines.append(f"✅ **Strong day** — {overall_wr}% win rate across {total_sig} signals on {tf} RR 1:{rr}")
    else:
        lines.append(f"⚠️ **Moderate day** — {overall_wr}% win rate. Focus on high-score setups only.")

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
        for p in PAIRS:
            try:
                bt = _run_live_backtest(intent._replace(symbol=p))
                if bt.get("total_signals", 0) > 0:
                    context_parts.append(
                        f"[BACKTEST {p} {intent.tf} {intent.days}d RR1:{intent.rr}] "
                        f"Signals={bt['total_signals']} Wins={bt['wins']} "
                        f"WinRate={bt['win_rate']}% NetPips={bt['net_pips']}"
                    )
            except (ValueError, RuntimeError, KeyError) as e:
                logger.warning("Gemini context backtest error %s: %s", p, e)
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
        "win rate", "result", "performance", "pips",
    ])


def _is_live_signal_intent(message: str) -> bool:
    lower = message.lower()
    live_kw = ["live signal", "scan now", "any signal", "signal now",
               "current signal", "active signal", "find signal",
               "give signal", "show signal", "what signal", "trade now",
               "entry now", "setup now", "any setup", "scan all"]
    return any(k in lower for k in live_kw)


def _handle_live_scan() -> Dict[str, Any]:
    """Run live scanner on all pairs M15+M3 and format results."""
    try:
        from app.signals.scanner import scan_all_pairs
        signals = scan_all_pairs(timeframes=("M15", "M3"), min_confidence="LOW")
    except Exception as e:
        return {"reply": f"⚠️ Scanner error: {e}"}

    if not signals:
        return {
            "reply": (
                "🔍 **Live Scan — All Pairs (M15 + M3)**\n\n"
                "No signals right now.\n"
                "Market is either outside London/NY killzone or no clean CHoCH/BOS pattern forming.\n\n"
                "💡 Best times to scan: **07:00–11:00 UTC** (London) or **12:00–17:00 UTC** (NY)"
            )
        }

    conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    pattern_short = {
        "A_CHOCH":     "CHoCH+OB",
        "B_BOS_SWEEP": "BOS+Sweep",
        "C_ACCUM":     "Accum+OB",
    }

    lines = [f"🔍 **Live Scan — {len(signals)} Signal(s) Found**\n"]

    for s in signals:
        icon   = conf_icon.get(s["confidence"], "⚪")
        pt     = pattern_short.get(s["pattern"], s["pattern"])
        htf_ok = "✅" if s.get("htf_aligned") else "⚠️"
        ind    = "✅ Inducement" if s.get("inducement") else ""
        fvg    = "✅ FVG" if s.get("fvg") else ""
        extras = "  ".join(filter(None, [ind, fvg]))

        lines.append(
            f"{icon} **{s['symbol']} {s['timeframe']} — {s['direction']}** | {pt} | Score: {s['score']}\n"
            f"  Entry: `{s['entry']}`  SL: `{s['sl']}`  TP(1:3): `{s['tp_1_3']}`\n"
            f"  SL: {s['sl_pips']} pips | HTF: {s['htf_bias']} {htf_ok} | Session: {s['session']}\n"
            + (f"  {extras}\n" if extras else "")
        )

    high = [s for s in signals if s["confidence"] == "HIGH"]
    if high:
        lines.append(f"\n🏆 **Best setup: {high[0]['symbol']} {high[0]['timeframe']} {high[0]['direction']} — Score {high[0]['score']}**")

    return {"reply": "\n".join(lines), "data": signals}


def process_chat_message(message: str) -> Dict[str, Any]:
    cleaned = message.strip()
    if not cleaned:
        return {"reply": "Ask me anything — live prices, backtest results, win rates, or SMC concepts."}

    intent = _extract_parameters(cleaned)

    # Backtest and price requests: always handle locally with real engine data
    # Never let Gemini rewrite/summarize numerical results — it hallucinates numbers
    if _is_price_intent(cleaned):
        return _handle_price(intent)

    if _is_live_signal_intent(cleaned):
        return _handle_live_scan()

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
