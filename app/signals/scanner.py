from app.mt5.source_router import get_fetcher
from app.signals.signal_engine import analyze_pair

def scan_all_pairs(risk_usd: float = 1.0, rr_ratio: float = 2.0, data_source: str = "yfinance") -> list:
    _, _, _, get_all_pairs = get_fetcher(data_source)
    results = []
    for symbol in get_all_pairs():
        result = analyze_pair(symbol, risk_usd, rr_ratio, data_source)
        results.append(result)
    return results

def get_active_signals(risk_usd: float = 1.0, rr_ratio: float = 2.0, data_source: str = "yfinance") -> list:
    all_results = scan_all_pairs(risk_usd, rr_ratio, data_source)
    return [r for r in all_results if r["direction"] in ("BUY", "SELL")]
