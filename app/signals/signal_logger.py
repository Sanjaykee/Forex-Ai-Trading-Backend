"""
Signal Logger — saves exact entry/SL/TP/lot at the moment a live signal fires.
Backtest reads from this log to replay trades with identical prices.

Log file: data/signal_log/YYYY-MM-DD.json
Format:
{
  "GBPUSD": {"entry": 1.34121, "sl": 1.34231, "tp": 1.33851,
             "direction": "SELL", "lot": 0.01, "time_ist": "19:21:35 IST",
             "session": "New York", "score": 13}
}
One file per date (UTC date). Multiple signals per pair per day are stored as a list.
"""

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent.parent / "data" / "signal_log"
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _log_path(date_str: str) -> Path:
    return _LOG_DIR / f"{date_str}.json"


def save_signal(symbol: str, direction: str, entry: float, sl: float,
                tp: float, lot: float, session: str, score: int) -> None:
    """Append a signal to today's log file. Called at the moment signal fires."""
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    ist_time = (now_utc + timedelta(hours=5, minutes=30)).strftime("%H:%M:%S IST")

    record = {
        "entry":     entry,
        "sl":        sl,
        "tp":        tp,
        "direction": direction,
        "lot":       lot,
        "time_ist":  ist_time,
        "session":   session,
        "score":     score,
    }

    path = _log_path(date_str)
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}

    # Store as list to support multiple signals per pair per day
    if symbol not in data:
        data[symbol] = []
    if isinstance(data[symbol], dict):          # migrate old single-record format
        data[symbol] = [data[symbol]]

    # Dedup: skip if identical entry/sl/tp/direction already logged today
    for existing in data[symbol]:
        if (existing.get("entry") == entry and
                existing.get("sl") == sl and
                existing.get("tp") == tp and
                existing.get("direction") == direction):
            return

    data[symbol].append(record)

    path.write_text(json.dumps(data, indent=2))


def load_signals(date_str: str) -> dict:
    """Return signal log for a date as {symbol: [records]}. Empty dict if none."""
    path = _log_path(date_str)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        # Normalise to list format
        for sym in data:
            if isinstance(data[sym], dict):
                data[sym] = [data[sym]]
        return data
    except Exception:
        return {}
