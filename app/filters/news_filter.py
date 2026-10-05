import requests
from datetime import datetime, timedelta, timezone

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
HIGH_IMPACT_CURRENCIES = ["USD", "EUR", "GBP", "JPY"]
BUFFER_MINUTES = 30

def get_news_events() -> list:
    try:
        res = requests.get(CALENDAR_URL, timeout=5)
        return res.json()
    except Exception:
        return []

def is_news_safe() -> bool:
    now    = datetime.now(timezone.utc)
    events = get_news_events()
    for event in events:
        if event.get("impact") != "High":
            continue
        if event.get("currency") not in HIGH_IMPACT_CURRENCIES:
            continue
        try:
            event_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            diff = abs((event_time - now).total_seconds() / 60)
            if diff <= BUFFER_MINUTES:
                return False
        except Exception:
            continue
    return True
