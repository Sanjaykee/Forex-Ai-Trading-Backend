from datetime import datetime, timezone

SESSIONS = {
    "London":   (7, 16),   # 07:00 - 16:00 UTC
    "NewYork":  (12, 21),  # 12:00 - 21:00 UTC
}

def get_active_session() -> str:
    hour = datetime.now(timezone.utc).hour
    active = []
    for name, (start, end) in SESSIONS.items():
        if start <= hour < end:
            active.append(name)
    if "London" in active and "NewYork" in active:
        return "London/NewYork"
    if active:
        return active[0]
    return "Asian"

def is_tradeable_session() -> bool:
    session = get_active_session()
    return session != "Asian"
