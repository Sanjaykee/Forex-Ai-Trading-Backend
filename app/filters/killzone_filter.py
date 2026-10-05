from datetime import datetime, timezone

# ICT Killzones in UTC
KILLZONES = {
    "London Open":    (7,  9),   # 07:00 - 09:00 UTC  (12:30 - 14:30 IST)
    "New York Open":  (12, 14),  # 12:00 - 14:00 UTC  (17:30 - 19:30 IST)
    "London Close":   (15, 16),  # 15:00 - 16:00 UTC  (20:30 - 21:30 IST)
    "Asian Range":    (0,  3),   # 00:00 - 03:00 UTC  (05:30 - 08:30 IST)
}

HIGH_PROBABILITY = ["London Open", "New York Open"]

def get_active_killzone() -> dict:
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour

    for name, (start, end) in KILLZONES.items():
        if start <= hour < end:
            return {
                "active":            True,
                "name":              name,
                "high_probability":  name in HIGH_PROBABILITY,
            }

    return {"active": False, "name": None, "high_probability": False}

def is_killzone_active() -> bool:
    kz = get_active_killzone()
    return kz["active"] and kz["high_probability"]
