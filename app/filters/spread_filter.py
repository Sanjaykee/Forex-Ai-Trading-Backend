MAX_SPREAD_PIPS = 1.5

def is_spread_acceptable(spread_pips: float) -> bool:
    return spread_pips <= MAX_SPREAD_PIPS
