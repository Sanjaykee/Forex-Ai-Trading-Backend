try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    mt5 = None
    HAS_MT5 = False

MT5_PATH = "C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe"

def connect(login: int, password: str, server: str) -> bool:
    if not HAS_MT5 or mt5 is None:
        return False
    try:
        mt5.shutdown()
        if not mt5.initialize(path=MT5_PATH, login=int(login), password=password, server=server):
            return False
        return mt5.account_info() is not None
    except Exception:
        return False

def disconnect():
    if HAS_MT5 and mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass

def is_connected() -> bool:
    if not HAS_MT5 or mt5 is None:
        return False
    try:
        return mt5.terminal_info() is not None
    except Exception:
        return False

def get_account_info() -> dict:
    if not HAS_MT5 or mt5 is None:
        return {}
    try:
        info = mt5.account_info()
        if not info:
            return {}
        return {
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "currency": info.currency,
            "server":   info.server,
            "login":    info.login,
        }
    except Exception:
        return {}
