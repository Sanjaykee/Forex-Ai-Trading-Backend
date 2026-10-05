import MetaTrader5 as mt5

MT5_PATH = "C:\\Program Files\\MetaTrader 5 IC Markets Global\\terminal64.exe"

def connect(login: int, password: str, server: str) -> bool:
    mt5.shutdown()
    if not mt5.initialize(path=MT5_PATH, login=int(login), password=password, server=server):
        return False
    return mt5.account_info() is not None

def disconnect():
    mt5.shutdown()

def is_connected() -> bool:
    return mt5.terminal_info() is not None

def get_account_info() -> dict:
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
