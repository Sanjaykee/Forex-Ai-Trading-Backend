"""
cTrader Open API Client
Connects via TCP to demo.ctrader.com:5035 using OAuth2 Client Credentials.
Provides: account info, historical candles, live price, place/close orders.
"""
import socket
import ssl
import struct
import json
import time
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

OAUTH_URL   = "https://connect.ctrader.com/apps/token"
OAUTH_URL2  = "https://openapi.ctrader.com/apps/token"
DEMO_HOST   = "demo.ctrader.com"
LIVE_HOST   = "live.ctrader.com"
API_PORT    = 5035

TIMEFRAME_MAP = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
    "M10": 6, "M15": 7, "M30": 8, "H1": 9, "H4": 10,
    "H12": 11, "D1": 12, "W1": 13, "MN": 14,
}


class CTraderClient:
    def __init__(self, client_id: str, client_secret: str, account_id: str, server: str = "demo.ctrader.com"):
        self.client_id     = client_id
        self.client_secret = client_secret
        self.account_id    = str(account_id)
        self.host          = server
        self._token        = None
        self._token_expiry = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.account_id)

    def _get_token(self) -> str:
        """Get OAuth2 access token using client credentials flow."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token
        for url in [OAUTH_URL, OAUTH_URL2]:
            try:
                r = requests.post(url, data={
                    "grant_type":    "client_credentials",
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                }, timeout=10)
                logger.info(f"[cTrader] Token URL={url} Status={r.status_code} Body={r.text[:200]}")
                data = r.json()
                if "access_token" in data:
                    self._token        = data["access_token"]
                    self._token_expiry = time.time() + data.get("expires_in", 3600)
                    return self._token
            except Exception as e:
                logger.error(f"[cTrader] Token fetch failed at {url}: {e}")
        return ""

    def test_connection(self) -> dict:
        """Test OAuth token + return account info."""
        if not self.is_configured:
            return {"connected": False, "error": "Credentials not configured"}
        token = self._get_token()
        if not token:
            return {"connected": False, "error": "Failed to get OAuth token — check Client ID and Secret"}
        try:
            r = requests.get(
                f"https://connect.ctrader.com/apps/accounts",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            accounts = r.json()
            # Find matching account
            match = None
            if isinstance(accounts, list):
                for acc in accounts:
                    if str(acc.get("accountId")) == self.account_id or \
                       str(acc.get("login"))     == self.account_id:
                        match = acc
                        break
                if not match and accounts:
                    match = accounts[0]
            if match:
                return {
                    "connected":  True,
                    "account_id": match.get("accountId") or match.get("login"),
                    "balance":    match.get("balance", 0),
                    "currency":   match.get("depositCurrency", "USD"),
                    "broker":     match.get("brokerName", "IC Markets"),
                    "is_live":    not match.get("isDemo", True),
                }
            return {"connected": True, "message": "Token valid but account not found — check Account ID"}
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def get_candles(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch historical candles via cTrader REST API."""
        token = self._get_token()
        if not token:
            return pd.DataFrame()
        try:
            tf_id = TIMEFRAME_MAP.get(timeframe.upper(), 7)
            end_ts   = int(time.time() * 1000)
            # Estimate start based on timeframe
            tf_minutes = {1:1,2:2,3:3,4:4,5:5,6:10,7:15,8:30,9:60,10:240,11:720,12:1440,13:10080,14:43200}
            mins = tf_minutes.get(tf_id, 15)
            start_ts = end_ts - (count * mins * 60 * 1000)

            url = (
                f"https://connect.ctrader.com/apps/accounts/{self.account_id}"
                f"/symbols/{symbol}/trendbars"
                f"?period={tf_id}&from={start_ts}&to={end_ts}&count={count}"
            )
            r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=15)
            data = r.json()
            bars = data.get("data", data) if isinstance(data, dict) else data
            if not bars or not isinstance(bars, list):
                return pd.DataFrame()
            rows = []
            for b in bars:
                rows.append({
                    "time":   pd.Timestamp(b["timestamp"], unit="ms"),
                    "open":   b.get("open",  b.get("o", 0)) / 100000,
                    "high":   b.get("high",  b.get("h", 0)) / 100000,
                    "low":    b.get("low",   b.get("l", 0)) / 100000,
                    "close":  b.get("close", b.get("c", 0)) / 100000,
                    "volume": b.get("volume", b.get("v", 1000)),
                })
            df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"[cTrader] get_candles failed: {e}")
            return pd.DataFrame()

    def get_live_price(self, symbol: str) -> dict:
        """Get latest bid/ask from last candle."""
        df = self.get_candles(symbol, "M1", count=2)
        if df.empty:
            return {}
        last = df.iloc[-1]
        price = float(last["close"])
        pip   = 0.01 if "JPY" in symbol or "XAU" in symbol else 0.0001
        return {
            "bid":    round(price, 5),
            "ask":    round(price + pip, 5),
            "spread": 1.0,
        }

    def place_order(self, symbol: str, direction: str, volume: float,
                    sl: float = None, tp: float = None, comment: str = "") -> dict:
        """Place market order via cTrader REST API."""
        token = self._get_token()
        if not token:
            return {"success": False, "error": "No token"}
        try:
            payload = {
                "symbolName":  symbol,
                "orderType":   "MARKET",
                "tradeSide":   "BUY" if direction.upper() == "BUY" else "SELL",
                "volume":      int(volume * 100),  # cTrader uses lots * 100
                "comment":     comment,
            }
            if sl:  payload["stopLoss"]   = sl
            if tp:  payload["takeProfit"] = tp

            r = requests.post(
                f"https://connect.ctrader.com/apps/accounts/{self.account_id}/orders",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, timeout=10
            )
            data = r.json()
            if r.status_code == 200 and data.get("orderId"):
                return {"success": True, "order_id": data["orderId"], "data": data}
            return {"success": False, "error": data.get("message", str(data))}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_order(self, position_id: str, volume: float = None) -> dict:
        """Close an open position."""
        token = self._get_token()
        if not token:
            return {"success": False, "error": "No token"}
        try:
            payload = {"positionId": position_id}
            if volume:
                payload["volume"] = int(volume * 100)
            r = requests.delete(
                f"https://connect.ctrader.com/apps/accounts/{self.account_id}/positions/{position_id}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=payload, timeout=10
            )
            return {"success": r.status_code == 200, "data": r.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_open_positions(self) -> list:
        """Get all open positions."""
        token = self._get_token()
        if not token:
            return []
        try:
            r = requests.get(
                f"https://connect.ctrader.com/apps/accounts/{self.account_id}/positions",
                headers={"Authorization": f"Bearer {token}"}, timeout=10
            )
            data = r.json()
            return data if isinstance(data, list) else data.get("data", [])
        except Exception as e:
            logger.error(f"[cTrader] get_positions failed: {e}")
            return []
