"""
MetaApi Cloud Client
Provides direct cloud REST integration with MetaTrader 5 broker servers.
Allows executing real trades and fetching broker-exact quotes from Linux cloud servers (Render)
without needing a local Windows PC running.
"""

import requests
import pandas as pd
from datetime import datetime
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class MetaApiClient:
    def __init__(self, token: str = None, account_id: str = None, region: str = None):
        self.token = token or settings.METAAPI_TOKEN
        self.account_id = account_id or settings.METAAPI_ACCOUNT_ID
        self.region = region or getattr(settings, "METAAPI_REGION", "new-york")

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.token.strip() and self.account_id and self.account_id.strip())

    @property
    def headers(self) -> dict:
        return {
            "auth-token": self.token.strip(),
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    @property
    def client_base_url(self) -> str:
        return f"https://mt-client-api-v1.{self.region}.agiliumtrade.ai/users/current/accounts/{self.account_id.strip()}"

    @property
    def market_base_url(self) -> str:
        return f"https://mt-market-data-client-api-v1.{self.region}.agiliumtrade.ai/users/current/accounts/{self.account_id.strip()}"

    def get_account_information(self) -> dict:
        """Fetch live account balance, equity, margin, leverage, and currency from broker"""
        if not self.is_configured:
            return {}
        url = f"{self.client_base_url}/accountInformation"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                return {
                    "balance": data.get("balance", 0.0),
                    "equity": data.get("equity", 0.0),
                    "margin": data.get("margin", 0.0),
                    "freeMargin": data.get("freeMargin", 0.0),
                    "currency": data.get("currency", "USD"),
                    "server": data.get("server", ""),
                    "login": data.get("login", ""),
                    "connected": True
                }
            logger.warning(f"MetaApi accountInformation returned {res.status_code}: {res.text}")
            return {}
        except Exception as e:
            logger.error(f"MetaApi get_account_information failed: {e}")
            return {}

    def get_live_price(self, symbol: str) -> dict:
        """Fetch exact real-time bid, ask, and spread from broker's MT5 feed"""
        if not self.is_configured:
            return {}
        url = f"{self.market_base_url}/symbols/{symbol}/current-price"
        try:
            res = requests.get(url, headers=self.headers, timeout=8)
            if res.status_code == 200:
                data = res.json()
                bid = data.get("bid", 0.0)
                ask = data.get("ask", 0.0)
                spread = round((ask - bid) * (100 if "JPY" in symbol else 10000), 1)
                return {"bid": bid, "ask": ask, "spread": spread}
            return {}
        except Exception as e:
            logger.error(f"MetaApi get_live_price failed: {e}")
            return {}

    def get_candles(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch historical OHLC candles directly from broker"""
        if not self.is_configured:
            return pd.DataFrame()

        # Map MT5 timeframes to MetaApi format (e.g., "15m", "1h", "4h", "1d")
        tf_map = {
            "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d"
        }
        meta_tf = tf_map.get(timeframe.upper(), "15m")
        url = f"{self.market_base_url}/historical-market-data/symbols/{symbol}/timeframes/{meta_tf}/candles"
        params = {"limit": min(count, 1000)}

        try:
            res = requests.get(url, headers=self.headers, params=params, timeout=12)
            if res.status_code == 200:
                candles = res.json()
                if not candles:
                    return pd.DataFrame()
                df = pd.DataFrame(candles)
                # MetaApi returns time, open, high, low, close, tickVolume
                if "time" in df.columns:
                    df["time"] = pd.to_datetime(df["time"])
                if "tickVolume" in df.columns:
                    df.rename(columns={"tickVolume": "volume"}, inplace=True)
                elif "volume" not in df.columns:
                    df["volume"] = 100
                return df[["time", "open", "high", "low", "close", "volume"]]
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"MetaApi get_candles failed: {e}")
            return pd.DataFrame()

    def place_order(self, symbol: str, direction: str, lot: float, sl: float = None, tp: float = None, comment: str = "ForexAI") -> dict:
        """
        Executes a real order directly onto the broker's MT5 server.
        The trade appears immediately inside your MT5 Mobile App!
        """
        if not self.is_configured:
            return {"success": False, "error": "MetaApi credentials not configured in backend"}

        action = "ORDER_TYPE_BUY" if direction.upper() == "BUY" else "ORDER_TYPE_SELL"
        url = f"{self.client_base_url}/trade"

        body = {
            "actionType": action,
            "symbol": symbol,
            "volume": float(lot),
            "comment": comment
        }
        if sl:
            body["stopLoss"] = round(float(sl), 5 if "JPY" not in symbol else 3)
        if tp:
            body["takeProfit"] = round(float(tp), 5 if "JPY" not in symbol else 3)

        try:
            res = requests.post(url, headers=self.headers, json=body, timeout=15)
            data = res.json()
            if res.status_code in (200, 201) and data.get("numericCode") in (10008, 10009, None):
                ticket = data.get("orderId") or data.get("positionId") or "METAAPI_LIVE"
                logger.info(f"MetaApi order executed successfully: Ticket {ticket} for {symbol} {direction} {lot} lots")
                return {"success": True, "ticket": ticket, "data": data}
            else:
                err_msg = data.get("message") or data.get("stringCode") or str(data)
                logger.error(f"MetaApi trade execution rejected: {err_msg}")
                return {"success": False, "error": err_msg}
        except Exception as e:
            logger.error(f"MetaApi place_order network exception: {e}")
            return {"success": False, "error": str(e)}

    def close_order(self, position_id: str) -> dict:
        """Close an open position on MT5"""
        if not self.is_configured:
            return {"success": False, "error": "MetaApi not configured"}

        url = f"{self.client_base_url}/trade"
        body = {
            "actionType": "POSITION_CLOSE_ID",
            "positionId": str(position_id)
        }
        try:
            res = requests.post(url, headers=self.headers, json=body, timeout=15)
            if res.status_code in (200, 201):
                return {"success": True, "ticket": position_id}
            return {"success": False, "error": res.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

metaapi_client = MetaApiClient()
