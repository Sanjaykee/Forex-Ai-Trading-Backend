"""
MetaApi Cloud Client
Provides direct cloud REST integration with MetaTrader 5 broker servers.
Allows executing real trades and fetching broker-exact quotes from Linux cloud servers (Render)
without needing a local Windows PC running.
"""

import requests
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetaApiClient:
    def __init__(self, token: str = None, account_id: str = None, region: str = None):
        self._token = token
        self._account_id = account_id
        self._region = region

    def _ensure_loaded(self):
        if not self._token or not self._account_id:
            try:
                from app.db.database import SessionLocal
                from app.db.models import User
                from app.db import crud
                db = SessionLocal()
                u = db.query(User).first()
                if u:
                    if not self._token and u.metaapi_token:
                        self._token = u.metaapi_token
                    if not self._account_id and u.metaapi_account_id:
                        self._account_id = u.metaapi_account_id
                    if not self._region:
                        self._region = crud.get_setting(db, u.id, "metaapi_region") or "new-york"
                db.close()
            except Exception:
                pass
        if not self._region:
            self._region = "backup-new-york"

    @property
    def token(self) -> str:
        self._ensure_loaded()
        return self._token or ""

    @property
    def account_id(self) -> str:
        self._ensure_loaded()
        return self._account_id or ""

    @property
    def region(self) -> str:
        self._ensure_loaded()
        return self._region or "backup-new-york"

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
            return {"connected": False, "error": f"{res.status_code}: {res.json().get('message','')[:120]}"}
        except Exception as e:
            logger.error(f"MetaApi get_account_information failed: {e}")
            return {}

    def get_live_price(self, symbol: str) -> dict:
        """Fetch exact real-time bid, ask, and spread from broker's MT5 feed"""
        if not self.is_configured:
            return {}
        url = f"{self.client_base_url}/symbols/{symbol}/current-price"
        try:
            res = requests.get(url, headers=self.headers, timeout=15)
            if res.status_code == 200:
                data = res.json()
                bid = data.get("bid", 0.0)
                ask = data.get("ask", 0.0)
                spread = round((ask - bid) * (100 if "JPY" in symbol else 10000), 1)
                return {"bid": bid, "ask": ask, "spread": spread}
            return {}
        except requests.exceptions.Timeout:
            # Primary region timed out — try alternate region once
            alt_region = "new-york" if "backup" in self.region else "backup-new-york"
            alt_url = f"https://mt-client-api-v1.{alt_region}.agiliumtrade.ai/users/current/accounts/{self.account_id.strip()}/symbols/{symbol}/current-price"
            try:
                res = requests.get(alt_url, headers=self.headers, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    bid = data.get("bid", 0.0)
                    ask = data.get("ask", 0.0)
                    spread = round((ask - bid) * (100 if "JPY" in symbol else 10000), 1)
                    logger.info(f"MetaApi get_live_price succeeded via fallback region {alt_region}")
                    return {"bid": bid, "ask": ask, "spread": spread}
            except Exception as e2:
                logger.error(f"MetaApi get_live_price fallback also failed: {e2}")
            return {}
        except Exception as e:
            logger.error(f"MetaApi get_live_price failed: {e}")
            return {}

    def get_candles(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch historical OHLC candles directly from broker.
        Paginates in batches of 1000 to cover the full requested count.
        """
        if not self.is_configured:
            return pd.DataFrame()

        tf_map = {
            "M1": "1m", "M3": "3m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d"
        }
        meta_tf = tf_map.get(timeframe.upper(), "15m")
        url = f"{self.market_base_url}/historical-market-data/symbols/{symbol}/timeframes/{meta_tf}/candles"

        def _parse_batch(candles: list) -> pd.DataFrame:
            if not candles:
                return pd.DataFrame()
            df = pd.DataFrame(candles)
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
            if "tickVolume" in df.columns:
                df.rename(columns={"tickVolume": "volume"}, inplace=True)
            elif "volume" not in df.columns:
                df["volume"] = 100
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    return pd.DataFrame()
            return df[["time", "open", "high", "low", "close", "volume"]]

        def _fetch_batch(params: dict) -> pd.DataFrame:
            """Single request with region fallback on timeout."""
            try:
                res = requests.get(url, headers=self.headers, params=params, timeout=20)
                if res.status_code == 200:
                    return _parse_batch(res.json())
                logger.error("MetaApi get_candles HTTP %d for %s %s: %s",
                             res.status_code, symbol, timeframe, res.text[:200])
                return pd.DataFrame()
            except requests.exceptions.Timeout:
                alt_region = "backup-new-york" if self.region == "new-york" else "new-york"
                alt_url = (
                    f"https://mt-market-data-client-api-v1.{alt_region}.agiliumtrade.ai"
                    f"/users/current/accounts/{self.account_id.strip()}"
                    f"/historical-market-data/symbols/{symbol}/timeframes/{meta_tf}/candles"
                )
                try:
                    res = requests.get(alt_url, headers=self.headers, params=params, timeout=20)
                    if res.status_code == 200:
                        logger.info("MetaApi get_candles fallback region %s OK for %s %s", alt_region, symbol, timeframe)
                        return _parse_batch(res.json())
                except Exception as e2:
                    logger.error("MetaApi get_candles fallback failed for %s %s: %s", symbol, timeframe, e2)
                return pd.DataFrame()
            except Exception as e:
                logger.error("MetaApi get_candles failed for %s %s: %s", symbol, timeframe, e)
                return pd.DataFrame()

        # Paginate: fetch up to `count` candles in batches of 1000
        BATCH = 1000
        all_frames = []
        remaining  = count
        start_time = None   # None = most recent batch first

        while remaining > 0:
            params = {"limit": min(remaining, BATCH)}
            if start_time:
                params["startTime"] = start_time

            batch = _fetch_batch(params)
            if batch.empty:
                break

            all_frames.append(batch)
            remaining -= len(batch)

            # If we got fewer than BATCH candles, broker has no more history
            if len(batch) < BATCH:
                break

            # Next batch: start before the oldest candle we have
            oldest = batch["time"].min()
            start_time = oldest.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            logger.info("MetaApi paginating %s %s: fetched %d so far, next startTime=%s",
                        symbol, timeframe, sum(len(f) for f in all_frames), start_time)

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        logger.info("MetaApi total candles fetched for %s %s: %d", symbol, timeframe, len(combined))
        return combined

    def place_order(self, symbol: str, direction: str, lot: float, entry: float = None, sl: float = None, tp: float = None, comment: str = "ForexAI") -> dict:
        """
        Places a LIMIT order at the OB entry price — matches backtest scenario exactly.
        Order waits at entry price until price returns to the OB zone, then fills.
        """
        if not self.is_configured:
            return {"success": False, "error": "MetaApi credentials not configured in backend"}

        digits = 3 if "JPY" in symbol else 5
        # Use limit order to match backtest entry — waits for OB retest
        if entry:
            action = "ORDER_TYPE_BUY_LIMIT" if direction.upper() == "BUY" else "ORDER_TYPE_SELL_LIMIT"
        else:
            action = "ORDER_TYPE_BUY" if direction.upper() == "BUY" else "ORDER_TYPE_SELL"

        url = f"{self.client_base_url}/trade"

        body = {
            "actionType": action,
            "symbol": symbol,
            "volume": float(lot),
            "comment": comment
        }
        if entry:
            body["openPrice"] = round(float(entry), digits)
        if sl:
            body["stopLoss"] = round(float(sl), digits)
        if tp:
            body["takeProfit"] = round(float(tp), digits)

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
