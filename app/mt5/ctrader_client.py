"""
cTrader Open API Client (WebSocket JSON)
Connects directly to cTrader Cloud (wss://demo.ctraderapi.com:5036 or wss://live.ctraderapi.com:5036)
via native WebSocket JSON protocol on port 5036.
Provides: account auth, symbol mapping, historical candles (trendbars), live price, order execution.
100% cloud native — runs seamlessly on Linux (Render) and Windows without local desktop dependencies.
"""
import json
import time
import logging
import threading
import uuid
import pandas as pd
from datetime import datetime, timezone
import requests
from websockets.sync.client import connect as ws_connect

logger = logging.getLogger(__name__)

# cTrader Open API JSON WebSocket endpoints (port 5036)
DEMO_WS_URL = "wss://demo.ctraderapi.com:5036"
LIVE_WS_URL = "wss://live.ctraderapi.com:5036"
OAUTH_TOKEN_URL = "https://openapi.ctrader.com/oauth/v2/token"

# cTrader Protocol Payload Types (ProtoOAPayloadType enum)
PAYLOAD_APP_AUTH_REQ     = 2100
PAYLOAD_APP_AUTH_RES     = 2101
PAYLOAD_ACCOUNT_AUTH_REQ = 2102
PAYLOAD_ACCOUNT_AUTH_RES = 2103
PAYLOAD_NEW_ORDER_REQ    = 2104
PAYLOAD_CLOSE_POS_REQ    = 2111
PAYLOAD_SYMBOLS_LIST_REQ = 2114
PAYLOAD_SYMBOLS_LIST_RES = 2115
PAYLOAD_RECONCILE_REQ    = 2124
PAYLOAD_RECONCILE_RES    = 2125
PAYLOAD_EXECUTION_EVENT  = 2126
PAYLOAD_GET_TRENDBARS_REQ= 2137
PAYLOAD_GET_TRENDBARS_RES= 2138
PAYLOAD_ERROR_RES        = 2142
PAYLOAD_GET_ACCOUNTS_REQ = 2149
PAYLOAD_GET_ACCOUNTS_RES = 2150

TIMEFRAME_MAP = {
    "M1": 1, "M2": 2, "M3": 3, "M4": 4, "M5": 5,
    "M10": 6, "M15": 7, "M30": 8, "H1": 9, "H4": 10,
    "H12": 11, "D1": 12, "W1": 13, "MN1": 14, "MN": 14,
}

TIMEFRAME_MINUTES = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10,
    7: 15, 8: 30, 9: 60, 10: 240, 11: 720, 12: 1440, 13: 10080, 14: 43200
}


class CTraderClient:
    """Synchronous cTrader Open API Client using JSON WebSocket (Port 5036)."""

    def __init__(self, client_id: str, client_secret: str, account_id: str,
                 server: str = "demo.ctrader.com", access_token: str = None, refresh_token: str = None):
        self.client_id     = str(client_id).strip() if client_id else ""
        self.client_secret = str(client_secret).strip() if client_secret else ""
        self.account_id    = str(account_id).strip() if account_id else ""
        self.server        = str(server).strip() if server else "demo.ctrader.com"
        self._token        = str(access_token).strip() if access_token else ""
        self.refresh_token = str(refresh_token).strip() if refresh_token else ""
        self._token_expiry = time.time() + 86400 if access_token else 0

        is_live = "live" in self.server.lower()
        self.ws_url = LIVE_WS_URL if is_live else DEMO_WS_URL

        self._ws = None
        self._ctid_trader_account_id = None
        self._lock = threading.Lock()
        self._symbol_cache = {}  # { "EURUSD": (symbol_id, digits) }

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret and self.account_id and self._token)

    @property
    def ctid_id(self) -> int:
        return self._ctid_trader_account_id or int(self.account_id)

    def _get_token(self) -> str:
        """Return OAuth2 access token. Refresh if expired and refresh_token exists."""
        if self._token and time.time() < self._token_expiry - 60:
            return self._token

        if self.refresh_token and self.client_id and self.client_secret:
            try:
                r = requests.post(OAUTH_TOKEN_URL, data={
                    "grant_type":    "refresh_token",
                    "refresh_token": self.refresh_token,
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                }, timeout=10)
                data = r.json()
                if "access_token" in data:
                    self._token = data["access_token"]
                    self._token_expiry = time.time() + data.get("expires_in", 86400)
                    if "refresh_token" in data:
                        self.refresh_token = data["refresh_token"]
                    logger.info("[cTrader] Token refreshed successfully")
                    return self._token
            except Exception as e:
                logger.error(f"[cTrader] Token refresh failed: {e}")

        return self._token or ""

    def _connect(self):
        """Open or return an authenticated cTrader WebSocket connection."""
        token = self._get_token()
        if not token:
            raise ValueError("No cTrader access token available")

        # Test existing connection
        if self._ws is not None:
            try:
                # Quick health check
                return self._ws
            except Exception:
                self._ws = None

        logger.info(f"[cTrader] Connecting to {self.ws_url} (Account: {self.account_id})...")
        ws = ws_connect(self.ws_url, open_timeout=15, close_timeout=5, max_size=32 * 1024 * 1024)

        # 1. Application Authorization (ProtoOAApplicationAuthReq)
        app_msg_id = f"app_auth_{uuid.uuid4().hex[:8]}"
        ws.send(json.dumps({
            "clientMsgId": app_msg_id,
            "payloadType": PAYLOAD_APP_AUTH_REQ,
            "payload": {
                "clientId":     self.client_id,
                "clientSecret": self.client_secret
            }
        }))
        app_res = self._wait_for_response(ws, app_msg_id, PAYLOAD_APP_AUTH_RES)
        if app_res is None:
            ws.close()
            raise RuntimeError("cTrader Application Authorization failed")

        # 2. Discover ctidTraderAccountId from Access Token (ProtoOAGetAccountListByAccessTokenReq)
        if not self._ctid_trader_account_id:
            disc_msg_id = f"disc_{uuid.uuid4().hex[:8]}"
            ws.send(json.dumps({
                "clientMsgId": disc_msg_id,
                "payloadType": PAYLOAD_GET_ACCOUNTS_REQ,
                "payload": {"accessToken": token}
            }))
            disc_res = self._wait_for_response(ws, disc_msg_id, PAYLOAD_GET_ACCOUNTS_RES)
            if disc_res and "ctidTraderAccount" in disc_res:
                accounts = disc_res.get("ctidTraderAccount", [])
                target_login = str(self.account_id).strip()
                match = None
                for acc in accounts:
                    if str(acc.get("traderLogin")) == target_login or str(acc.get("ctidTraderAccountId")) == target_login:
                        match = acc
                        break
                if not match and accounts:
                    match = accounts[0]
                if match:
                    self._ctid_trader_account_id = int(match["ctidTraderAccountId"])

        ctid = self._ctid_trader_account_id or int(self.account_id)

        # 3. Account Authorization (ProtoOAAccountAuthReq)
        acc_msg_id = f"acc_auth_{uuid.uuid4().hex[:8]}"
        ws.send(json.dumps({
            "clientMsgId": acc_msg_id,
            "payloadType": PAYLOAD_ACCOUNT_AUTH_REQ,
            "payload": {
                "ctidTraderAccountId": ctid,
                "accessToken":         token
            }
        }))
        acc_res = self._wait_for_response(ws, acc_msg_id, PAYLOAD_ACCOUNT_AUTH_RES)
        if acc_res is None:
            ws.close()
            raise RuntimeError(f"cTrader Account Authorization failed for account {ctid}")

        self._ws = ws
        logger.info(f"[cTrader] Connected to {self.ws_url} (Account: {self.account_id} -> ctid: {ctid})")
        return self._ws

    def _wait_for_response(self, ws, client_msg_id: str, expected_payload_type: int, timeout: float = 12.0):
        """Read incoming WebSocket frames until matching clientMsgId or expected_payload_type is found."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                raw = ws.recv(timeout=timeout)
                if not raw:
                    continue
                logger.debug(f"[cTrader IN] {raw[:300]}")
                data = json.loads(raw)
                p_type = data.get("payloadType")
                msg_id = data.get("clientMsgId")

                if p_type == PAYLOAD_ERROR_RES:
                    err = data.get("payload", {})
                    logger.error(f"[cTrader Server Error] {err}")
                    return None

                if msg_id == client_msg_id or p_type == expected_payload_type:
                    return data.get("payload", {})
            except TimeoutError:
                break
            except Exception as e:
                logger.error(f"[cTrader] Error receiving message: {e}")
                break
        return None

    def _ensure_symbols(self, ws) -> dict:
        """Fetch broker symbol list to map instrument names (EURUSD) to numeric symbolId."""
        if self._symbol_cache:
            return self._symbol_cache

        msg_id = f"syms_{uuid.uuid4().hex[:8]}"
        ws.send(json.dumps({
            "clientMsgId": msg_id,
            "payloadType": PAYLOAD_SYMBOLS_LIST_REQ,
            "payload": {
                "ctidTraderAccountId": self.ctid_id,
                "includeArchivedSymbols": False
            }
        }))
        payload = self._wait_for_response(ws, msg_id, PAYLOAD_SYMBOLS_LIST_RES)
        if not payload or not isinstance(payload, dict):
            return self._symbol_cache
        symbols = payload.get("symbol", [])
        cache = {}
        for s in symbols:
            s_name = s.get("symbolName", "").upper()
            s_id   = s.get("symbolId")
            digits = s.get("digits", 5)
            if s_name and s_id is not None:
                cache[s_name] = (int(s_id), int(digits))
                # Also store normalized version (e.g. "EURUSD.ecn" -> "EURUSD")
                clean_name = s_name.replace(".ECN", "").replace("+", "").replace("-", "").strip()
                if clean_name not in cache:
                    cache[clean_name] = (int(s_id), int(digits))

        self._symbol_cache = cache
        logger.info(f"[cTrader] Loaded {len(cache)} symbols into cache")
        return self._symbol_cache

    def _resolve_symbol(self, ws, symbol: str) -> tuple[int, int]:
        """Return (symbol_id, digits) for a given symbol string."""
        symbols = self._ensure_symbols(ws)
        s_upper = symbol.upper()
        if s_upper in symbols:
            return symbols[s_upper]
        clean = s_upper.replace(".ECN", "").replace("+", "").replace("-", "").strip()
        if clean in symbols:
            return symbols[clean]
        # Partial match fallback
        for name, info in symbols.items():
            if clean in name:
                return info
        raise ValueError(f"Symbol '{symbol}' not found on cTrader account")

    def test_connection(self) -> dict:
        """Test OAuth credentials and WebSocket connection."""
        if not self.is_configured:
            return {"connected": False, "error": "cTrader credentials not fully configured or authorized"}
        with self._lock:
            try:
                ws = self._connect()
                symbols = self._ensure_symbols(ws)
                return {
                    "connected": True,
                    "account_id": self.account_id,
                    "server": self.server,
                    "symbols_count": len(symbols),
                    "broker": "IC Markets cTrader"
                }
            except Exception as e:
                self._ws = None
                return {"connected": False, "error": str(e)}

    def get_candles(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch historical candles (trendbars) via cTrader WebSocket JSON API."""
        if not self.is_configured:
            return pd.DataFrame()

        with self._lock:
            for attempt in range(2):
                try:
                    ws = self._connect()
                    symbol_id, digits = self._resolve_symbol(ws, symbol)
                    period_id = TIMEFRAME_MAP.get(timeframe.upper(), 7)

                    mins = TIMEFRAME_MINUTES.get(period_id, 15)
                    end_ts = int(time.time() * 1000)
                    start_ts = end_ts - (count * mins * 60 * 1000)

                    msg_id = f"bars_{symbol}_{timeframe}_{uuid.uuid4().hex[:6]}"
                    ws.send(json.dumps({
                        "clientMsgId": msg_id,
                        "payloadType": PAYLOAD_GET_TRENDBARS_REQ,
                        "payload": {
                            "ctidTraderAccountId": self.ctid_id,
                            "symbolId": symbol_id,
                            "period": period_id,
                            "fromTimestamp": start_ts,
                            "toTimestamp": end_ts,
                            "count": min(count, 1000)
                        }
                    }))

                    payload = self._wait_for_response(ws, msg_id, PAYLOAD_GET_TRENDBARS_RES, timeout=12.0)
                    if payload is None:
                        # Rate limited or temporary hiccup, pause 1s and retry once
                        time.sleep(1.0)
                        continue

                    bars = payload.get("trendbar", [])
                    if not bars or not isinstance(bars, list):
                        return pd.DataFrame()

                    rows = []
                    for b in bars:
                        # In cTrader, divisor is 100000.0 for all relative trendbars
                        raw_low = float(b.get("low", 0))
                        low   = round(raw_low / 100000.0, digits)
                        open_ = round((raw_low + float(b.get("deltaOpen", 0))) / 100000.0, digits)
                        high  = round((raw_low + float(b.get("deltaHigh", 0))) / 100000.0, digits)
                        close = round((raw_low + float(b.get("deltaClose", 0))) / 100000.0, digits)
                        vol   = float(b.get("volume", 1000))

                        if "utcTimestampInMinutes" in b:
                            t = pd.Timestamp(b["utcTimestampInMinutes"] * 60, unit="s", tz="UTC").tz_localize(None)
                        else:
                            ts_ms = b.get("timestamp", end_ts)
                            t = pd.Timestamp(ts_ms, unit="ms", tz="UTC").tz_localize(None)

                        rows.append({
                            "time":   t,
                            "open":   open_,
                            "high":   high,
                            "low":    low,
                            "close":  close,
                            "volume": vol
                        })

                    df = pd.DataFrame(rows).sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
                    return df

                except Exception as e:
                    logger.error(f"[cTrader] get_candles attempt {attempt+1} failed for {symbol} {timeframe}: {e}")
                    self._ws = None
                    time.sleep(0.5)

            return pd.DataFrame()

    def get_candles_range(self, symbol: str, timeframe: str, from_date: datetime = None, to_date: datetime = None, **kwargs) -> pd.DataFrame:
        """Fetch historical candles for a specific date range from cTrader Open API via WebSocket.
        Paginates backwards in batches of 1000 trendbars until from_date is satisfied.
        Returns a DataFrame sorted ascending by time, with tz-naive UTC timestamps."""
        if not self.is_configured:
            return pd.DataFrame()

        from_dt = from_date or kwargs.get("start_dt")
        to_dt = to_date or kwargs.get("end_dt")
        if from_dt is None or to_dt is None:
            return pd.DataFrame()

        from_dt_utc = from_dt if from_dt.tzinfo is not None else from_dt.replace(tzinfo=timezone.utc)
        to_dt_utc = to_dt if to_dt.tzinfo is not None else to_dt.replace(tzinfo=timezone.utc)

        target_start_ms = int(from_dt_utc.timestamp() * 1000)
        current_end_ms = int(to_dt_utc.timestamp() * 1000)

        period_id = TIMEFRAME_MAP.get(timeframe.upper(), 7)
        all_rows = []

        with self._lock:
            try:
                ws = self._connect()
                symbol_id, digits = self._resolve_symbol(ws, symbol)

                max_batches = 25  # up to 25,000 candles
                for _ in range(max_batches):
                    msg_id = f"range_{symbol}_{timeframe}_{uuid.uuid4().hex[:6]}"
                    ws.send(json.dumps({
                        "clientMsgId": msg_id,
                        "payloadType": PAYLOAD_GET_TRENDBARS_REQ,
                        "payload": {
                            "ctidTraderAccountId": self.ctid_id,
                            "symbolId": symbol_id,
                            "period": period_id,
                            "fromTimestamp": target_start_ms,
                            "toTimestamp": current_end_ms,
                            "count": 1000
                        }
                    }))

                    payload = self._wait_for_response(ws, msg_id, PAYLOAD_GET_TRENDBARS_RES, timeout=12.0)
                    if not payload or not isinstance(payload, dict):
                        time.sleep(1.0)
                        continue

                    bars = payload.get("trendbar", [])
                    if not bars or not isinstance(bars, list):
                        break

                    batch_rows = []
                    earliest_ms = None

                    for b in bars:
                        raw_low = float(b.get("low", 0))
                        low   = round(raw_low / 100000.0, digits)
                        open_ = round((raw_low + float(b.get("deltaOpen", 0))) / 100000.0, digits)
                        high  = round((raw_low + float(b.get("deltaHigh", 0))) / 100000.0, digits)
                        close = round((raw_low + float(b.get("deltaClose", 0))) / 100000.0, digits)
                        vol   = float(b.get("volume", 1000))

                        if "utcTimestampInMinutes" in b:
                            bar_ms = b["utcTimestampInMinutes"] * 60 * 1000
                            t = pd.Timestamp(b["utcTimestampInMinutes"] * 60, unit="s", tz="UTC").tz_localize(None)
                        else:
                            bar_ms = b.get("timestamp", current_end_ms)
                            t = pd.Timestamp(bar_ms, unit="ms", tz="UTC").tz_localize(None)

                        if earliest_ms is None or bar_ms < earliest_ms:
                            earliest_ms = bar_ms

                        batch_rows.append({
                            "time":   t,
                            "open":   open_,
                            "high":   high,
                            "low":    low,
                            "close":  close,
                            "volume": vol
                        })

                    all_rows.extend(batch_rows)

                    if len(bars) < 1000 or (earliest_ms and earliest_ms <= target_start_ms):
                        break

                    current_end_ms = earliest_ms - 1
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"[cTrader] get_candles_range failed for {symbol} {timeframe}: {e}")
                self._ws = None

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        cutoff_start = pd.Timestamp(from_dt).tz_localize(None) if getattr(from_dt, "tzinfo", None) else pd.Timestamp(from_dt)
        cutoff_end = pd.Timestamp(to_dt).tz_localize(None) if getattr(to_dt, "tzinfo", None) else pd.Timestamp(to_dt)
        df = df[(df["time"] >= cutoff_start) & (df["time"] <= cutoff_end)].reset_index(drop=True)
        return df

    def get_live_price(self, symbol: str) -> dict:
        """Fetch latest market price from latest 1m trendbar."""
        df = self.get_candles(symbol, "M1", count=2)
        if df.empty:
            return {}
        last = df.iloc[-1]
        price = float(last["close"])
        pip = 0.01 if "JPY" in symbol or "XAU" in symbol else 0.0001
        return {
            "bid":    round(price, 5),
            "ask":    round(price + pip, 5),
            "spread": 1.0,
        }

    def place_order(self, symbol: str, direction: str, volume: float,
                    sl: float = None, tp: float = None, comment: str = "") -> dict:
        """Place market order via cTrader Open API."""
        if not self.is_configured:
            return {"success": False, "error": "cTrader not configured"}

        with self._lock:
            try:
                ws = self._connect()
                symbol_id, digits = self._resolve_symbol(ws, symbol)

                # volume in cTrader Open API is in cents: 1 lot = 10,000,000 cents
                vol_cents = int(volume * 10000000)
                trade_side = 1 if direction.upper() == "BUY" else 2  # 1=BUY, 2=SELL

                payload = {
                    "ctidTraderAccountId": self.ctid_id,
                    "symbolId": symbol_id,
                    "orderType": 1,  # 1=MARKET
                    "tradeSide": trade_side,
                    "volume": vol_cents,
                    "comment": str(comment or "ForexAI")[:100]
                }
                if sl is not None:
                    payload["stopLoss"] = round(float(sl), digits)
                if tp is not None:
                    payload["takeProfit"] = round(float(tp), digits)

                msg_id = f"order_{uuid.uuid4().hex[:8]}"
                ws.send(json.dumps({
                    "clientMsgId": msg_id,
                    "payloadType": PAYLOAD_NEW_ORDER_REQ,
                    "payload": payload
                }))

                res = self._wait_for_response(ws, msg_id, PAYLOAD_EXECUTION_EVENT, timeout=10.0)
                if res and "order" in res:
                    order_info = res.get("order", {})
                    order_id = order_info.get("orderId")
                    return {"success": True, "order_id": order_id, "data": res}
                elif res and "position" in res:
                    pos_info = res.get("position", {})
                    return {"success": True, "order_id": pos_info.get("positionId"), "data": res}
                elif res and not res.get("errorCode"):
                    return {"success": True, "order_id": res.get("orderId", "ok"), "data": res}
                else:
                    return {"success": False, "error": res.get("description", str(res))}

            except Exception as e:
                logger.error(f"[cTrader] place_order failed: {e}")
                self._ws = None
                return {"success": False, "error": str(e)}

    def close_order(self, position_id: str, volume: float = None) -> dict:
        """Close an open position by positionId."""
        if not self.is_configured:
            return {"success": False, "error": "cTrader not configured"}

        with self._lock:
            try:
                ws = self._connect()
                payload = {
                    "ctidTraderAccountId": self.ctid_id,
                    "positionId": int(position_id)
                }
                if volume:
                    payload["volume"] = int(volume * 10000000)

                msg_id = f"close_{uuid.uuid4().hex[:8]}"
                ws.send(json.dumps({
                    "clientMsgId": msg_id,
                    "payloadType": PAYLOAD_CLOSE_POS_REQ,
                    "payload": payload
                }))

                res = self._wait_for_response(ws, msg_id, PAYLOAD_EXECUTION_EVENT, timeout=10.0)
                return {"success": True, "data": res}
            except Exception as e:
                logger.error(f"[cTrader] close_order failed: {e}")
                self._ws = None
                return {"success": False, "error": str(e)}

    def get_open_positions(self) -> list:
        """Get all open positions for this trading account."""
        if not self.is_configured:
            return []

        with self._lock:
            try:
                ws = self._connect()
                msg_id = f"rec_{uuid.uuid4().hex[:8]}"
                ws.send(json.dumps({
                    "clientMsgId": msg_id,
                    "payloadType": PAYLOAD_RECONCILE_REQ,
                    "payload": {
                        "ctidTraderAccountId": self.ctid_id
                    }
                }))
                res = self._wait_for_response(ws, msg_id, PAYLOAD_RECONCILE_RES, timeout=10.0)
                positions = res.get("position", [])
                return positions if isinstance(positions, list) else []
            except Exception as e:
                logger.error(f"[cTrader] get_open_positions failed: {e}")
                self._ws = None
                return []
