"""
MetaApi Cloud Client
Provides direct cloud REST integration with MetaTrader 5 broker servers.
Allows executing real trades and fetching broker-exact quotes from Linux cloud servers (Render)
without needing a local Windows PC running.
"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

_metaapi_session = None

def get_session() -> requests.Session:
    global _metaapi_session
    if _metaapi_session is None:
        _metaapi_session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=50)
        _metaapi_session.mount("https://", adapter)
        _metaapi_session.mount("http://", adapter)
    return _metaapi_session

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
                from app.config import settings
                db = SessionLocal()
                # Prioritize user who actually has metaapi credentials configured
                u = db.query(User).filter(
                    User.metaapi_account_id.isnot(None),
                    User.metaapi_account_id != ""
                ).order_by(User.id.desc()).first()
                if not u:
                    u = db.query(User).first()
                if u:
                    if not self._token and u.metaapi_token:
                        self._token = u.metaapi_token
                    if not self._account_id and u.metaapi_account_id:
                        self._account_id = u.metaapi_account_id
                    if not self._region:
                        self._region = crud.get_setting(db, u.id, "metaapi_region") or settings.METAAPI_REGION or "backup-new-york"
                db.close()
            except Exception:
                pass
        # Fall back to settings (.env.local / .env.prod) if not found in DB
        try:
            from app.config import settings
            if not self._token:
                self._token = settings.METAAPI_TOKEN
            if not self._account_id:
                self._account_id = settings.METAAPI_ACCOUNT_ID
            if not self._region:
                self._region = settings.METAAPI_REGION or "backup-new-york"
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
        session = get_session()
        for attempt in range(2):
            try:
                res = session.get(url, headers=self.headers, timeout=12)
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
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                logger.error(f"MetaApi get_account_information network error: {e}")
                return {}
            except Exception as e:
                logger.error(f"MetaApi get_account_information failed: {e}")
                return {}
        return {}

    def get_symbol_specs(self, symbol: str) -> dict:
        """Fetch broker's actual symbol specs — minStopDistance and point size per pair."""
        if not self.is_configured:
            return {}
        url = f"{self.client_base_url}/symbols/{symbol}"
        session = get_session()
        try:
            res = session.get(url, headers=self.headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                point = data.get("point", 0.00001)
                min_stop_points = data.get("minStopDistance", 0)
                digits = data.get("digits", 5)
                min_stop_price = min_stop_points * point
                logger.info(f"Symbol specs {symbol}: point={point}, minStopDistance={min_stop_points}pts={min_stop_price:.6f}, digits={digits}")
                return {"point": point, "min_stop_distance": min_stop_price, "digits": digits}
            logger.warning(f"get_symbol_specs {symbol} returned {res.status_code}")
            return {}
        except Exception as e:
            logger.error(f"get_symbol_specs failed for {symbol}: {e}")
            return {}

    def get_live_price(self, symbol: str) -> dict:
        """Fetch exact real-time bid, ask, and spread from broker's MT5 feed"""
        if not self.is_configured:
            return {}
        url = f"{self.client_base_url}/symbols/{symbol}/current-price"
        session = get_session()
        for attempt in range(2):
            try:
                res = session.get(url, headers=self.headers, timeout=15)
                if res.status_code == 200:
                    data = res.json()
                    bid = data.get("bid", 0.0)
                    ask = data.get("ask", 0.0)
                    spread = round((ask - bid) * (100 if "JPY" in symbol else 10000), 1)
                    return {"bid": bid, "ask": ask, "spread": spread}
                return {}
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                # Primary region timed out / connection error — try alternate region once
                alt_region = "new-york" if "backup" in self.region else "backup-new-york"
                alt_url = f"https://mt-client-api-v1.{alt_region}.agiliumtrade.ai/users/current/accounts/{self.account_id.strip()}/symbols/{symbol}/current-price"
                try:
                    res = session.get(alt_url, headers=self.headers, timeout=15)
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
            """Single request with retry and region fallback on timeout/connection error."""
            session = get_session()
            for attempt in range(3):
                try:
                    res = session.get(url, headers=self.headers, params=params, timeout=20)
                    if res.status_code == 200:
                        return _parse_batch(res.json())
                    logger.error("MetaApi get_candles HTTP %d for %s %s: %s",
                                 res.status_code, symbol, timeframe, res.text[:200])
                    return pd.DataFrame()
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    if attempt < 2:
                        time.sleep(0.8)
                        continue
                    alt_region = "new-york" if "backup" in self.region else "backup-new-york"
                    alt_url = (
                        f"https://mt-market-data-client-api-v1.{alt_region}.agiliumtrade.ai"
                        f"/users/current/accounts/{self.account_id.strip()}"
                        f"/historical-market-data/symbols/{symbol}/timeframes/{meta_tf}/candles"
                    )
                    try:
                        res = session.get(alt_url, headers=self.headers, params=params, timeout=20)
                        if res.status_code == 200:
                            logger.info("MetaApi get_candles fallback region %s OK for %s %s", alt_region, symbol, timeframe)
                            return _parse_batch(res.json())
                    except Exception as e2:
                        logger.error("MetaApi get_candles fallback failed for %s %s: %s", symbol, timeframe, e2)
                    return pd.DataFrame()
                except Exception as e:
                    logger.error("MetaApi get_candles failed for %s %s: %s", symbol, timeframe, e)
                    return pd.DataFrame()
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

    def get_candles_range(
        self,
        symbol: str,
        timeframe: str,
        start_dt: "datetime",
        end_dt: "datetime",
        warmup_candles: int = 100,
    ) -> "pd.DataFrame":
        """
        Fetch historical candles for a specific UTC date range from MetaApi.
        Paginates backwards from end_dt until start_dt is covered.
        warmup_candles: extra candles fetched before start_dt for indicator warm-up.
        Returns a DataFrame sorted ascending by time, with tz-naive UTC timestamps.
        """
        if not self.is_configured:
            raise RuntimeError("MetaApi credentials not configured")

        tf_map = {
            "M1": "1m", "M3": "3m", "M5": "5m", "M15": "15m", "M30": "30m",
            "H1": "1h", "H4": "4h", "D1": "1d"
        }
        meta_tf = tf_map.get(timeframe.upper())
        if not meta_tf:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        url = (
            f"{self.market_base_url}/historical-market-data"
            f"/symbols/{symbol}/timeframes/{meta_tf}/candles"
        )
        session = get_session()
        BATCH = 1000
        all_frames = []
        cursor = end_dt

        while True:
            params = {
                "limit": BATCH,
                "startTime": cursor.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            }
            batch_df = pd.DataFrame()
            for attempt in range(3):
                try:
                    res = session.get(url, headers=self.headers, params=params, timeout=30)
                    if res.status_code == 200:
                        raw = res.json()
                        if not raw:
                            break
                        batch_df = self._parse_candles(raw)
                        break
                    logger.error(
                        "get_candles_range HTTP %d %s %s: %s",
                        res.status_code, symbol, timeframe, res.text[:200]
                    )
                    if res.status_code in (401, 403):
                        raise RuntimeError(
                            f"MetaApi auth error {res.status_code} for {symbol} {timeframe}"
                        )
                    break
                except RuntimeError:
                    raise
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(
                        f"MetaApi network error for {symbol} {timeframe}: {e}"
                    ) from e

            if batch_df.empty:
                break

            all_frames.append(batch_df)
            oldest = batch_df["time"].min()

            warmup_start = pd.Timestamp(start_dt) - pd.Timedelta(
                minutes=self._tf_minutes(timeframe) * warmup_candles
            )
            if oldest <= warmup_start:
                break
            if len(batch_df) < BATCH:
                break

            cursor = oldest.to_pydatetime()

        if not all_frames:
            return pd.DataFrame()

        combined = pd.concat(all_frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )
        if getattr(combined["time"].dt, "tz", None) is not None:
            combined["time"] = combined["time"].dt.tz_convert("UTC").dt.tz_localize(None)
        logger.info(
            "get_candles_range %s %s: %d candles %s -> %s",
            symbol, timeframe, len(combined),
            combined["time"].min(), combined["time"].max()
        )
        return combined

    @staticmethod
    def _tf_minutes(tf: str) -> int:
        return {"M1": 1, "M3": 3, "M5": 5, "M15": 15, "M30": 30,
                "H1": 60, "H4": 240, "D1": 1440}.get(tf.upper(), 15)

    @staticmethod
    def _parse_candles(raw: list) -> "pd.DataFrame":
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        if "time" not in df.columns:
            return pd.DataFrame()
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
        if "tickVolume" in df.columns:
            df.rename(columns={"tickVolume": "volume"}, inplace=True)
        if "volume" not in df.columns:
            df["volume"] = 100
        needed = ["time", "open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return pd.DataFrame()
        return df[needed].copy()

    def place_order(self, symbol: str, direction: str, lot: float, entry: float = None, sl: float = None, tp: float = None, comment: str = "ForexAI") -> dict:
        """
        Places a LIMIT order at the OB entry price — matches backtest scenario exactly.
        Order waits at entry price until price returns to the OB zone, then fills.
        """
        if not self.is_configured:
            return {"success": False, "error": "MetaApi credentials not configured in backend"}

        digits = 3 if "JPY" in symbol else 5
        # Order type determined after live price check below
        if entry:
            action = None  # set dynamically after price check
        else:
            action = "ORDER_TYPE_BUY" if direction.upper() == "BUY" else "ORDER_TYPE_SELL"

        sl_r = round(float(sl), digits) if sl else None
        tp_r = round(float(tp), digits) if tp else None

        # Pre-validate: fetch live price + broker symbol specs for dynamic distance checks
        if entry:
            price_data = self.get_live_price(symbol)
            specs = self.get_symbol_specs(symbol)
            if price_data:
                ask = price_data.get("ask", 0)
                bid = price_data.get("bid", 0)
                entry_r = round(float(entry), digits)
                pip_mult = 100 if "JPY" in symbol else 10000

                # Use broker's real minStopDistance; fallback to safe defaults if specs unavailable
                broker_min = specs.get("min_stop_distance", 0)
                min_distance = broker_min if broker_min > 0 else (0.03 if "JPY" in symbol else 0.00030)
                min_sl_dist  = broker_min if broker_min > 0 else (0.05 if "JPY" in symbol else 0.00050)

                if direction.upper() == "BUY":
                    if sl_r and ask <= sl_r:
                        return {"success": False, "error": f"BUY rejected: current ask {ask} is at or below SL {sl_r}. OB structure broken."}
                    elif ask <= entry_r:
                        # Price inside OB zone (between entry and SL) — market order immediately at discount
                        action = "ORDER_TYPE_BUY"
                        logger.info(f"BUY MARKET (Discount in OB): price {ask} inside OB zone (entry={entry_r}, SL={sl_r})")
                    else:
                        gap = ask - entry_r
                        if gap < min_distance:
                            # Solution 1: Price is inside broker's Stops Level dead-zone.
                            # Instead of rejecting, execute instant MARKET order directly!
                            action = "ORDER_TYPE_BUY"
                            logger.info(f"BUY MARKET (Smart Hybrid): price {ask} is within {round(gap * pip_mult, 1)} pips of entry {entry_r} (dead-zone < {round(min_distance * pip_mult, 1)} pips) — placing MARKET order directly")
                        else:
                            # Price above entry with ample distance — place pending LIMIT order for pullback
                            action = "ORDER_TYPE_BUY_LIMIT"

                    # Solution 2: Spread and minimum stop distance protection
                    ref_price = ask if action == "ORDER_TYPE_BUY" else entry_r
                    if sl_r and (ref_price - sl_r) < min_sl_dist:
                        adjusted_sl = round(ref_price - min_sl_dist, digits)
                        logger.info(f"Adjusting BUY SL from {sl_r} to {adjusted_sl} to meet broker min stop distance ({min_sl_dist})")
                        sl_r = adjusted_sl
                    if tp_r and (tp_r - ref_price) < min_sl_dist:
                        adjusted_tp = round(ref_price + min_sl_dist * 2.0, digits)
                        logger.info(f"Adjusting BUY TP from {tp_r} to {adjusted_tp} to meet broker min stop distance")
                        tp_r = adjusted_tp

                else:
                    if sl_r and bid >= sl_r:
                        return {"success": False, "error": f"SELL rejected: current bid {bid} is at or above SL {sl_r}. OB structure broken."}
                    elif bid >= entry_r:
                        # Price inside OB zone (between entry and SL) — market order immediately at premium
                        action = "ORDER_TYPE_SELL"
                        logger.info(f"SELL MARKET (Premium in OB): price {bid} inside OB zone (entry={entry_r}, SL={sl_r})")
                    else:
                        gap = entry_r - bid
                        if gap < min_distance:
                            # Solution 1: Price is inside broker's Stops Level dead-zone.
                            # Auto-switch to instant MARKET order directly!
                            action = "ORDER_TYPE_SELL"
                            logger.info(f"SELL MARKET (Smart Hybrid): price {bid} is within {round(gap * pip_mult, 1)} pips of entry {entry_r} (dead-zone < {round(min_distance * pip_mult, 1)} pips) — placing MARKET order directly")
                        else:
                            action = "ORDER_TYPE_SELL_LIMIT"

                    # Solution 2: Spread and minimum stop distance protection
                    ref_price = bid if action == "ORDER_TYPE_SELL" else entry_r
                    if sl_r and (sl_r - ref_price) < min_sl_dist:
                        adjusted_sl = round(ref_price + min_sl_dist, digits)
                        logger.info(f"Adjusting SELL SL from {sl_r} to {adjusted_sl} to meet broker min stop distance ({min_sl_dist})")
                        sl_r = adjusted_sl
                    if tp_r and (ref_price - tp_r) < min_sl_dist:
                        adjusted_tp = round(ref_price - min_sl_dist * 2.0, digits)
                        logger.info(f"Adjusting SELL TP from {tp_r} to {adjusted_tp} to meet broker min stop distance")
                        tp_r = adjusted_tp

                logger.info(f"Validation passed: {symbol} {direction} {action} entry={entry_r}, SL={sl_r}, TP={tp_r}")

        url = f"{self.client_base_url}/trade"

        body = {
            "actionType": action,
            "symbol": symbol,
            "volume": float(lot),
            "comment": comment
        }
        if entry and action in ("ORDER_TYPE_BUY_LIMIT", "ORDER_TYPE_SELL_LIMIT"):
            body["openPrice"] = round(float(entry), digits)
            logger.info(f"Limit order {action} at openPrice={body['openPrice']}")
        if sl_r:
            body["stopLoss"] = sl_r
        elif sl:
            body["stopLoss"] = round(float(sl), digits)
        if tp_r:
            body["takeProfit"] = tp_r
        elif tp:
            body["takeProfit"] = round(float(tp), digits)

        try:
            logger.info(f"MetaApi placing order: {body}")
            session = get_session()
            res = session.post(url, headers=self.headers, json=body, timeout=15)
            data = res.json()
            if res.status_code in (200, 201) and data.get("numericCode") in (10008, 10009, None):
                ticket = data.get("orderId") or data.get("positionId") or "METAAPI_LIVE"
                logger.info(f"MetaApi order executed successfully: Ticket {ticket} for {symbol} {direction} {lot} lots")
                return {"success": True, "ticket": ticket, "data": data, "order_type": action}
            else:
                details = data.get("details")
                detail_str = f" ({details})" if details else ""
                err_msg = f"{data.get('message', '')}{detail_str}" or data.get("stringCode") or str(data)
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
            session = get_session()
            res = session.post(url, headers=self.headers, json=body, timeout=15)
            if res.status_code in (200, 201):
                return {"success": True, "ticket": position_id}
            return {"success": False, "error": res.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a pending limit or stop order on MT5"""
        if not self.is_configured:
            return {"success": False, "error": "MetaApi not configured"}

        url = f"{self.client_base_url}/trade"
        body = {
            "actionType": "ORDER_CANCEL",
            "orderId": str(order_id)
        }
        try:
            session = get_session()
            res = session.post(url, headers=self.headers, json=body, timeout=15)
            if res.status_code in (200, 201):
                logger.info(f"MetaApi pending order {order_id} cancelled successfully")
                return {"success": True, "ticket": order_id}
            return {"success": False, "error": res.text}
        except Exception as e:
            return {"success": False, "error": str(e)}

metaapi_client = MetaApiClient()
