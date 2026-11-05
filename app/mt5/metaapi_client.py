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
        if len(combined) < 50:
            logger.warning(
                "MetaApi returned only %d candles for %s %s (requested %d) — "
                "broker history may be limited or terminal not fully synced",
                len(combined), symbol, timeframe, count,
            )
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

            warmup_start = pd.Timestamp(start_dt).tz_localize(None) if pd.Timestamp(start_dt).tzinfo is None else pd.Timestamp(start_dt).tz_convert("UTC").tz_localize(None)
            warmup_start = warmup_start - pd.Timedelta(
                minutes=self._tf_minutes(timeframe) * warmup_candles
            )
            oldest_naive = oldest.tz_localize(None) if oldest.tzinfo is None else oldest.tz_convert("UTC").tz_localize(None)
            if oldest_naive <= warmup_start:
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

    def _get_position_by_ticket(self, ticket: str) -> dict:
        """
        Fetch a single confirmed open position by its ticket/position ID.
        Retries up to 3 times with 1s back-off to allow broker settlement.
        Returns the raw position dict, or {} if not found.
        """
        url     = f"{self.client_base_url}/positions/{ticket}"
        session = get_session()
        for attempt in range(3):
            try:
                res = session.get(url, headers=self.headers, timeout=10)
                if res.status_code == 200:
                    return res.json()
                if res.status_code == 404 and attempt < 2:
                    # Position may not be visible yet — wait and retry
                    time.sleep(1)
                    continue
                logger.warning("_get_position_by_ticket %s HTTP %d", ticket, res.status_code)
                return {}
            except Exception as e:
                logger.warning("_get_position_by_ticket %s attempt %d error: %s", ticket, attempt, e)
                if attempt < 2:
                    time.sleep(1)
        return {}

    def place_order(self, symbol: str, direction: str, lot: float, entry: float = None, sl: float = None, tp: float = None, comment: str = "ForexAI", client_order_id: str | None = None) -> dict:
        """
        Issue 3: Place a LIMIT order at signal price with 3-pip tolerance.
        If price is within 3 pips of entry, execute as MARKET immediately.
        Issue 5: Log actual fill price vs signal price for slippage tracking.
        Issue 6: Query broker min stop distance before submission and enforce it.
        After a successful submission:
          1. Fetch the confirmed broker position to get actual fill_price and lots.
          2. Write the trade record to the DB with confirmed data.
          3. If the DB write fails, immediately close the broker position and raise.
        Returns: {success, ticket, fill_price, actual_lots, db_write_ok, order_type, data}
        """
        if not self.is_configured:
            return {"success": False, "error": "MetaApi credentials not configured in backend"}

        digits   = 3 if "JPY" in symbol else 5
        pip_mult = 100 if "JPY" in symbol else 10000
        # Issue 3: 3-pip tolerance window for limit vs market decision
        LIMIT_TOLERANCE_PIPS = 3
        tolerance = LIMIT_TOLERANCE_PIPS / pip_mult

        sl_r = round(float(sl), digits) if sl else None
        tp_r = round(float(tp), digits) if tp else None

        # Issue 6: Query broker min stop distance before submission
        specs           = self.get_symbol_specs(symbol)
        broker_min_dist = specs.get("min_stop_distance", 0)
        min_sl_dist     = broker_min_dist if broker_min_dist > 0 else (0.05 if "JPY" in symbol else 0.00050)

        price_data = self.get_live_price(symbol) if entry else {}
        ask = price_data.get("ask", 0)
        bid = price_data.get("bid", 0)

        if entry:
            entry_r = round(float(entry), digits)

            if direction.upper() == "BUY":
                if sl_r and ask <= sl_r:
                    return {"success": False, "error": f"BUY rejected: ask {ask} <= SL {sl_r}. OB structure broken."}

                # Issue 6: enforce broker min SL distance
                if sl_r and (ask - sl_r) < min_sl_dist:
                    sl_r = round(ask - min_sl_dist, digits)
                    logger.info("Adjusting BUY SL to %s to meet broker min stop distance", sl_r)
                if tp_r and (tp_r - ask) < min_sl_dist:
                    tp_r = round(ask + min_sl_dist * 2.0, digits)
                    logger.info("Adjusting BUY TP to %s to meet broker min stop distance", tp_r)

                # Issue 3: within tolerance -> MARKET, else LIMIT
                gap = ask - entry_r
                if gap <= tolerance:
                    action = "ORDER_TYPE_BUY"
                    logger.info("BUY MARKET: ask %s within %.1f pips of entry %s", ask, LIMIT_TOLERANCE_PIPS, entry_r)
                else:
                    action = "ORDER_TYPE_BUY_LIMIT"
                    logger.info("BUY LIMIT at %s (ask=%s, gap=%.1f pips)", entry_r, ask, gap * pip_mult)

            else:  # SELL
                if sl_r and bid >= sl_r:
                    return {"success": False, "error": f"SELL rejected: bid {bid} >= SL {sl_r}. OB structure broken."}

                # Issue 6: enforce broker min SL distance
                if sl_r and (sl_r - bid) < min_sl_dist:
                    sl_r = round(bid + min_sl_dist, digits)
                    logger.info("Adjusting SELL SL to %s to meet broker min stop distance", sl_r)
                if tp_r and (bid - tp_r) < min_sl_dist:
                    tp_r = round(bid - min_sl_dist * 2.0, digits)
                    logger.info("Adjusting SELL TP to %s to meet broker min stop distance", tp_r)

                # Issue 3: within tolerance -> MARKET, else LIMIT
                gap = entry_r - bid
                if gap <= tolerance:
                    action = "ORDER_TYPE_SELL"
                    logger.info("SELL MARKET: bid %s within %.1f pips of entry %s", bid, LIMIT_TOLERANCE_PIPS, entry_r)
                else:
                    action = "ORDER_TYPE_SELL_LIMIT"
                    logger.info("SELL LIMIT at %s (bid=%s, gap=%.1f pips)", entry_r, bid, gap * pip_mult)
        else:
            action = "ORDER_TYPE_BUY" if direction.upper() == "BUY" else "ORDER_TYPE_SELL"

        url  = f"{self.client_base_url}/trade"
        body = {
            "actionType": action,
            "symbol":     symbol,
            "volume":     float(lot),
            "comment":    comment,
        }
        # Idempotency key: lets the broker reject a duplicate submission when a
        # retry fires after a timeout-but-succeeded request. Without it,
        # ORDER_MAX_RETRIES=2 could double-fill the same signal.
        if client_order_id:
            body["clientId"] = str(client_order_id)
        if entry and action in ("ORDER_TYPE_BUY_LIMIT", "ORDER_TYPE_SELL_LIMIT"):
            body["openPrice"] = round(float(entry), digits)
        if sl_r:
            body["stopLoss"] = sl_r
        elif sl:
            body["stopLoss"] = round(float(sl), digits)
        if tp_r:
            body["takeProfit"] = tp_r
        elif tp:
            body["takeProfit"] = round(float(tp), digits)

        try:
            logger.info("MetaApi placing order: %s", body)
            session = get_session()
            res  = session.post(url, headers=self.headers, json=body, timeout=15)
            data = res.json()
            if res.status_code not in (200, 201) or data.get("numericCode") not in (10008, 10009, None):
                details = data.get("details")
                err_msg = f"{data.get('message', '')}{' (' + details + ')' if details else ''}" or str(data)
                logger.error("MetaApi trade rejected: %s", err_msg)
                return {"success": False, "error": err_msg}
        except Exception as e:
            logger.error("MetaApi place_order exception: %s", e)
            return {"success": False, "error": str(e)}

        # ── Order accepted by broker ──────────────────────────────────────────
        # A ticket is mandatory: reconciliation, SL/TP reattach and orphan
        # detection all key off it. The previous fallback string "METAAPI_LIVE"
        # was shared by every position that hit this branch, which made ghost and
        # orphan detection meaningless. Treat a missing ticket as ambiguous —
        # never fabricate one, and never let the caller blindly retry.
        ticket_raw = data.get("orderId") or data.get("positionId")
        if not ticket_raw:
            logger.critical(
                "MetaApi accepted the order but returned no orderId/positionId — "
                "position cannot be reconciled. payload=%s", data,
            )
            return {
                "success":   False,
                "ambiguous": True,
                "error":     "Broker accepted order but returned no ticket; "
                             "manual reconciliation required",
                "data":      data,
            }
        ticket = str(ticket_raw)

        # Step 1: Fetch confirmed position from broker to get actual fill & lots
        pos          = self._get_position_by_ticket(ticket)
        fill_price   = float(pos.get("openPrice") or
                             data.get("openPrice") or data.get("currentPrice") or
                             data.get("price") or 0)
        actual_lots  = float(pos.get("volume") or lot)

        # Issue 5: log slippage vs signal entry price
        if fill_price and entry:
            slip_pips = round((fill_price - float(entry)) * pip_mult, 1)
            if abs(slip_pips) > 3:
                logger.warning(
                    "[Slippage] %s %s fill=%.5f signal=%.5f slippage=%.1f pips (>3 pip threshold)",
                    symbol, direction, fill_price, float(entry), slip_pips,
                )
            else:
                logger.info(
                    "[Slippage] %s %s fill=%.5f signal=%.5f slippage=%.1f pips",
                    symbol, direction, fill_price, float(entry), slip_pips,
                )

        logger.info("MetaApi order executed: ticket=%s %s %s fill=%.5f lots=%.4f",
                    ticket, symbol, direction, fill_price, actual_lots)

        # Step 2: Write trade record to DB with confirmed broker data
        db_write_ok = False
        try:
            from app.db.database import SessionLocal
            from app.db import models as _models
            _db = SessionLocal()
            try:
                # Locate the PENDING trade row created by the caller (matched by
                # symbol + direction + result=PENDING, most recently opened)
                pending = (
                    _db.query(_models.Trade)
                    .filter(
                        _models.Trade.symbol    == symbol,
                        _models.Trade.direction == direction,
                        _models.Trade.result    == "PENDING",
                    )
                    .order_by(_models.Trade.open_time.desc())
                    .first()
                )
                if pending:
                    pending.mt5_ticket        = ticket
                    pending.actual_fill_price = fill_price if fill_price else None
                    pending.lot_size          = actual_lots
                    pending.result            = "OPEN"
                    _db.commit()
                    db_write_ok = True
                    logger.info("[place_order] DB trade #%d updated: ticket=%s fill=%.5f lots=%.4f",
                                pending.id, ticket, fill_price or 0, actual_lots)
                else:
                    # No PENDING row found — insert a minimal record so the
                    # trade is never invisible to the monitor
                    new_trade = _models.Trade(
                        symbol            = symbol,
                        direction         = direction,
                        entry_price       = fill_price or (float(entry) if entry else 0),
                        actual_fill_price = fill_price if fill_price else None,
                        stop_loss         = float(sl) if sl else None,
                        take_profit       = float(tp) if tp else None,
                        lot_size          = actual_lots,
                        mt5_ticket        = ticket,
                        open_time         = datetime.utcnow(),
                        result            = "OPEN",
                        strategy          = comment,
                    )
                    _db.add(new_trade)
                    _db.commit()
                    db_write_ok = True
                    logger.warning("[place_order] No PENDING row found — inserted fallback trade #%d ticket=%s",
                                   new_trade.id, ticket)
            finally:
                _db.close()
        except Exception as db_exc:
            # Step 3: DB write failed — close the broker position immediately
            # to prevent a ghost trade (live position with no DB record)
            logger.critical(
                "[place_order] GHOST_TRADE_CLOSED ticket=%s symbol=%s — DB write failed: %s",
                ticket, symbol, db_exc,
            )
            try:
                self.close_order(ticket)
                logger.critical(
                    "[place_order] GHOST_TRADE_CLOSED ticket=%s — broker position closed successfully",
                    ticket,
                )
            except Exception as close_exc:
                logger.critical(
                    "[place_order] GHOST_TRADE_CLOSED ticket=%s — broker close ALSO failed: %s",
                    ticket, close_exc,
                )
            raise db_exc

        return {
            "success":     True,
            "ticket":      ticket,
            "fill_price":  fill_price,
            "actual_lots": actual_lots,
            "db_write_ok": db_write_ok,
            # Legacy keys kept so any other callers don't break
            "order_type":  action,
            "data":        data,
        }

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

    def get_open_positions(self) -> list | None:
        """
        Fetch all currently open positions from the broker.
        Returns a list of dicts with keys: id, symbol, type, volume,
        openPrice, stopLoss, takeProfit, openTime.

        Returns an EMPTY LIST only when the broker genuinely reports no open
        positions. Returns None when the state is UNKNOWN (network error, HTTP
        error, not configured).

        This distinction is critical: TradeMonitor uses this to decide whether a
        DB trade is a ghost. Returning [] on error made a single failed or
        rate-limited request look like "all positions are gone", which marked
        live trades RECONCILED_MISSING and let orphan logic close real trades.
        Callers must treat None as "skip this cycle", never as "empty".
        """
        if not self.is_configured:
            return None
        url = f"{self.client_base_url}/positions"
        session = get_session()
        for attempt in range(2):
            try:
                res = session.get(url, headers=self.headers, timeout=15)
                if res.status_code == 200:
                    raw = res.json()
                    positions = []
                    for p in (raw if isinstance(raw, list) else []):
                        positions.append({
                            "id":         str(p.get("id", "")),
                            "symbol":     p.get("symbol", ""),
                            "type":       p.get("type", ""),       # POSITION_TYPE_BUY / SELL
                            "volume":     float(p.get("volume", 0)),
                            "openPrice":  float(p.get("openPrice", 0)),
                            "stopLoss":   float(p.get("stopLoss") or 0),
                            "takeProfit": float(p.get("takeProfit") or 0),
                            "openTime":   p.get("time") or p.get("openTime", ""),
                            "comment":    p.get("comment", ""),
                        })
                    return positions
                logger.warning("get_open_positions HTTP %d: %s", res.status_code, res.text[:120])
                return None
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                logger.error("get_open_positions network error after retry — state UNKNOWN")
                return None
            except Exception as e:
                logger.error("get_open_positions failed — state UNKNOWN: %s", e)
                return None
        return None

    def modify_position(self, position_id: str, sl: float, tp: float) -> dict:
        """
        Re-attach SL and TP to an existing open position.
        Used by TradeMonitor when SL/TP are found missing on the broker.
        """
        if not self.is_configured:
            return {"success": False, "error": "MetaApi not configured"}
        url  = f"{self.client_base_url}/trade"
        body = {
            "actionType": "POSITION_MODIFY",
            "positionId": str(position_id),
            "stopLoss":   round(float(sl), 5),
            "takeProfit": round(float(tp), 5),
        }
        try:
            session = get_session()
            res  = session.post(url, headers=self.headers, json=body, timeout=15)
            data = res.json()
            if res.status_code in (200, 201):
                logger.info("modify_position %s SL=%s TP=%s OK", position_id, sl, tp)
                return {"success": True}
            err = data.get("message", res.text[:120])
            logger.error("modify_position %s failed: %s", position_id, err)
            return {"success": False, "error": err}
        except Exception as e:
            logger.error("modify_position exception: %s", e)
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
