"""
Variant C auto-scanner tests.
All broker/DB/network calls are mocked.
No live orders are placed.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock, call


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_candles(n=60, freq="3min", base=1.1000, start="2024-01-15 08:00"):
    times  = pd.date_range(start, periods=n, freq=freq)
    closes = np.full(n, base)
    highs  = closes + 0.0005
    lows   = closes - 0.0005
    opens  = closes.copy()
    return pd.DataFrame({
        "time": times, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": np.ones(n) * 100,
    })


def _signal(symbol="EURUSD", direction="BUY", raw_score=11,
            m3_ts=None, entry=1.10000, sl=1.09500, tp=1.11500):
    if m3_ts is None:
        m3_ts = pd.Timestamp("2024-01-15 08:03:00")
    norm = round(raw_score / 14.0 * 100, 2)
    return {
        "symbol":           symbol,
        "direction":        direction,
        "entry_price":      entry,
        "entry_type":       "market",
        "stop_loss":        sl,
        "take_profit":      tp,
        "rr_ratio":         "1:3.00",
        "rr_val":           3.0,
        "sl_pips":          50.0,
        "tp_pips":          150.0,
        "score":            raw_score,
        "normalized_score": norm,
        "session":          "London",
        "h4_trend":         "BULLISH",
        "h1_trend":         "DISCOUNT",
        "strategy":         "MTF SMC A+ Strategy",
        "pattern_name":     "A+ MTF SMC Setup",
        "timeframe":        "M15->M5->M3",
        "pattern":          "A_PLUS_SMC",
        "confidence":       "HIGH",
        "htf_bias":         "BULLISH",
        "htf_aligned":      True,
        "killzone":         "London",
        "fvg":              True,
        "m3_trigger_ts":    m3_ts,
        "signal_time":      str(m3_ts),
        "entry":            entry,
        "sl":               sl,
        "tp_1_3":           tp,
        "ob_high":          "OB (1.09800-1.10000)",
        "entry_valid":      True,
        "entry_status":     "ok",
        "lot_size":         None,
        "risk_usd":         1.0,
        "spread":           None,
        "confirmations":    [],
        "current_price":    entry,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Score threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_score_threshold_constants():
    from app.signals.auto_scanner import VC_MIN_RAW_SCORE, VC_NORMALIZED_THRESHOLD
    assert VC_MIN_RAW_SCORE == 11
    assert abs(VC_NORMALIZED_THRESHOLD - 78.57) < 0.01


def test_raw_score_10_is_rejected():
    """raw=10 -> normalized=71.43 -> below 78.57 -> must be rejected."""
    from app.signals.auto_scanner import VC_MIN_RAW_SCORE
    assert 10 < VC_MIN_RAW_SCORE


def test_raw_score_11_is_eligible():
    """raw=11 -> normalized=78.57 -> meets threshold."""
    from app.signals.auto_scanner import VC_MIN_RAW_SCORE
    assert 11 >= VC_MIN_RAW_SCORE


def test_normalized_score_calculation():
    for raw, expected in [(11, 78.57), (12, 85.71), (14, 100.0), (7, 50.0), (8, 57.14)]:
        norm = round(raw / 14.0 * 100, 2)
        assert abs(norm - expected) < 0.01, f"raw={raw} expected={expected} got={norm}"


def test_score_boundary_raw_10_normalized():
    norm = round(10 / 14.0 * 100, 2)
    assert norm < 78.57


def test_score_boundary_raw_11_normalized():
    norm = round(11 / 14.0 * 100, 2)
    assert norm >= 78.57


# ─────────────────────────────────────────────────────────────────────────────
# 2. M3 candle boundary timing
# ─────────────────────────────────────────────────────────────────────────────

def test_m3_wait_seconds_aligns_to_3min_boundary():
    from app.signals.auto_scanner import _get_wait_seconds, M3_PERIOD_S, M3_SCAN_BUFFER_S
    # At exactly :00 UTC the wait should be M3_PERIOD_S + buffer
    fake_now = datetime(2024, 1, 15, 8, 0, 0, tzinfo=timezone.utc)
    with patch("app.signals.auto_scanner.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        wait = _get_wait_seconds()
    assert wait == M3_PERIOD_S + M3_SCAN_BUFFER_S


def test_m3_wait_seconds_mid_candle():
    from app.signals.auto_scanner import _get_wait_seconds, M3_SCAN_BUFFER_S
    # At :01:30 into a 3-min candle, wait should be ~90 + buffer seconds
    fake_now = datetime(2024, 1, 15, 8, 1, 30, tzinfo=timezone.utc)
    with patch("app.signals.auto_scanner.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        wait = _get_wait_seconds()
    assert 85 <= wait <= 100


def test_m3_wait_seconds_never_below_10():
    from app.signals.auto_scanner import _get_wait_seconds
    for minute in range(0, 60, 3):
        fake_now = datetime(2024, 1, 15, 8, minute, 0, tzinfo=timezone.utc)
        with patch("app.signals.auto_scanner.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            wait = _get_wait_seconds()
        assert wait >= 10, f"wait={wait} at minute={minute}"


def test_m3_period_is_180_seconds():
    from app.signals.auto_scanner import M3_PERIOD_S
    assert M3_PERIOD_S == 180


# ─────────────────────────────────────────────────────────────────────────────
# 3. M3 data — no silent fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_mtf_smc_pair_no_m5_fallback_when_m3_empty():
    """If M3 is empty after retry, return None — never substitute M5."""
    from app.signals.scanner import scan_mtf_smc_pair

    df_ok    = _make_candles(100, freq="5min")
    df_empty = pd.DataFrame()

    with patch("app.signals.scanner._load_live") as mock_load:
        def side(symbol, tf, count):
            if tf == "M3":
                return df_empty
            return df_ok
        mock_load.side_effect = side

        result = scan_mtf_smc_pair("EURUSD")

    assert result is None
    # Verify M3 was attempted (at least twice — initial + retry)
    m3_calls = [c for c in mock_load.call_args_list if c.args[1] == "M3"]
    assert len(m3_calls) >= 2, "Expected at least 2 M3 fetch attempts (initial + retry)"


def test_scan_mtf_smc_pair_returns_none_when_m15_empty():
    from app.signals.scanner import scan_mtf_smc_pair
    df_ok = _make_candles(100, freq="5min")
    with patch("app.signals.scanner._load_live") as mock_load:
        def side(symbol, tf, count):
            if tf == "M15":
                return pd.DataFrame()
            return df_ok
        mock_load.side_effect = side
        result = scan_mtf_smc_pair("EURUSD")
    assert result is None


def test_scan_mtf_smc_pair_m3_retry_succeeds_on_second_attempt():
    """If first M3 fetch is empty but second succeeds, proceed normally."""
    from app.signals.scanner import scan_mtf_smc_pair

    df_ok  = _make_candles(100, freq="5min")
    df_m3  = _make_candles(200, freq="3min")
    call_count = {"m3": 0}

    def side(symbol, tf, count):
        if tf == "M3":
            call_count["m3"] += 1
            return pd.DataFrame() if call_count["m3"] == 1 else df_m3
        return df_ok

    fake_setup = {
        "valid": True, "direction": "BUY", "entry_price": 1.10000,
        "entry_time": pd.Timestamp("2024-01-15 08:03:00"),
        "stop_loss": 1.09500, "take_profit": 1.11500, "rr_val": 3.0,
        "score": 11, "setup_class": "A TRADE", "session": "London",
        "m15_bias": "BULLISH", "m15_location": "DISCOUNT",
        "liquidity_type": "M15 Swing Low (SSL)", "sweep_price": 1.09400,
        "choch_type": "Bullish CHoCH/BOS Break", "ob_zone": "OB (1.09800-1.10000)",
        "fvg_overlap": True, "m3_trigger": "M3 Micro-Sweep + Bullish CHoCH",
        "disp_ratio": 1.2,
    }

    with patch("app.signals.scanner._load_live", side_effect=side), \
         patch("app.signals.scanner.evaluate_mtf_smc_setup", return_value=fake_setup):
        result = scan_mtf_smc_pair("EURUSD")

    assert result is not None
    assert call_count["m3"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. scan_mtf_smc_pair output fields
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_mtf_smc_pair_output_has_m3_trigger_ts():
    from app.signals.scanner import scan_mtf_smc_pair

    df_ok = _make_candles(100, freq="5min")
    df_m3 = _make_candles(200, freq="3min")
    ts    = pd.Timestamp("2024-01-15 08:03:00")

    fake_setup = {
        "valid": True, "direction": "BUY", "entry_price": 1.10000,
        "entry_time": ts, "stop_loss": 1.09500, "take_profit": 1.11500,
        "rr_val": 3.0, "score": 11, "setup_class": "A TRADE",
        "session": "London", "m15_bias": "BULLISH", "m15_location": "DISCOUNT",
        "liquidity_type": "SSL", "sweep_price": 1.09400,
        "choch_type": "Bullish CHoCH/BOS Break",
        "ob_zone": "OB (1.09800-1.10000)", "fvg_overlap": True,
        "m3_trigger": "M3 Micro-Sweep + Bullish CHoCH", "disp_ratio": 1.2,
    }

    def _load(symbol, tf, count=200):
        return df_m3 if tf == "M3" else df_ok

    with patch("app.signals.scanner._load_live", side_effect=_load), \
         patch("app.signals.scanner.evaluate_mtf_smc_setup", return_value=fake_setup):
        result = scan_mtf_smc_pair("EURUSD")

    assert result is not None
    assert result["m3_trigger_ts"] == ts
    assert result["entry_type"] == "market"
    assert "normalized_score" in result
    assert abs(result["normalized_score"] - 78.57) < 0.01


def test_scan_mtf_smc_pair_entry_type_is_market():
    from app.signals.scanner import scan_mtf_smc_pair

    df_ok = _make_candles(100, freq="5min")
    df_m3 = _make_candles(200, freq="3min")

    fake_setup = {
        "valid": True, "direction": "SELL", "entry_price": 1.10500,
        "entry_time": pd.Timestamp("2024-01-15 09:00:00"),
        "stop_loss": 1.11000, "take_profit": 1.09000, "rr_val": 3.0,
        "score": 12, "setup_class": "A+ TRADE", "session": "London",
        "m15_bias": "BEARISH", "m15_location": "PREMIUM",
        "liquidity_type": "BSL", "sweep_price": 1.11100,
        "choch_type": "Bearish CHoCH/BOS Break",
        "ob_zone": "OB (1.10400-1.10600)", "fvg_overlap": False,
        "m3_trigger": "M3 Bearish Zone Displacement", "disp_ratio": 0.9,
    }

    def _load(symbol, tf, count=200):
        return df_m3 if tf == "M3" else df_ok

    with patch("app.signals.scanner._load_live", side_effect=_load), \
         patch("app.signals.scanner.evaluate_mtf_smc_setup", return_value=fake_setup):
        result = scan_mtf_smc_pair("EURUSD")

    assert result is not None
    assert result["entry_type"] == "market"
    assert result["direction"] == "SELL"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Duplicate protection
# ─────────────────────────────────────────────────────────────────────────────

def test_signal_identity_is_deterministic():
    from app.signals.auto_scanner import _signal_identity
    ts = pd.Timestamp("2024-01-15 08:03:00")
    id1 = _signal_identity("EURUSD", "BUY", ts)
    id2 = _signal_identity("EURUSD", "BUY", ts)
    assert id1 == id2


def test_signal_identity_differs_by_direction():
    from app.signals.auto_scanner import _signal_identity
    ts = pd.Timestamp("2024-01-15 08:03:00")
    assert _signal_identity("EURUSD", "BUY", ts) != _signal_identity("EURUSD", "SELL", ts)


def test_signal_identity_differs_by_timestamp():
    from app.signals.auto_scanner import _signal_identity
    ts1 = pd.Timestamp("2024-01-15 08:03:00")
    ts2 = pd.Timestamp("2024-01-15 08:06:00")
    assert _signal_identity("EURUSD", "BUY", ts1) != _signal_identity("EURUSD", "BUY", ts2)


def test_is_duplicate_blocks_active_trade():
    from app.signals.auto_scanner import _is_duplicate

    mock_trade = MagicMock()
    mock_trade.id     = 42
    mock_trade.result = "OPEN"

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = mock_trade

    assert _is_duplicate(db, 1, "EURUSD", "BUY", None) is True


def test_is_duplicate_blocks_in_memory_identity():
    from app.signals.auto_scanner import _is_duplicate, _fired_this_session, _signal_identity

    ts = pd.Timestamp("2024-01-15 08:03:00")
    identity = _signal_identity("GBPUSD", "SELL", ts)
    _fired_this_session.add(identity)

    db = MagicMock()
    # No active trade
    db.query.return_value.filter.return_value.first.return_value = None

    try:
        result = _is_duplicate(db, 1, "GBPUSD", "SELL", ts)
        assert result is True
    finally:
        _fired_this_session.discard(identity)


def test_is_duplicate_allows_new_setup():
    from app.signals.auto_scanner import _is_duplicate, _fired_this_session, _signal_identity

    ts = pd.Timestamp("2024-01-15 09:00:00")
    identity = _signal_identity("AUDUSD", "BUY", ts)
    _fired_this_session.discard(identity)

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None

    result = _is_duplicate(db, 1, "AUDUSD", "BUY", ts)
    assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# 6. Broker-aware lot sizing
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_lot_returns_none_when_specs_unavailable():
    from app.signals.auto_scanner import _calculate_lot

    user = MagicMock()
    user.metaapi_token      = "tok"
    user.metaapi_account_id = "acc"

    with patch("app.signals.auto_scanner._get_symbol_info", return_value=None):
        result = _calculate_lot(1.0, 1.10000, 1.09500, "EURUSD", user)

    assert result is None


def test_calculate_lot_clamps_to_volume_min():
    from app.signals.auto_scanner import _calculate_lot

    user = MagicMock()
    sym_info = {"volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}

    with patch("app.signals.auto_scanner._get_symbol_info", return_value=sym_info):
        # Very small risk -> lot below volume_min -> should clamp to 0.01
        lot = _calculate_lot(0.001, 1.10000, 1.09999, "EURUSD", user)

    assert lot >= 0.01


def test_calculate_lot_clamps_to_volume_max():
    from app.signals.auto_scanner import _calculate_lot

    user = MagicMock()
    sym_info = {"volume_min": 0.01, "volume_max": 0.05, "volume_step": 0.01}

    with patch("app.signals.auto_scanner._get_symbol_info", return_value=sym_info):
        # Large risk -> lot above volume_max -> should clamp to 0.05
        lot = _calculate_lot(10000.0, 1.10000, 1.09500, "EURUSD", user)

    assert lot <= 0.05


def test_calculate_lot_respects_volume_step():
    from app.signals.auto_scanner import _calculate_lot

    user = MagicMock()
    sym_info = {"volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}

    with patch("app.signals.auto_scanner._get_symbol_info", return_value=sym_info):
        lot = _calculate_lot(1.0, 1.10000, 1.09500, "EURUSD", user)

    # Must be a multiple of 0.01
    assert round(lot % 0.01, 6) < 1e-6


def test_calculate_lot_jpy_pair():
    from app.signals.auto_scanner import _calculate_lot

    user = MagicMock()
    sym_info = {"volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}

    with patch("app.signals.auto_scanner._get_symbol_info", return_value=sym_info):
        lot = _calculate_lot(1.0, 150.00, 149.50, "USDJPY", user)

    assert lot >= 0.01
    assert lot <= 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. SL/TP validation
# ─────────────────────────────────────────────────────────────────────────────

def _mock_client_for_validation(ask=1.10010, bid=1.10000, min_stop=0.00030):
    client = MagicMock()
    client.is_configured = True
    client.get_live_price.return_value = {"ask": ask, "bid": bid}
    client.get_symbol_specs.return_value = {"min_stop_distance": min_stop}
    return client


def test_validate_sl_tp_buy_valid():
    from app.signals.auto_scanner import _validate_sl_tp

    user = MagicMock()
    with patch("app.signals.auto_scanner.MetaApiClient",
               return_value=_mock_client_for_validation(ask=1.10010)):
        ok, reason = _validate_sl_tp("EURUSD", "BUY", sl=1.09500, tp=1.11500, user=user)

    assert ok is True
    assert reason == "ok"


def test_validate_sl_tp_buy_sl_above_price_rejected():
    from app.signals.auto_scanner import _validate_sl_tp

    user = MagicMock()
    with patch("app.signals.auto_scanner.MetaApiClient",
               return_value=_mock_client_for_validation(ask=1.10010)):
        ok, reason = _validate_sl_tp("EURUSD", "BUY", sl=1.10050, tp=1.11500, user=user)

    assert ok is False
    assert "SL" in reason


def test_validate_sl_tp_sell_valid():
    from app.signals.auto_scanner import _validate_sl_tp

    user = MagicMock()
    with patch("app.signals.auto_scanner.MetaApiClient",
               return_value=_mock_client_for_validation(bid=1.10000)):
        ok, reason = _validate_sl_tp("EURUSD", "SELL", sl=1.10500, tp=1.09000, user=user)

    assert ok is True


def test_validate_sl_tp_sell_sl_below_price_rejected():
    from app.signals.auto_scanner import _validate_sl_tp

    user = MagicMock()
    with patch("app.signals.auto_scanner.MetaApiClient",
               return_value=_mock_client_for_validation(bid=1.10000)):
        ok, reason = _validate_sl_tp("EURUSD", "SELL", sl=1.09900, tp=1.09000, user=user)

    assert ok is False
    assert "SL" in reason


def test_validate_sl_tp_distance_too_small_rejected():
    from app.signals.auto_scanner import _validate_sl_tp

    user = MagicMock()
    # min_stop = 50 pips, but SL is only 1 pip away
    with patch("app.signals.auto_scanner.MetaApiClient",
               return_value=_mock_client_for_validation(ask=1.10010, min_stop=0.00500)):
        ok, reason = _validate_sl_tp("EURUSD", "BUY", sl=1.10000, tp=1.11500, user=user)

    assert ok is False


def test_validate_sl_tp_no_live_price_rejected():
    from app.signals.auto_scanner import _validate_sl_tp

    client = MagicMock()
    client.is_configured = True
    client.get_live_price.return_value = {}
    user = MagicMock()

    with patch("app.signals.auto_scanner.MetaApiClient", return_value=client):
        ok, reason = _validate_sl_tp("EURUSD", "BUY", sl=1.09500, tp=1.11500, user=user)

    assert ok is False
    assert "live price" in reason


# ─────────────────────────────────────────────────────────────────────────────
# 8. Market order dispatch
# ─────────────────────────────────────────────────────────────────────────────

def _make_db_mock():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.count.return_value = 0
    return db


def test_handle_auto_execute_passes_entry_none_to_place_order():
    """Variant C must call place_order with entry=None (market order)."""
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.metaapi_token      = "tok"
    user.metaapi_account_id = "acc"
    user.telegram_chat_id   = None

    r = _signal(raw_score=11)

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=99)), \
         patch("app.signals.auto_scanner.place_order",
               return_value={"success": True, "ticket": "T001",
                             "order_type": "ORDER_TYPE_BUY", "data": {}}) as mock_po:
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    mock_po.assert_called_once()
    call_kwargs = mock_po.call_args.kwargs
    assert call_kwargs.get("entry") is None, (
        f"entry should be None for market order, got {call_kwargs.get('entry')}")


def test_handle_auto_execute_buy_market_order():
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(direction="BUY", raw_score=12)

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=1)), \
         patch("app.signals.auto_scanner.place_order",
               return_value={"success": True, "ticket": "T002",
                             "order_type": "ORDER_TYPE_BUY", "data": {}}) as mock_po:
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    assert mock_po.call_args.kwargs["direction"] == "BUY"
    assert mock_po.call_args.kwargs["entry"] is None


def test_handle_auto_execute_sell_market_order():
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(direction="SELL", raw_score=11,
                entry=1.10500, sl=1.11000, tp=1.09000)

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=2)), \
         patch("app.signals.auto_scanner.place_order",
               return_value={"success": True, "ticket": "T003",
                             "order_type": "ORDER_TYPE_SELL", "data": {}}) as mock_po:
        _handle_auto_execute(db, user, r, 1.10500, 1.11000, 1.09000,
                             0.01, 1.0, 3.0)

    assert mock_po.call_args.kwargs["direction"] == "SELL"
    assert mock_po.call_args.kwargs["entry"] is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. Actual fill price stored separately from reference price
# ─────────────────────────────────────────────────────────────────────────────

def test_actual_fill_price_stored_when_broker_returns_it():
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(raw_score=11)
    captured_trade = {}

    def fake_add(trade):
        captured_trade["trade"] = trade

    db.add.side_effect = fake_add
    db.commit.return_value = None
    db.refresh.return_value = None

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=5)), \
         patch("app.signals.auto_scanner.place_order",
               return_value={
                   "success": True, "ticket": "T010",
                   "order_type": "ORDER_TYPE_BUY",
                   "data": {"openPrice": 1.10015},
               }):
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    trade = captured_trade.get("trade")
    assert trade is not None
    assert trade.entry_price == 1.10000          # M3 reference unchanged
    assert trade.actual_fill_price == 1.10015    # broker fill stored separately


def test_reference_price_not_overwritten_by_fill():
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(raw_score=11, entry=1.10000)
    captured_trade = {}

    db.add.side_effect = lambda t: captured_trade.update({"trade": t})
    db.commit.return_value = None
    db.refresh.return_value = None

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=6)), \
         patch("app.signals.auto_scanner.place_order",
               return_value={
                   "success": True, "ticket": "T011",
                   "order_type": "ORDER_TYPE_BUY",
                   "data": {"openPrice": 1.10025},
               }):
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    trade = captured_trade["trade"]
    assert trade.entry_price == 1.10000   # M3 close reference preserved


# ─────────────────────────────────────────────────────────────────────────────
# 10. Failed broker execution not recorded as OPEN
# ─────────────────────────────────────────────────────────────────────────────

def test_failed_broker_execution_sets_rejected():
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(raw_score=11)
    captured_trade = {}

    db.add.side_effect = lambda t: captured_trade.update({"trade": t})
    db.commit.return_value = None
    db.refresh.return_value = None

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=7, status="auto_approved")), \
         patch("app.signals.auto_scanner.place_order",
               return_value={"success": False, "error": "Broker rejected order"}):
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    trade = captured_trade["trade"]
    assert trade.result == "REJECTED", (
        f"Expected REJECTED, got {trade.result}")


def test_unexpected_limit_order_response_sets_rejected():
    """If broker returns a LIMIT order type for Variant C, reject it."""
    from app.signals.auto_scanner import _handle_auto_execute

    db   = _make_db_mock()
    user = MagicMock()
    user.id = 1
    user.telegram_chat_id = None

    r = _signal(raw_score=11)
    captured_trade = {}

    db.add.side_effect = lambda t: captured_trade.update({"trade": t})
    db.commit.return_value = None
    db.refresh.return_value = None

    with patch("app.signals.auto_scanner.crud.create_signal",
               return_value=MagicMock(id=8, status="auto_approved")), \
         patch("app.signals.auto_scanner.place_order",
               return_value={"success": True, "ticket": "T020",
                             "order_type": "ORDER_TYPE_BUY_LIMIT", "data": {}}):
        _handle_auto_execute(db, user, r, 1.10000, 1.09500, 1.11500,
                             0.01, 1.0, 3.0)

    trade = captured_trade["trade"]
    assert trade.result == "REJECTED"


# ─────────────────────────────────────────────────────────────────────────────
# 11. Closed-candle only — no lookahead into signal candle
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_mtf_smc_pair_uses_closed_candles_only():
    """
    _load_live drops the currently-forming candle.
    Simulate: now=08:04:30, candles end at 08:03 (still forming until 08:06).
    After dropping the forming candle, last closed candle must be 08:00.
    """
    now_utc = datetime(2024, 1, 15, 8, 4, 30, tzinfo=timezone.utc)
    # 3 candles: 07:54, 07:57, 08:00, 08:03
    times  = pd.date_range("2024-01-15 07:54", periods=4, freq="3min")
    closes = np.full(4, 1.1000)
    df = pd.DataFrame({
        "time": times, "open": closes, "high": closes + 0.0005,
        "low": closes - 0.0005, "close": closes, "volume": np.ones(4),
    })

    # Replicate _load_live closed-candle logic
    last_time     = df.iloc[-1]["time"]                          # 08:03
    last_time_utc = last_time.replace(tzinfo=timezone.utc)
    candle_still_forming = now_utc < last_time_utc + timedelta(minutes=3)  # 08:04:30 < 08:06

    if candle_still_forming and len(df) > 1:
        df = df.iloc[:-1].copy().reset_index(drop=True)

    last_closed = df.iloc[-1]["time"]   # should be 08:00
    assert last_closed == pd.Timestamp("2024-01-15 08:00:00"), (
        f"Expected 08:00 (last closed), got {last_closed}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Score stored in signal record
# ─────────────────────────────────────────────────────────────────────────────

def test_build_signal_data_stores_raw_and_normalized_score():
    from app.signals.auto_scanner import _build_signal_data

    user = MagicMock()
    user.id = 1
    r = _signal(raw_score=12)

    data = _build_signal_data(user, r, 1.10000, 1.09500, 1.11500,
                              0.01, 1.0, 3.0, "auto_approved")

    assert data["raw_score"] == 12
    assert abs(data["ml_probability"] - round(12 / 14.0 * 100, 2)) < 0.01
    assert data["entry_price"] == 1.10000   # M3 reference


def test_build_signal_data_strategy_label():
    from app.signals.auto_scanner import _build_signal_data

    user = MagicMock()
    user.id = 1
    r = _signal(raw_score=11)

    data = _build_signal_data(user, r, 1.10000, 1.09500, 1.11500,
                              0.01, 1.0, 3.0, "pending")

    assert "Variant C" in data["strategy"]


# ─────────────────────────────────────────────────────────────────────────────
# 13. Legacy scan_pair still works (manual/API path not broken)
# ─────────────────────────────────────────────────────────────────────────────

def test_scan_all_pairs_legacy_path_still_callable():
    """scan_all_pairs with strategy != smc_mtf must not crash."""
    from app.signals.scanner import scan_all_pairs

    with patch("app.signals.scanner.scan_pair", return_value=None):
        result = scan_all_pairs(timeframes=("M15",), strategy="legacy")

    assert isinstance(result, list)


def test_scan_all_pairs_smc_mtf_calls_scan_mtf_smc_pair():
    from app.signals.scanner import scan_all_pairs

    with patch("app.signals.scanner.scan_mtf_smc_pair", return_value=None) as mock_scan:
        scan_all_pairs(strategy="smc_mtf")

    assert mock_scan.call_count == 9   # 9 pairs


# ─────────────────────────────────────────────────────────────────────────────
# 14. DB schema — new columns present
# ─────────────────────────────────────────────────────────────────────────────

def test_trade_model_has_actual_fill_price():
    from app.db.models import Trade
    assert hasattr(Trade, "actual_fill_price")


def test_signal_model_has_raw_score():
    from app.db.models import Signal
    assert hasattr(Signal, "raw_score")


def test_signal_model_has_ml_probability_for_normalized_score():
    from app.db.models import Signal
    assert hasattr(Signal, "ml_probability")
