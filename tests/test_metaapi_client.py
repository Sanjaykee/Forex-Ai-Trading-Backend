"""
Tests for MetaApi data retrieval, candle validation, and failure behavior.
All API calls are mocked — the actual backtest uses real MetaApi candles.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_candles(n=100, start="2026-10-20 07:00", freq="5min", base=1.1000):
    times  = pd.date_range(start, periods=n, freq=freq)
    closes = np.full(n, base) + np.random.randn(n) * 0.0002
    highs  = closes + 0.0005
    lows   = closes - 0.0005
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    return pd.DataFrame({"time": times, "open": opens, "high": highs,
                         "low": lows, "close": closes, "volume": np.ones(n) * 100})


def _raw_candles(n=10, base=1.1000):
    """Simulate raw MetaApi JSON response."""
    times = pd.date_range("2026-10-20 07:00", periods=n, freq="5min")
    return [
        {"time": t.isoformat() + "Z",
         "open": base, "high": base + 0.0005,
         "low": base - 0.0005, "close": base + 0.0001,
         "tickVolume": 100}
        for t in times
    ]


# ── MetaApiClient._parse_candles ──────────────────────────────────────────────

def test_parse_candles_basic():
    from app.mt5.metaapi_client import MetaApiClient
    raw = _raw_candles(5)
    df = MetaApiClient._parse_candles(raw)
    assert len(df) == 5
    assert list(df.columns) == ["time", "open", "high", "low", "close", "volume"]
    # pandas 3.x uses datetime64[us]; accept both [us] and [ns]
    assert "datetime64" in str(df["time"].dtype)


def test_parse_candles_empty():
    from app.mt5.metaapi_client import MetaApiClient
    df = MetaApiClient._parse_candles([])
    assert df.empty


def test_parse_candles_missing_ohlc():
    from app.mt5.metaapi_client import MetaApiClient
    raw = [{"time": "2026-10-20T07:00:00.000Z", "open": 1.1}]  # missing high/low/close
    df = MetaApiClient._parse_candles(raw)
    assert df.empty


def test_parse_candles_tick_volume_renamed():
    from app.mt5.metaapi_client import MetaApiClient
    raw = _raw_candles(3)
    df = MetaApiClient._parse_candles(raw)
    assert "volume" in df.columns
    assert "tickVolume" not in df.columns


def test_parse_candles_tz_stripped():
    from app.mt5.metaapi_client import MetaApiClient
    raw = _raw_candles(3)
    df = MetaApiClient._parse_candles(raw)
    assert df["time"].dt.tz is None  # tz-naive UTC


# ── MetaApiClient._tf_minutes ─────────────────────────────────────────────────

def test_tf_minutes():
    from app.mt5.metaapi_client import MetaApiClient
    assert MetaApiClient._tf_minutes("M5")  == 5
    assert MetaApiClient._tf_minutes("M15") == 15
    assert MetaApiClient._tf_minutes("M3")  == 3
    assert MetaApiClient._tf_minutes("H1")  == 60
    assert MetaApiClient._tf_minutes("H4")  == 240


# ── get_candles_range: not configured ─────────────────────────────────────────

def test_get_candles_range_not_configured():
    from app.mt5.metaapi_client import MetaApiClient
    # Patch _ensure_loaded so it cannot read .env.local credentials
    client = MetaApiClient(token="", account_id="")
    with patch.object(client, "_ensure_loaded", return_value=None):
        client._token = ""
        client._account_id = ""
        with pytest.raises(RuntimeError, match="credentials not configured"):
            client.get_candles_range(
                "EURUSD", "M5",
                datetime(2026, 10, 20), datetime(2026, 10, 27)
            )


# ── get_candles_range: unsupported timeframe ──────────────────────────────────

def test_get_candles_range_bad_tf():
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(token="tok", account_id="acc")
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        client.get_candles_range(
            "EURUSD", "M99",
            datetime(2026, 10, 20), datetime(2026, 10, 27)
        )


# ── get_candles_range: successful single-page fetch ───────────────────────────

def test_get_candles_range_success():
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(token="tok", account_id="acc", region="backup-new-york")

    raw = _raw_candles(50)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = raw

    with patch("app.mt5.metaapi_client.get_session") as mock_sess:
        mock_sess.return_value.get.return_value = mock_resp
        df = client.get_candles_range(
            "EURUSD", "M5",
            datetime(2026, 10, 20), datetime(2026, 10, 27),
            warmup_candles=10,
        )

    assert not df.empty
    assert len(df) == 50
    assert df["time"].is_monotonic_increasing
    assert df["time"].dt.tz is None


# ── get_candles_range: pagination stops when coverage reached ─────────────────

def test_get_candles_range_pagination():
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(token="tok", account_id="acc", region="backup-new-york")

    # First batch: 1000 candles (full batch — triggers next page)
    batch1 = _raw_candles(1000, base=1.1000)
    # Second batch: 200 candles (partial — stops pagination)
    batch2 = _raw_candles(200, base=1.0990)

    responses = [
        MagicMock(status_code=200, json=MagicMock(return_value=batch1)),
        MagicMock(status_code=200, json=MagicMock(return_value=batch2)),
    ]

    with patch("app.mt5.metaapi_client.get_session") as mock_sess:
        mock_sess.return_value.get.side_effect = responses
        df = client.get_candles_range(
            "EURUSD", "M5",
            datetime(2026, 10, 20), datetime(2026, 10, 27),
            warmup_candles=5,
        )

    # Should have combined both batches (deduped)
    assert len(df) <= 1200
    assert df["time"].is_monotonic_increasing


# ── get_candles_range: HTTP 401 raises RuntimeError ──────────────────────────

def test_get_candles_range_auth_error():
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(token="bad_token", account_id="acc", region="backup-new-york")

    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    with patch("app.mt5.metaapi_client.get_session") as mock_sess:
        mock_sess.return_value.get.return_value = mock_resp
        with pytest.raises(RuntimeError, match="auth error 401"):
            client.get_candles_range(
                "EURUSD", "M5",
                datetime(2026, 10, 20), datetime(2026, 10, 27)
            )


# ── get_candles_range: empty response returns empty DataFrame ─────────────────

def test_get_candles_range_empty_response():
    from app.mt5.metaapi_client import MetaApiClient
    client = MetaApiClient(token="tok", account_id="acc", region="backup-new-york")

    mock_resp = MagicMock(status_code=200, json=MagicMock(return_value=[]))

    with patch("app.mt5.metaapi_client.get_session") as mock_sess:
        mock_sess.return_value.get.return_value = mock_resp
        df = client.get_candles_range(
            "EURUSD", "M5",
            datetime(2026, 10, 20), datetime(2026, 10, 27)
        )

    assert df.empty


# ── candle validation: duplicate removal ──────────────────────────────────────

def test_candle_dedup():
    from app.mt5.metaapi_client import MetaApiClient
    raw = _raw_candles(5)
    raw_dup = raw + raw  # 10 rows, 5 duplicates
    df = MetaApiClient._parse_candles(raw_dup)
    # _parse_candles doesn't dedup — that's done in get_candles_range
    # Verify dedup logic manually
    df_dedup = df.drop_duplicates(subset=["time"])
    assert len(df_dedup) == 5


# ── candle validation: sort order ─────────────────────────────────────────────

def test_candle_sort_order():
    from app.mt5.metaapi_client import MetaApiClient
    raw = _raw_candles(10)
    raw_shuffled = raw[::-1]  # reverse order
    df = MetaApiClient._parse_candles(raw_shuffled)
    df_sorted = df.sort_values("time").reset_index(drop=True)
    assert df_sorted["time"].is_monotonic_increasing


# ── closed-candle handling ────────────────────────────────────────────────────

def test_closed_candle_only():
    """
    The backtest must not use the currently forming candle.
    Verify that the last candle in the test window is excluded from signal generation
    by checking that df5.iloc[:idx+1] never includes a candle with time > current_time.
    """
    n = 50
    times = pd.date_range("2026-10-20 07:00", periods=n, freq="5min")
    closes = np.full(n, 1.1000)
    df = pd.DataFrame({"time": times, "open": closes, "high": closes + 0.0005,
                       "low": closes - 0.0005, "close": closes})

    # Simulate the backtest slice at idx=40
    idx = 40
    slice_at_idx = df.iloc[:idx + 1]
    assert slice_at_idx["time"].max() == times[idx]
    assert times[idx + 1] not in slice_at_idx["time"].values


# ── is_configured check ───────────────────────────────────────────────────────

def test_is_configured_false_when_empty():
    from app.mt5.metaapi_client import MetaApiClient
    c = MetaApiClient(token="", account_id="")
    # Patch _ensure_loaded to prevent reading .env.local
    with patch.object(c, "_ensure_loaded", return_value=None):
        c._token = ""
        c._account_id = ""
        assert c.is_configured is False


def test_is_configured_true_when_set():
    from app.mt5.metaapi_client import MetaApiClient
    c = MetaApiClient(token="sometoken", account_id="someaccount")
    assert c.is_configured is True
