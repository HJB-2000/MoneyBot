from __future__ import annotations

import time

import pytest


pytest.importorskip("ccxt")

import ccxt
import brain.market_reader as market_reader_module
from brain.market_reader import MarketReader


class FakeExchange:
    def __init__(self):
        self.calls = 0

    def fetch_order_book(self, symbol, limit):
        self.calls += 1
        if symbol == "FAKE/USDT":
            raise ccxt.ExchangeError("bad symbol")
        return {
            "bids": [(100.0, 10.0), (99.9, 9.0), (99.8, 8.0)],
            "asks": [(100.02, 10.0), (100.03, 9.0), (100.04, 8.0)],
        }

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        base = int(time.time() * 1000)
        return [[base + i * 300000, 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i] for i in range(limit)]

    def fetch_trades(self, symbol, limit=200):
        return [{"side": "buy", "amount": 1.0, "price": 100.0, "timestamp": int(time.time() * 1000)} for _ in range(limit)]

    def fetch_funding_rate(self, symbol):
        return {"fundingRate": 0.0005}

    def fetch_open_interest(self, symbol):
        return {"openInterest": 12345.0}

    def fetch_ticker(self, symbol):
        return {"quoteVolume": 999.0}


def patch_market_reader(monkeypatch):
    fake_exchange = FakeExchange()
    monkeypatch.setattr(market_reader_module.ccxt, "binance", lambda *args, **kwargs: fake_exchange)
    return fake_exchange


def test_orderbook_returns_bids_and_asks(config, monkeypatch):
    fake_exchange = patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    orderbook = reader.get_orderbook("SOL/USDT")
    assert orderbook["bids"] and orderbook["asks"]
    assert fake_exchange.calls == 1


def test_orderbook_bids_sorted_descending(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    orderbook = reader.get_orderbook("SOL/USDT")
    assert orderbook["bids"][0][0] > orderbook["bids"][1][0]


def test_orderbook_asks_sorted_ascending(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    orderbook = reader.get_orderbook("SOL/USDT")
    assert orderbook["asks"][0][0] < orderbook["asks"][1][0]


def test_candles_returns_correct_shape(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    candles = reader.get_candles("SOL/USDT", "5m", limit=100)
    assert candles.shape == (100, 6)


def test_candles_no_gaps(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    candles = reader.get_candles("SOL/USDT", "5m", limit=100)
    diffs = candles["timestamp"].diff().dropna().dt.total_seconds().unique()
    assert len(diffs) == 1 and diffs[0] == 300.0


def test_funding_rate_returns_float(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    assert isinstance(reader.get_funding_rate("SOL/USDT"), float)


def test_api_latency_tracked(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    reader.get_orderbook("SOL/USDT")
    assert reader.avg_latency_ms > 0


def test_cache_returns_same_data_within_5s(config, monkeypatch):
    fake_exchange = patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    first = reader.get_orderbook("SOL/USDT")
    second = reader.get_orderbook("SOL/USDT")
    assert first == second
    assert fake_exchange.calls == 1


def test_graceful_handling_of_bad_symbol(config, monkeypatch):
    patch_market_reader(monkeypatch)
    reader = MarketReader(config)
    assert reader.get_orderbook("FAKE/USDT") is None


def test_graceful_handling_of_timeout(config, monkeypatch):
    fake_exchange = patch_market_reader(monkeypatch)

    def boom(*args, **kwargs):
        raise ccxt.NetworkError("timeout")

    fake_exchange.fetch_order_book = boom
    reader = MarketReader(config)
    assert reader.get_orderbook("SOL/USDT") is None
