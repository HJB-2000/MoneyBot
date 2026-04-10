from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def config() -> dict:
    with open(ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "daily").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def make_candles():
    def _make_candles(
        prices,
        volumes=None,
        start: str | pd.Timestamp = "2026-04-08T00:00:00Z",
        freq: str = "5min",
    ):
        if volumes is None:
            volumes = [100.0] * len(prices)
        if len(volumes) != len(prices):
            raise ValueError("prices and volumes must match")

        timestamps = pd.date_range(start=start, periods=len(prices), freq=freq)
        opens = [prices[0]] + list(prices[:-1])
        highs = [max(o, c) * 1.002 for o, c in zip(opens, prices)]
        lows = [min(o, c) * 0.998 for o, c in zip(opens, prices)]

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            }
        )

    return _make_candles


@pytest.fixture()
def make_orderbook():
    def _make_orderbook(best_bid=100.0, best_ask=100.02, levels=10, bid_size=10.0, ask_size=10.0):
        bids = [(round(best_bid - i * 0.01, 4), bid_size) for i in range(levels)]
        asks = [(round(best_ask + i * 0.01, 4), ask_size) for i in range(levels)]
        return {"bids": bids, "asks": asks}

    return _make_orderbook


@pytest.fixture()
def make_trades():
    def _make_trades(side: str, count: int, price: float = 100.0, amount: float = 1.0, start_ts: int | None = None):
        if start_ts is None:
            start_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        trades = []
        for i in range(count):
            trades.append(
                {
                    "side": side,
                    "price": price,
                    "amount": amount,
                    "timestamp": start_ts - i * 1000,
                }
            )
        return trades

    return _make_trades


@pytest.fixture()
def make_mixed_trades(make_trades):
    def _make_mixed_trades(buys: int, sells: int, buy_amount: float = 1.0, sell_amount: float = 1.0, price: float = 100.0):
        return make_trades("buy", buys, price=price, amount=buy_amount) + make_trades("sell", sells, price=price, amount=sell_amount)

    return _make_mixed_trades
