import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from strategies.triangular_arb import TriangularArbStrategy
from strategies.stat_arb import StatArbStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.volume_spike import VolumeSpikeStrategy


def make_candles(n=100, price=100.0, trend="flat"):
    prices = []
    for i in range(n):
        if trend == "flat":
            p = price + np.random.randn() * 0.1
        elif trend == "up":
            p = price + i * 0.01 + np.random.randn() * 0.05
        else:
            p = price - i * 0.01 + np.random.randn() * 0.05
        prices.append(p)
    closes = np.array(prices)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open": closes * 0.999,
        "high": closes * 1.002,
        "low": closes * 0.998,
        "close": closes,
        "volume": np.random.uniform(100, 1000, n),
    })


def make_orderbook(bid=99.9, ask=100.1, depth=1000):
    return {
        "bids": [[bid - i * 0.01, depth] for i in range(20)],
        "asks": [[ask + i * 0.01, depth] for i in range(20)],
    }


def make_mr(candles=None, ob=None, ticker=None):
    mr = MagicMock()
    mr.get_candles.return_value = candles if candles is not None else make_candles()
    mr.get_orderbook.return_value = ob or make_orderbook()
    mr.get_ticker.return_value = ticker or {"last": 100.0, "quoteVolume": 1e6}
    mr.avg_latency_ms = 50.0
    return mr


def make_signal_objects(arb_friendly=True):
    cr = MagicMock()
    cr.arb_friendly = arb_friendly
    cr.confidence = 0.75
    return {"_combiner_result": cr, "_route_result": MagicMock(bias=None)}


def test_strategies():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    print("\n=== Strategy Tests ===")

    # TriangularArb: scan returns list
    strat = TriangularArbStrategy()
    ob = make_orderbook(99.9, 100.1, 10000)
    mr = make_mr(ob=ob)
    result = strat.scan("RANGING", 1.0, {}, make_signal_objects(), mr, 50.0, config)
    assert isinstance(result, list), "scan() must return list"
    print(f"  triangular_arb       scan() returns list  len={len(result)}  PASS")

    # TriangularArb: returns [] in VOLATILE regime
    result_vol = strat.scan("VOLATILE", 1.0, {}, make_signal_objects(), mr, 50.0, config)
    assert result_vol == [], "Should return [] in VOLATILE"
    print(f"  triangular_arb       VOLATILE → []  PASS")

    # MeanReversion: returns list
    candles_drop = make_candles(30, 100.0)
    # Simulate a drop: last candle price is lower
    candles_drop.loc[candles_drop.index[-1], "close"] = 97.0  # 3% drop
    mr2 = make_mr(candles=candles_drop)
    strat2 = MeanReversionStrategy()
    result2 = strat2.scan("RANGING", 1.0, {}, make_signal_objects(), mr2, 50.0, config)
    assert isinstance(result2, list)
    print(f"  mean_reversion       scan() returns list  len={len(result2)}  PASS")

    # VolumeSpike: returns list
    candles_spike = make_candles(50, 100.0)
    candles_spike.loc[candles_spike.index[-1], "volume"] = 50000  # huge spike
    candles_spike.loc[candles_spike.index[-1], "close"] = 101.5   # price moved up
    mr3 = make_mr(candles=candles_spike)
    strat3 = VolumeSpikeStrategy()
    signal_objs = make_signal_objects()
    result3 = strat3.scan("TRENDING_UP", 0.6, {"cvd": 0.3}, signal_objs, mr3, 50.0, config)
    assert isinstance(result3, list)
    print(f"  volume_spike         scan() returns list  len={len(result3)}  PASS")

    # All Opportunity fields populated check
    if result3:
        opp = result3[0]
        assert opp.strategy, "strategy empty"
        assert opp.pair, "pair empty"
        assert opp.entry_price > 0, "entry_price must be > 0"
        assert opp.trade_size_usd > 0, "trade_size_usd must be > 0"
        assert opp.detected_at is not None, "detected_at missing"
        print(f"  Opportunity fields   populated  PASS")

    print("=== All strategy tests PASSED ===\n")


if __name__ == "__main__":
    test_strategies()
