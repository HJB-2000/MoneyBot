import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
from brain.market_reader import MarketReader
from brain.signals.price_action import PriceActionSignal
from brain.signals.momentum import MomentumSignal
from brain.signals.microstructure import MicrostructureSignal
from brain.signals.volatility import VolatilitySignal
from brain.signals.sentiment import SentimentSignal
from brain.signals.cvd import CVDSignal
from brain.signals.vwap import VWAPSignal
from brain.signals.whale_detector import WhaleDetectorSignal


def test_all_signals():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    mr = MarketReader(config)
    sym = "SOL/USDT"
    candles = mr.get_candles(sym)
    ob = mr.get_orderbook(sym)
    trades = mr.get_trades(sym)
    funding = mr.get_funding_rate(sym) or 0.0
    oi = mr.get_open_interest(sym)

    signals = [
        ("price_action",   PriceActionSignal(),   lambda: PriceActionSignal().calculate(candles)),
        ("momentum",       MomentumSignal(),       lambda: MomentumSignal().calculate(candles)),
        ("microstructure", MicrostructureSignal(), lambda: MicrostructureSignal().calculate(ob, trades)),
        ("volatility",     VolatilitySignal(),     lambda: VolatilitySignal().calculate(candles)),
        ("sentiment",      SentimentSignal(),      lambda: SentimentSignal().calculate(funding, [], oi)),
        ("cvd",            CVDSignal(),            lambda: CVDSignal().calculate(trades, candles)),
        ("vwap",           VWAPSignal(),           lambda: VWAPSignal().calculate(candles, trades)),
        ("whale",          WhaleDetectorSignal(),  lambda: WhaleDetectorSignal().calculate(trades, ob)),
    ]

    print("\n=== Signal Tests ===")
    for name, obj, fn in signals:
        t0 = time.time()
        score = fn()
        ms = (time.time() - t0) * 1000

        assert -1.0 <= score <= 1.0, f"{name} score out of range: {score}"
        assert ms < 500, f"{name} too slow: {ms:.0f}ms"
        print(f"  {name:15} score={score:+.3f}  time={ms:.0f}ms  OK")

    # Test missing data returns 0.0
    assert PriceActionSignal().calculate(None) == 0.0
    assert MomentumSignal().calculate(None) == 0.0
    assert MicrostructureSignal().calculate(None, None) == 0.0
    assert VolatilitySignal().calculate(None) == 0.0
    print("  Missing data → 0.0  OK")

    # Test exposed attributes
    vol = VolatilitySignal()
    vol.calculate(candles)
    assert hasattr(vol, "atr_ratio"), "atr_ratio missing"
    assert hasattr(vol, "vol_spike"), "vol_spike missing"

    sent = SentimentSignal()
    sent.calculate(funding, [], oi)
    assert hasattr(sent, "funding_rate"), "funding_rate missing"

    cvd = CVDSignal()
    cvd.calculate(trades, candles)
    assert hasattr(cvd, "cvd_divergence"), "cvd_divergence missing"

    whale = WhaleDetectorSignal()
    whale.calculate(trades, ob)
    assert hasattr(whale, "whale_buying"), "whale_buying missing"
    assert hasattr(whale, "whale_selling"), "whale_selling missing"

    print("  Exposed attributes  OK")
    print("=== All signal tests PASSED ===\n")


if __name__ == "__main__":
    test_all_signals()
