from __future__ import annotations

import pandas as pd
import pytest

from brain.signals.cvd import CVDSignal
from brain.signals.microstructure import MicrostructureSignal
from brain.signals.momentum import MomentumSignal
from brain.signals.price_action import PriceActionSignal
from brain.signals.sentiment import SentimentSignal
from brain.signals.volatility import VolatilitySignal
from brain.signals.vwap import VWAPSignal
from brain.signals.whale_detector import WhaleDetectorSignal


def test_price_action_score_in_range(make_candles):
    candles = make_candles([100 + i * 0.1 for i in range(40)])
    score = PriceActionSignal().calculate(candles)
    assert -1.0 <= score <= 1.0


def test_price_action_returns_zero_on_empty_data():
    assert PriceActionSignal().calculate(None) == 0.0


def test_price_action_exposes_breakout_detected(make_candles):
    prices = [100.0] * 25 + [101.0]
    volumes = [100.0] * 25 + [500.0]
    candles = make_candles(prices, volumes=volumes)
    signal = PriceActionSignal()
    signal.calculate(candles)
    assert isinstance(signal.breakout_detected, bool)


def test_price_action_exposes_trend(make_candles):
    candles = make_candles([100 + i * 0.5 for i in range(40)])
    signal = PriceActionSignal()
    signal.calculate(candles)
    assert signal.trend in ["uptrend", "downtrend", "ranging"]


def test_price_action_consolidation_on_flat_data():
    import pandas as pd
    # Build candles with explicit tight H/L so range < 0.3% threshold
    prices = [100.0 + (i % 2) * 0.001 for i in range(40)]
    candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-04-08", periods=40, freq="5min"),
        "open": prices,
        "high": [p + 0.001 for p in prices],
        "low":  [p - 0.001 for p in prices],
        "close": prices,
        "volume": [100.0] * 40,
    })
    signal = PriceActionSignal()
    signal.calculate(candles)
    assert signal.consolidating == True


def test_momentum_score_in_range(make_candles):
    candles = make_candles([100 + i * 0.2 for i in range(60)])
    score = MomentumSignal().calculate(candles)
    assert -1.0 <= score <= 1.0


def test_rsi_oversold_gives_positive_score(make_candles):
    candles = make_candles([150 - i * 1.0 for i in range(30)] + [120.0] * 10)
    assert MomentumSignal().calculate(candles) > 0.0


def test_rsi_overbought_gives_negative_score(make_candles):
    candles = make_candles([50 + i * 1.0 for i in range(30)] + [80.0] * 10)
    assert MomentumSignal().calculate(candles) < 0.0


def test_macd_crossover_detected(make_candles):
    candles = make_candles([100 - i * 0.2 for i in range(35)] + [93 + i * 0.6 for i in range(10)])
    score = MomentumSignal().calculate(candles)
    assert score != 0.0


def test_momentum_returns_zero_on_insufficient_data(make_candles):
    candles = make_candles([100 + i for i in range(5)])
    assert MomentumSignal().calculate(candles) == 0.0


def test_microstructure_score_in_range(make_orderbook, make_trades):
    score = MicrostructureSignal().calculate(make_orderbook(), make_trades("buy", 5) + make_trades("sell", 5))
    assert -1.0 <= score <= 1.0


def test_ofi_balanced_gives_positive_score(make_orderbook, make_mixed_trades):
    score = MicrostructureSignal().calculate(make_orderbook(), make_mixed_trades(5, 5))
    assert score > 0.0


def test_ofi_heavy_buying_detected(make_orderbook, make_trades):
    score = MicrostructureSignal().calculate(make_orderbook(), make_trades("buy", 8) + make_trades("sell", 2))
    assert score > 0.0


def test_wide_spread_gives_negative_score(make_orderbook, make_trades):
    orderbook = make_orderbook(best_bid=100.0, best_ask=100.5)
    score = MicrostructureSignal().calculate(orderbook, make_trades("buy", 5) + make_trades("sell", 5))
    assert score < 0.0


def test_tight_spread_gives_positive_score(make_orderbook, make_trades):
    orderbook = make_orderbook(best_bid=100.0, best_ask=100.01)
    score = MicrostructureSignal().calculate(orderbook, make_trades("buy", 5) + make_trades("sell", 5))
    assert score > 0.0


def test_whale_detection_on_large_trade(make_orderbook):
    trades = [
        {"side": "buy", "price": 1000.0, "amount": 60.0, "timestamp": 1},
        {"side": "sell", "price": 1000.0, "amount": 1.0, "timestamp": 1},
    ]
    signal = MicrostructureSignal()
    signal.calculate(make_orderbook(), trades)
    assert signal.whale_detected is True


def test_whale_detection_absent_on_small_trades(make_orderbook):
    trades = [
        {"side": "buy", "price": 10.0, "amount": 1.0, "timestamp": 1},
        {"side": "sell", "price": 10.0, "amount": 1.0, "timestamp": 1},
    ]
    signal = MicrostructureSignal()
    signal.calculate(make_orderbook(), trades)
    assert signal.whale_detected is False


def test_volatility_score_in_range(make_candles):
    candles = make_candles([100 + ((-1) ** i) * 0.2 for i in range(60)])
    score = VolatilitySignal().calculate(candles)
    assert -1.0 <= score <= 1.0


def test_atr_ratio_exposed(make_candles):
    candles = make_candles([100 + ((-1) ** i) * 0.2 for i in range(60)])
    signal = VolatilitySignal()
    signal.calculate(candles)
    assert signal.atr_ratio > 0.0


def test_high_atr_ratio_gives_negative_score(make_candles):
    quiet = [100.0 + 0.01 * (i % 2) for i in range(20)]
    volatile = [100.0 + (i % 2) * 5.0 for i in range(20)]
    candles = make_candles(quiet + volatile)
    assert VolatilitySignal().calculate(candles) < 0.0


def test_low_atr_ratio_gives_positive_score(make_candles):
    volatile = [100.0 + (i % 2) * 5.0 for i in range(20)]
    quiet = [110.0 + 0.01 * (i % 2) for i in range(20)]
    candles = make_candles(volatile + quiet)
    assert VolatilitySignal().calculate(candles) > 0.0


def test_vol_spike_exposed(make_candles):
    candles = make_candles([100 + i * 0.1 for i in range(60)], volumes=[100.0] * 50 + [1000.0] * 10)
    signal = VolatilitySignal()
    signal.calculate(candles)
    assert isinstance(signal.vol_spike, bool)


def test_vol_spike_true_on_sudden_expansion(make_candles):
    candles = make_candles([100 + i * 0.1 for i in range(60)], volumes=[10.0] * 48 + [500.0] * 12)
    signal = VolatilitySignal()
    signal.calculate(candles)
    assert signal.vol_spike is True


def test_bollinger_squeeze_detected(make_candles):
    candles = make_candles([100.0 + (i % 3) * 0.005 for i in range(60)])
    signal = VolatilitySignal()
    signal.calculate(candles)
    assert signal.is_squeezing is True


def test_sentiment_score_in_range():
    score = SentimentSignal().calculate(0.0005, [], 1000.0)
    assert -1.0 <= score <= 1.0


def test_positive_funding_gives_positive_score():
    assert SentimentSignal().calculate(0.001, [], 1000.0) > 0.0


def test_negative_funding_gives_negative_score():
    assert SentimentSignal().calculate(-0.001, [], 1000.0) < 0.0


def test_funding_rate_exposed():
    signal = SentimentSignal()
    signal.calculate(0.001, [], 1000.0)
    assert isinstance(signal.funding_rate, float)


def test_liquidation_cascade_gives_negative_score():
    liquidations = [{"side": "long", "amount": 10.0}, {"side": "long", "amount": 9.0}, {"side": "short", "amount": 1.0}]
    assert SentimentSignal().calculate(0.0, liquidations, 1000.0) < -0.5


def test_sentiment_returns_zero_on_missing_data():
    assert SentimentSignal().calculate(None, None, None) == 0.0


def test_cvd_score_in_range(make_candles):
    candles = make_candles([100 + i * 0.2 for i in range(30)])
    trades = [{"side": "buy", "amount": 1.0} for _ in range(10)]
    score = CVDSignal().calculate(trades, candles)
    assert -1.0 <= score <= 1.0


def test_cvd_bullish_divergence_gives_strong_positive(make_candles):
    candles = make_candles([100 - i * 0.5 for i in range(30)])
    candles.loc[:, "close"] = candles.loc[:, "open"] + 0.1
    candles.iloc[-4:, candles.columns.get_loc("close")] = candles.iloc[-4:]["open"].values + 0.1
    score = CVDSignal().calculate([{"side": "buy", "amount": 1.0}], candles)
    assert score > 0.5


def test_cvd_bearish_divergence_gives_strong_negative(make_candles):
    candles = make_candles([100 + i * 0.5 for i in range(30)])
    candles.loc[:, "close"] = candles.loc[:, "open"] - 0.1
    score = CVDSignal().calculate([{"side": "sell", "amount": 1.0}], candles)
    assert score < -0.5


def test_cvd_confirmed_uptrend(make_candles):
    candles = make_candles([100 + i * 0.5 for i in range(30)])
    score = CVDSignal().calculate([{"side": "buy", "amount": 1.0}], candles)
    assert score > 0.0


def test_cvd_flat_gives_neutral_score(make_candles):
    candles = make_candles([100.0] * 30)
    score = CVDSignal().calculate([{"side": "buy", "amount": 1.0}], candles)
    assert -0.2 <= score <= 0.2


def test_vwap_score_in_range(make_candles):
    candles = make_candles([100 + i * 0.1 for i in range(30)])
    score = VWAPSignal().calculate(candles, [])
    assert -1.0 <= score <= 1.0


def test_vwap_price_exposed(make_candles):
    candles = make_candles([100 + i * 0.1 for i in range(30)])
    signal = VWAPSignal()
    signal.calculate(candles, [])
    assert signal.vwap_price > 0.0


def test_price_above_upper_band_negative_score(make_candles):
    candles = make_candles([100.0] * 29 + [110.0], volumes=[100.0] * 30)
    assert VWAPSignal().calculate(candles, []) < 0.0


def test_price_below_lower_band_positive_score(make_candles):
    candles = make_candles([100.0] * 29 + [90.0], volumes=[100.0] * 30)
    assert VWAPSignal().calculate(candles, []) > 0.0


def test_vwap_reclaim_gives_strong_positive(make_candles):
    prices = [100.0] * 28 + [99.0, 101.0]
    volumes = [100.0] * 29 + [1000.0]
    candles = make_candles(prices, volumes=volumes)
    assert VWAPSignal().calculate(candles, []) > 0.4


def test_vwap_resets_at_midnight(make_candles):
    _today_start = pd.Timestamp.now("UTC").normalize()
    _yesterday_start = _today_start - pd.Timedelta(days=1)
    yesterday = make_candles([80.0] * 20, start=_yesterday_start)
    today = make_candles([100.0, 101.0, 102.0, 103.0, 104.0], start=_today_start)
    candles = pd.concat([yesterday, today], ignore_index=True)
    signal = VWAPSignal()
    signal.calculate(candles, [])
    expected = today[["high", "low", "close", "volume"]].assign(
        typical=lambda df: (df["high"] + df["low"] + df["close"]) / 3
    )
    expected_vwap = (expected["typical"] * expected["volume"]).sum() / expected["volume"].sum()
    assert signal.vwap_price == pytest.approx(expected_vwap)


def test_whale_detector_score_in_range(make_orderbook):
    trades = [{"side": "buy", "price": 1000.0, "amount": 60.0, "timestamp": 1}]
    score = WhaleDetectorSignal().calculate(trades, make_orderbook())
    assert -1.0 <= score <= 1.0


def test_large_buy_orders_give_positive_score(make_orderbook):
    trades = [{"side": "buy", "price": 1000.0, "amount": 60.0, "timestamp": 1} for _ in range(2)]
    signal = WhaleDetectorSignal()
    assert signal.calculate(trades, make_orderbook()) > 0.0
    assert signal.whale_buying is True


def test_large_sell_orders_give_negative_score(make_orderbook):
    trades = [{"side": "sell", "price": 1000.0, "amount": 60.0, "timestamp": 1} for _ in range(2)]
    signal = WhaleDetectorSignal()
    assert signal.calculate(trades, make_orderbook()) < 0.0
    assert signal.whale_selling is True


def test_iceberg_bid_detected(make_orderbook):
    signal = WhaleDetectorSignal()
    for _ in range(3):
        signal.calculate([{"side": "buy", "price": 10.0, "amount": 1.0, "timestamp": 1}], make_orderbook())
    assert signal.iceberg_detected is True


def test_spoofing_detected_on_disappearing_order(make_orderbook):
    signal = WhaleDetectorSignal()
    snapshots = [
        make_orderbook(best_bid=100.0, best_ask=100.02),
        make_orderbook(best_bid=100.0, best_ask=100.02),
        make_orderbook(best_bid=100.0, best_ask=100.02),
        make_orderbook(best_bid=99.5, best_ask=100.02),
    ]
    score = 0.0
    for snap in snapshots:
        score = signal.calculate([{"side": "buy", "price": 10.0, "amount": 1.0, "timestamp": 1}], snap)
    assert score != 0.0


def test_whale_detector_exposes_whale_buying(make_orderbook):
    signal = WhaleDetectorSignal()
    signal.calculate([{"side": "buy", "price": 1000.0, "amount": 60.0, "timestamp": 1}], make_orderbook())
    assert isinstance(signal.whale_buying, bool)
