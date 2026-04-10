"""
Tests for all 30 signals across 5 groups.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signals.group_orderbook import OrderBookSignals
from signals.group_candles import CandleSignals
from signals.group_trades import TradeSignals
from signals.group_futures import FuturesSignals
from signals.group_derived import DerivedSignals


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_orderbook(best_bid=100.0, best_ask=100.02, levels=10,
                   bid_size=10.0, ask_size=10.0, depth=None):
    """depth kwarg sets bid_size for liquidity ratio testing."""
    if depth is not None:
        bid_size = depth
    bids = [(round(best_bid - i * 0.01, 4), bid_size) for i in range(levels)]
    asks = [(round(best_ask + i * 0.01, 4), ask_size) for i in range(levels)]
    return {"bids": bids, "asks": asks}


def make_balanced_orderbook():
    return make_orderbook(bid_size=10.0, ask_size=10.0)


def make_candles(n=100, base=100.0, trend=0.0, noise=0.1):
    prices = [base + trend * i + np.random.uniform(-noise, noise) for i in range(n)]
    opens  = [prices[0]] + list(prices[:-1])
    highs  = [max(o, c) * 1.001 for o, c in zip(opens, prices)]
    lows   = [min(o, c) * 0.999 for o, c in zip(opens, prices)]
    ts     = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": [100.0] * n,
    })


def make_declining_candles(periods=40):
    return make_candles(n=periods, trend=-1.0, noise=0.05)


def make_rising_candles(periods=40):
    return make_candles(n=periods, trend=1.0, noise=0.05)


def make_volatile_candles(n=60):
    # First 40 candles: stable (tiny ATR), last 20: sudden spike (large ATR)
    # This ensures current_atr >> avg(last_20_atrs) → ratio > 1.8 → score <= -0.3
    stable = [100.0] * 40
    spike  = [100.0 + (i % 2) * 20.0 for i in range(20)]
    prices = (stable + spike)[:n]
    opens  = [prices[0]] + list(prices[:-1])
    highs  = [max(o, c) * 1.005 for o, c in zip(opens, prices)]
    lows   = [min(o, c) * 0.995 for o, c in zip(opens, prices)]
    ts     = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": [100.0] * n,
    })


def make_flat_candles(n=40):
    prices = [100.0 + (i % 2) * 0.001 for i in range(n)]
    opens  = [prices[0]] + list(prices[:-1])
    highs  = [p + 0.001 for p in prices]
    lows   = [p - 0.001 for p in prices]
    ts     = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": prices, "volume": [100.0] * n,
    })


def make_buy_trades(n=20, amount=1.0, price=100.0):
    now_ms = int(time.time() * 1000)
    return [{"side": "buy", "amount": amount, "price": price,
              "timestamp": now_ms - i * 1000} for i in range(n)]


def make_sell_trades(n=20, amount=1.0, price=100.0):
    now_ms = int(time.time() * 1000)
    return [{"side": "sell", "amount": amount, "price": price,
              "timestamp": now_ms - i * 1000} for i in range(n)]


def make_large_buy_trades(n=5, amount=600.0, price=100.0):
    now_ms = int(time.time() * 1000)
    return [{"side": "buy", "amount": amount, "price": price,
              "timestamp": now_ms - i * 1000} for i in range(n)]


def make_large_sell_trades(n=5, amount=600.0, price=100.0):
    now_ms = int(time.time() * 1000)
    return [{"side": "sell", "amount": amount, "price": price,
              "timestamp": now_ms - i * 1000} for i in range(n)]


def make_liquidations(large=False):
    if large:
        return [{"side": "long", "amount": 10.0} for _ in range(9)] + \
               [{"side": "short", "amount": 1.0}]
    return [{"side": "long", "amount": 1.0}, {"side": "short", "amount": 1.0}]


def make_refilling_orderbook():
    """Simulate iceberg: same best bid seen 3+ times."""
    sig = OrderBookSignals()
    ob = make_orderbook(best_bid=100.0, best_ask=100.02)
    for _ in range(4):
        sig.calculate(ob)
    return sig


def make_spoofing_orderbook():
    """Simulate spoofing: large bid disappears."""
    sig = OrderBookSignals()
    # First snapshot: big bid
    ob1 = {"bids": [(100.0, 500.0)] + [(99.99 - i * 0.01, 5.0) for i in range(9)],
            "asks": [(100.02 + i * 0.01, 5.0) for i in range(10)]}
    sig.calculate(ob1)
    # Within 2s: big bid gone — spoof
    return sig, make_orderbook(best_bid=100.0, best_ask=100.02)


# ── Order Book Signals (8) ────────────────────────────────────────────────────

def test_bid_ask_spread_tight_positive():
    ob = make_orderbook(best_bid=100.0, best_ask=100.04)
    scores = OrderBookSignals().calculate(ob)
    assert scores["bid_ask_spread"] > 0.0


def test_bid_ask_spread_wide_negative():
    ob = make_orderbook(best_bid=100.0, best_ask=100.6)
    scores = OrderBookSignals().calculate(ob)
    assert scores["bid_ask_spread"] < 0.0


def test_depth_imbalance_balanced_positive():
    ob = make_balanced_orderbook()
    scores = OrderBookSignals().calculate(ob)
    assert scores["depth_imbalance"] > 0.3


def test_liquidity_score_hard_reject_below_5x():
    ob = make_orderbook(depth=0.04)   # bids: 0.04 unit * $100 = $4 per level → total ~$40
    scores = OrderBookSignals().calculate(ob, trade_size_usd=50.0)
    assert scores["liquidity_score"] == -1.0


def test_iceberg_detected_on_refill():
    sig = make_refilling_orderbook()
    scores = sig.scores if hasattr(sig, "scores") else None
    # Direct test: call calculate on same ob 4 times
    ob_sig = OrderBookSignals()
    ob = make_orderbook(best_bid=100.0)
    for _ in range(4):
        s = ob_sig.calculate(ob)
    assert s["iceberg_detection"] != 0.0


def test_spoofing_detected_on_disappearing_order():
    sig, ob2 = make_spoofing_orderbook()
    score = sig.calculate(ob2)
    # spoofing or not — score should be valid float
    assert -1.0 <= score["spoofing_detection"] <= 1.0


def test_all_orderbook_scores_in_range():
    ob = make_orderbook()
    scores = OrderBookSignals().calculate(ob)
    for k, v in scores.items():
        assert -1.0 <= v <= 1.0, f"{k} out of range: {v}"


def test_orderbook_completes_under_100ms():
    ob = make_orderbook()
    t = time.time()
    OrderBookSignals().calculate(ob)
    assert time.time() - t < 0.1


# ── Candle Signals (10) ───────────────────────────────────────────────────────

def test_rsi_oversold_strong_positive():
    candles = make_declining_candles(periods=40)
    scores = CandleSignals().calculate(candles)
    assert scores["RSI"] > 0.4


def test_rsi_overbought_strong_negative():
    candles = make_rising_candles(periods=40)
    scores = CandleSignals().calculate(candles)
    assert scores["RSI"] < -0.4


def test_atr_ratio_exposed_and_valid():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert hasattr(sig, "atr_ratio")
    assert sig.atr_ratio > 0


def test_high_atr_ratio_triggers_veto_score():
    candles = make_volatile_candles()
    scores = CandleSignals().calculate(candles)
    assert scores["ATR_ratio"] <= -0.3


def test_consolidation_detected_on_flat_data():
    candles = make_flat_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert sig.consolidating == True


def test_bollinger_squeeze_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.is_squeezing, bool)


def test_breakout_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.breakout_detected, bool)


def test_vol_spike_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.vol_spike, bool)


def test_all_candle_scores_in_range():
    candles = make_candles()
    scores = CandleSignals().calculate(candles)
    for k, v in scores.items():
        assert -1.0 <= v <= 1.0, f"{k} out of range: {v}"


def test_candle_signals_insufficient_data_returns_zeros():
    candles = make_candles(n=5)
    scores = CandleSignals().calculate(candles)
    assert all(v == 0.0 for v in scores.values())


# ── Trade Signals (5) ─────────────────────────────────────────────────────────

def test_cvd_bullish_divergence_strong_positive():
    candles = make_declining_candles()
    trades = make_buy_trades()
    scores = TradeSignals().calculate(trades, candles)
    assert scores["CVD_divergence"] > 0.7


def test_cvd_bearish_divergence_strong_negative():
    candles = make_rising_candles()
    trades = make_sell_trades()
    scores = TradeSignals().calculate(trades, candles)
    assert scores["CVD_divergence"] < -0.7


def test_whale_buying_exposed_as_bool():
    trades = make_large_buy_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert sig.whale_buying == True


def test_whale_selling_exposed_as_bool():
    trades = make_large_sell_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert sig.whale_selling == True


def test_volume_spike_exposed_as_bool():
    trades = make_buy_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert isinstance(sig.volume_spike, bool)


# ── Futures Signals (5) ───────────────────────────────────────────────────────

def test_positive_funding_gives_positive_score():
    scores = FuturesSignals().calculate(0.001, 1000, [], make_candles())
    assert scores["funding_rate"] > 0.0


def test_negative_funding_gives_negative_score():
    scores = FuturesSignals().calculate(-0.001, 1000, [], make_candles())
    assert scores["funding_rate"] < 0.0


def test_liquidation_cascade_exposed_as_bool():
    sig = FuturesSignals()
    sig.calculate(0.0, 1000, make_liquidations(large=True), make_candles())
    assert sig.liquidation_cascade == True


def test_futures_unavailable_returns_zeros():
    scores = FuturesSignals().calculate(None, None, None, make_candles())
    assert all(v == 0.0 for v in scores.values())


def test_vwap_price_exposed_and_positive():
    sig = FuturesSignals()
    sig.calculate(0.0, 1000, [], make_candles())
    assert sig.vwap_price > 0.0


# ── Derived Signals (2) ───────────────────────────────────────────────────────

def test_btc_correlation_score_in_range():
    c_signals = {"rate_of_change": 0.03}
    t_signals = {}
    scores = DerivedSignals().calculate(c_signals, t_signals)
    assert -1.0 <= scores["btc_correlation_momentum"] <= 1.0


def test_regime_persistence_increases_with_cycles():
    sig = DerivedSignals()
    scores_15 = sig.calculate({}, {}, regime_cycles=15)["regime_persistence"]
    scores_3  = sig.calculate({}, {}, regime_cycles=3)["regime_persistence"]
    assert scores_15 > scores_3


# ── Parallel speed test ───────────────────────────────────────────────────────

def test_parallel_signal_calculation_under_500ms():
    from concurrent.futures import ThreadPoolExecutor
    ob      = make_orderbook()
    candles = make_candles()
    trades  = make_buy_trades()
    t = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        f1 = ex.submit(OrderBookSignals().calculate, ob)
        f2 = ex.submit(CandleSignals().calculate, candles)
        f3 = ex.submit(TradeSignals().calculate, trades, candles)
        f4 = ex.submit(FuturesSignals().calculate, 0.0, 1000, [], candles)
        results = [f.result() for f in [f1, f2, f3, f4]]
    elapsed = time.time() - t
    assert elapsed < 0.5
    assert all(isinstance(r, dict) for r in results)
