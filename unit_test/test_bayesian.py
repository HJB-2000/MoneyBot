"""
Tests for the Bayesian network and its components.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesian.network import BayesianNetwork
from bayesian.correlation_filter import CorrelationFilter
from bayesian.threshold_manager import ThresholdManager
from bayesian.likelihood_engine import LikelihoodEngine


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_all_signals(override: dict = None) -> dict:
    signals = {
        "RSI": 0.2, "MACD": 0.1, "EMA_cross": 0.1, "ATR_ratio": 0.9,
        "bollinger_state": 0.1, "rate_of_change": 0.1,
        "support_resistance": 0.1, "trend_structure": 0.2,
        "consolidation_score": 0.3, "realized_vol": 0.2,
        "CVD_divergence": 0.3, "CVD_trend": 0.1,
        "volume_spike_signal": 0.1, "buy_sell_ratio": 0.2,
        "large_trade_flow": 0.1, "bid_ask_spread": 0.5,
        "depth_imbalance": 0.3, "order_flow_imbalance": 0.4,
        "large_order_presence": 0.1, "iceberg_detection": 0.0,
        "spoofing_detection": 0.0, "liquidity_score": 0.5,
        "book_pressure_ratio": 0.1, "funding_rate": 0.1,
        "open_interest_change": 0.1, "liquidation_pressure": 0.0,
        "vwap_position": 0.2, "vwap_reclaim": 0.0,
        "btc_correlation_momentum": 0.0, "regime_persistence": 0.2,
        "liquidation_cascade": False, "whale_buying": False, "whale_selling": False,
    }
    if override:
        signals.update(override)
    return signals


def make_neutral_signals() -> dict:
    s = {k: 0.0 for k in make_all_signals()}
    s["liquidation_cascade"] = False
    s["whale_buying"] = False
    s["whale_selling"] = False
    return s


def make_strong_positive_signals() -> dict:
    s = make_all_signals()
    for k in s:
        if isinstance(s[k], float):
            s[k] = 0.7
    return s


def make_mock_trade_history(n=100):
    return [
        {
            "strategy": "triangular_arb",
            "result": "WIN" if i % 2 == 0 else "LOSS",
            "timestamp": f"2026-01-{(i%28)+1:02d}T00:00:00",
            "net_pnl_usd": 0.1,
        }
        for i in range(n)
    ]


def make_mock_signal_history(n=100):
    return [
        {
            "timestamp": f"2026-01-{(i%28)+1:02d}T00:00:00",
            "RSI": 0.3,
            "bid_ask_spread": 0.5,
            "ATR_ratio": 0.8,
        }
        for i in range(n)
    ]


# ── BayesianNetwork tests ─────────────────────────────────────────────────────

def test_bayesian_returns_probability_in_range():
    network = BayesianNetwork()
    P = network.compute(make_all_signals(), "triangular_arb", "RANGING")
    assert 0.0 <= P <= 1.0


def test_bayesian_veto_on_volatile_regime():
    network = BayesianNetwork()
    P = network.compute(make_all_signals(), "triangular_arb", "VOLATILE")
    assert P == 0.0


def test_bayesian_veto_on_high_atr():
    network = BayesianNetwork()
    signals = make_all_signals({"ATR_ratio": 2.5})
    P = network.compute(signals, "triangular_arb", "RANGING")
    assert P == 0.0


def test_bayesian_veto_on_liquidation_cascade():
    network = BayesianNetwork()
    signals = make_all_signals({"liquidation_cascade": True})
    P = network.compute(signals, "mean_reversion", "RANGING")
    assert P == 0.0


def test_bayesian_higher_probability_with_strong_signals():
    network = BayesianNetwork()
    P_weak   = network.compute(make_neutral_signals(), "triangular_arb", "RANGING")
    P_strong = network.compute(make_strong_positive_signals(), "triangular_arb", "RANGING")
    assert P_strong > P_weak


def test_bayesian_lower_probability_with_contradicting_signals():
    network = BayesianNetwork()
    positive    = make_strong_positive_signals()
    contradicted = positive.copy()
    contradicted["CVD_divergence"] = -0.9
    contradicted["depth_imbalance"] = -0.7
    P_pos = network.compute(positive,    "triangular_arb", "RANGING")
    P_con = network.compute(contradicted, "triangular_arb", "RANGING")
    assert P_pos > P_con


def test_bayesian_uses_regime_base_rate():
    network = BayesianNetwork()
    signals    = make_neutral_signals()
    P_ranging  = network.compute(signals, "triangular_arb", "RANGING")
    P_volatile = network.compute(signals, "triangular_arb", "VOLATILE")
    assert P_ranging > P_volatile


def test_bayesian_explain_returns_dict():
    network = BayesianNetwork()
    explanation = network.explain(make_all_signals(), "triangular_arb", "RANGING")
    assert "prior" in explanation
    assert "final_probability" in explanation
    assert "decision" in explanation


# ── CorrelationFilter tests ───────────────────────────────────────────────────

def test_correlation_filter_reduces_correlated_impact():
    cf = CorrelationFilter()
    signals = {"RSI": 0.8, "MACD": 0.8, "EMA_cross": 0.8, "rate_of_change": 0.8}
    filtered = cf.filter(signals)
    total_original = sum(signals.values())
    total_filtered  = sum(filtered.values())
    assert total_filtered < total_original


# ── ThresholdManager tests ────────────────────────────────────────────────────

def test_threshold_manager_volatile_blocks_all():
    tm = ThresholdManager()
    threshold = tm.get_threshold("VOLATILE", 100, 0.6, 0)
    assert threshold > 0.95


def test_threshold_manager_ranging_lowest():
    tm = ThresholdManager()
    t_ranging = tm.get_threshold("RANGING", 100, 0.6, 0)
    t_choppy  = tm.get_threshold("CHOPPY",  100, 0.6, 0)
    assert t_ranging < t_choppy


def test_threshold_manager_bad_win_rate_raises():
    tm = ThresholdManager()
    t_good = tm.get_threshold("RANGING", 100, 0.70, 0)
    t_bad  = tm.get_threshold("RANGING", 100, 0.45, 0)
    assert t_bad > t_good


def test_threshold_manager_consecutive_losses_raises():
    tm = ThresholdManager()
    t_0 = tm.get_threshold("RANGING", 100, 0.6, 0)
    t_5 = tm.get_threshold("RANGING", 100, 0.6, 5)
    assert t_5 > t_0


def test_threshold_always_in_bounds():
    tm = ThresholdManager()
    for regime in ["RANGING", "VOLATILE", "CHOPPY", "TRENDING_UP"]:
        for win_rate in [0.3, 0.5, 0.7, 0.9]:
            for losses in [0, 3, 7]:
                t = tm.get_threshold(regime, 100, win_rate, losses)
                if regime == "VOLATILE":
                    assert t > 0.95
                else:
                    assert 0.55 <= t <= 0.85


# ── LikelihoodEngine tests ────────────────────────────────────────────────────

def test_likelihood_engine_calculates_from_history():
    engine = LikelihoodEngine()
    mock_trades  = make_mock_trade_history(n=100)
    mock_signals = make_mock_signal_history(n=100)
    ratios = engine.calculate_from_history(mock_trades, mock_signals)
    assert "triangular_arb" in ratios
    assert all(v > 0 for v in ratios["triangular_arb"].values())


def test_likelihood_engine_falls_back_to_priors():
    engine = LikelihoodEngine()
    mock_trades  = make_mock_trade_history(n=5)
    mock_signals = make_mock_signal_history(n=5)
    ratios  = engine.calculate_from_history(mock_trades, mock_signals)
    priors  = engine.load_literature_priors()
    # Both should have the same top-level keys (signals or strategies)
    assert set(ratios.keys()) == set(priors.keys())
