"""
Prior loader — loads literature-based starting priors and base rates.
"""
from __future__ import annotations

import json
import os

LITERATURE_PRIORS_FILE = "data/literature_priors.json"
BASE_RATES_FILE = "data/base_rates.json"
LIKELIHOOD_FILE = "data/likelihood_ratios.json"

LITERATURE_PRIORS = {
    "RSI":              {"mean_reversion": 1.8, "triangular_arb": 1.1,
                         "stat_arb": 1.6, "default": 1.2},
    "MACD":             {"stat_arb": 1.5, "correlation_breakout": 1.3, "default": 1.1},
    "EMA_cross":        {"correlation_breakout": 1.6, "stat_arb": 1.4, "default": 1.1},
    "ATR_ratio":        {"all": 0.4, "default": 0.4},
    "bollinger_state":  {"grid_trader": 1.7, "mean_reversion": 1.5, "default": 1.1},
    "rate_of_change":   {"correlation_breakout": 1.4, "default": 1.0},
    "support_resistance": {"mean_reversion": 1.9, "default": 1.2},
    "trend_structure":  {"stat_arb": 1.4, "default": 1.0},
    "consolidation_score": {"grid_trader": 2.1, "triangular_arb": 1.8, "default": 1.3},
    "realized_vol":     {"all": 0.5, "default": 0.5},
    "CVD_divergence":   {"mean_reversion": 2.3, "stat_arb": 1.9, "default": 1.4},
    "CVD_trend":        {"stat_arb": 1.6, "default": 1.1},
    "volume_spike_signal": {"volume_spike": 1.6, "correlation_breakout": 1.4, "default": 1.1},
    "buy_sell_ratio":   {"triangular_arb": 1.5, "default": 1.0},
    "large_trade_flow": {"stat_arb": 1.7, "default": 1.0},
    "bid_ask_spread":   {"triangular_arb": 2.8, "grid_trader": 2.1, "default": 1.5},
    "depth_imbalance":  {"stat_arb": 1.8, "triangular_arb": 1.4, "default": 1.1},
    "order_flow_imbalance": {"triangular_arb": 1.9, "default": 1.1},
    "large_order_presence": {"stat_arb": 1.5, "default": 1.0},
    "iceberg_detection": {"triangular_arb": 1.6, "default": 1.0},
    "spoofing_detection": {"triangular_arb": 1.3, "default": 1.0},
    "liquidity_score":  {"triangular_arb": 2.5, "grid_trader": 1.8, "default": 1.5},
    "book_pressure_ratio": {"stat_arb": 1.4, "default": 1.0},
    "funding_rate":     {"funding_arb": 3.2, "default": 1.2},
    "open_interest_change": {"funding_arb": 1.8, "default": 1.0},
    "liquidation_pressure": {"all": 0.1, "default": 0.1},
    "liquidation_cascade": {"all": 0.1, "default": 0.1},
    "vwap_position":    {"triangular_arb": 1.6, "grid_trader": 1.4, "default": 1.1},
    "vwap_reclaim":     {"mean_reversion": 1.9, "default": 1.2},
    "btc_correlation_momentum": {"correlation_breakout": 1.7, "default": 1.1},
    "regime_persistence": {"all": 1.2, "default": 1.2},
}

DEFAULT_BASE_RATES = {
    "RANGING":       0.55,
    "TRENDING_UP":   0.52,
    "TRENDING_DOWN": 0.50,
    "BREAKOUT":      0.48,
    "FUNDING_RICH":  0.57,
    "WHALE_MOVING":  0.45,
    "CHOPPY":        0.43,
    "VOLATILE":      0.30,
    "UNKNOWN":       0.50,
}


class PriorLoader:
    def load_likelihood_ratios(self) -> dict:
        if os.path.exists(LIKELIHOOD_FILE):
            try:
                with open(LIKELIHOOD_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return self.load_literature_priors()

    def load_literature_priors(self) -> dict:
        if os.path.exists(LITERATURE_PRIORS_FILE):
            try:
                with open(LITERATURE_PRIORS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return LITERATURE_PRIORS

    def load_base_rates(self) -> dict:
        if os.path.exists(BASE_RATES_FILE):
            try:
                with open(BASE_RATES_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return DEFAULT_BASE_RATES

    def save_literature_priors(self):
        os.makedirs("data", exist_ok=True)
        with open(LITERATURE_PRIORS_FILE, "w") as f:
            json.dump(LITERATURE_PRIORS, f, indent=2)

    def save_base_rates(self):
        os.makedirs("data", exist_ok=True)
        with open(BASE_RATES_FILE, "w") as f:
            json.dump(DEFAULT_BASE_RATES, f, indent=2)
