"""
Likelihood trainer — calculates real likelihood ratios from labeled historical data.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

import pandas as pd

from bayesian.prior_loader import PriorLoader, DEFAULT_BASE_RATES

LIKELIHOOD_FILE = "data/likelihood_ratios.json"
BASE_RATES_FILE = "data/base_rates.json"

STRATEGIES = [
    "triangular_arb", "stat_arb", "funding_arb", "grid_trader",
    "mean_reversion", "volume_spike", "correlation_breakout",
]

ALL_SIGNALS = [
    "RSI", "MACD", "EMA_cross", "ATR_ratio", "bollinger_state",
    "rate_of_change", "support_resistance", "trend_structure",
    "consolidation_score", "realized_vol", "CVD_divergence", "CVD_trend",
    "volume_spike_signal", "buy_sell_ratio", "large_trade_flow",
    "bid_ask_spread", "depth_imbalance", "order_flow_imbalance",
    "large_order_presence", "iceberg_detection", "spoofing_detection",
    "liquidity_score", "book_pressure_ratio", "funding_rate",
    "open_interest_change", "liquidation_pressure", "vwap_position",
    "vwap_reclaim", "btc_correlation_momentum", "regime_persistence",
]


class LikelihoodTrainer:
    def __init__(self):
        self._loader = PriorLoader()

    def train(self, features_with_labels, output_path: str = LIKELIHOOD_FILE) -> dict:
        """
        features_with_labels: DataFrame or path to CSV with signal columns + 'label' + 'strategy'.
        """
        if isinstance(features_with_labels, str):
            df = pd.read_csv(features_with_labels)
        else:
            df = features_with_labels.copy()

        ratios = {}
        base_rates = dict(DEFAULT_BASE_RATES)

        for strategy in STRATEGIES:
            subset = df[df.get("strategy", pd.Series()) == strategy] \
                if "strategy" in df.columns else df
            if len(subset) < 30:
                ratios[strategy] = self._loader.load_literature_priors().get(strategy, {})
                continue

            wins   = subset[subset["label"] == 1]
            losses = subset[subset["label"] == 0]
            if len(wins) == 0 or len(losses) == 0:
                ratios[strategy] = {}
                continue

            strat_ratios = {}
            for sig in ALL_SIGNALS:
                if sig not in subset.columns:
                    strat_ratios[sig] = 1.0
                    continue
                pw = (wins[sig] > 0).sum() / max(len(wins), 1)
                pl = (losses[sig] > 0).sum() / max(len(losses), 1)
                lr = pw / max(pl, 0.01)
                strat_ratios[sig] = round(float(lr), 4)

            ratios[strategy] = strat_ratios
            # Base rates by regime if regime column present
            if "regime" in subset.columns:
                for regime in subset["regime"].unique():
                    reg_rows = subset[subset["regime"] == regime]
                    wr = (reg_rows["label"] == 1).sum() / max(len(reg_rows), 1)
                    base_rates[str(regime)] = round(float(wr), 4)

        # Validate
        for strategy, sr in ratios.items():
            if sr:
                top5 = sorted(sr.items(), key=lambda x: x[1], reverse=True)[:5]
                neg  = [(k, v) for k, v in sr.items() if v < 0.5]
                print(f"\n[{strategy}] Top predictive: {top5}")
                if neg:
                    print(f"  Negative predictors: {neg}")

        os.makedirs("data", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(ratios, f, indent=2)
        with open(BASE_RATES_FILE, "w") as f:
            json.dump(base_rates, f, indent=2)
        self._log("train", f"{len(df)} rows, {len(ratios)} strategies")
        return ratios

    def train_sliding_window(self, trade_log: str, signal_log: str,
                             lookback_days: int = 60) -> dict:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        # Load signal log as features
        try:
            df = pd.read_csv(signal_log)
        except Exception:
            return {}
        if "timestamp" in df.columns:
            df = df[df["timestamp"] >= cutoff]
        if len(df) < 30:
            return {}
        return self.train(df)

    def _log(self, event: str, note: str):
        path = "data/training_log.csv"
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "event", "note"])
            writer.writerow([datetime.now(timezone.utc).isoformat(), event, note])
