"""
Correlation trainer — builds signal correlation matrix from historical features.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


CORRELATIONS_FILE = "data/signal_correlations.json"

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


class CorrelationTrainer:
    def train(self, features_df, output_path: str = CORRELATIONS_FILE) -> dict:
        if isinstance(features_df, str):
            df = pd.read_csv(features_df)
        else:
            df = features_df.copy()

        cols = [c for c in ALL_SIGNALS if c in df.columns]
        if len(cols) < 2:
            print("[CorrelationTrainer] Not enough signal columns")
            return {}

        matrix = {}
        corr_df = df[cols].corr(method="pearson")

        correlated_pairs = []
        for i, a in enumerate(cols):
            matrix[a] = {}
            for j, b in enumerate(cols):
                if i == j:
                    continue
                val = corr_df.loc[a, b]
                if not np.isnan(val):
                    matrix[a][b] = round(float(val), 4)
                    if j > i and abs(val) > 0.70:
                        correlated_pairs.append((a, b, round(float(val), 4)))

        correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        print(f"\n[CorrelationTrainer] Top 10 correlated pairs:")
        for a, b, c in correlated_pairs[:10]:
            print(f"  {a} ↔ {b}: {c:.3f}")

        os.makedirs("data", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(matrix, f, indent=2)
        print(f"[CorrelationTrainer] Saved → {output_path}")
        return matrix
