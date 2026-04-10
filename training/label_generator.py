"""
Label generator — adds WIN/LOSS labels to historical feature rows
by looking ahead N candles per strategy.
"""
from __future__ import annotations

import os

import pandas as pd


LOOKAHEAD = {
    "triangular_arb":       5,
    "stat_arb":             48,
    "mean_reversion":       12,
    "volume_spike":         6,
    "correlation_breakout": 4,
    "grid_trader":          96,
    "funding_arb":          12,
}

MIN_PROFIT = 0.001  # 0.1% after fees


class LabelGenerator:
    def generate(self, features_df: pd.DataFrame,
                 strategy: str = "triangular_arb",
                 candles_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        features_df: DataFrame with 'close' or matchable timestamps.
        candles_df:  Full candle DataFrame with 'close' column (optional;
                     uses features_df['close'] if not provided).
        Returns features_df with added 'label' column (1=win, 0=loss).
        """
        df = features_df.copy()
        lookahead = LOOKAHEAD.get(strategy, 12)

        close_col = "close"
        if close_col not in df.columns:
            df["label"] = 0
            return df

        closes = df[close_col].values.astype(float)
        labels = []
        for i in range(len(closes)):
            future_idx = min(i + lookahead, len(closes) - 1)
            entry  = closes[i]
            future = closes[future_idx]
            if entry == 0:
                labels.append(0)
                continue
            pct_move = (future - entry) / entry
            labels.append(1 if pct_move > MIN_PROFIT else 0)

        df["label"] = labels
        win_rate = sum(labels) / len(labels) if labels else 0.0
        print(f"[LabelGenerator] {strategy}: {len(df)} rows, win rate = {win_rate:.2%}")
        return df
