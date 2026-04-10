"""
Feature generator — runs all 30 signals on historical candle data.
Outputs data/ml_features.parquet (or .csv if parquet unavailable).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from signals.group_candles import CandleSignals
from signals.group_orderbook import OrderBookSignals
from signals.group_trades import TradeSignals
from signals.group_futures import FuturesSignals
from signals.group_derived import DerivedSignals


class FeatureGenerator:
    def generate(self, historical_dir: str = "data/historical",
                 output_path: str = "data/ml_features.csv") -> pd.DataFrame:
        hist_path = Path(historical_dir)
        if not hist_path.exists():
            print(f"[FeatureGenerator] No historical data at {historical_dir}")
            return pd.DataFrame()

        candle_files = sorted(hist_path.glob("*.csv"))
        if not candle_files:
            print("[FeatureGenerator] No CSV files found")
            return pd.DataFrame()

        all_rows = []
        t0 = time.time()
        for fpath in candle_files:
            symbol = fpath.stem.replace("_", "/")
            print(f"  Processing {symbol}…")
            try:
                rows = self._process_symbol(symbol, fpath)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(output_path) or "data", exist_ok=True)
        df.to_csv(output_path, index=False)
        elapsed = time.time() - t0
        print(f"[FeatureGenerator] {len(df)} rows → {output_path} ({elapsed:.1f}s)")
        return df

    def _process_symbol(self, symbol: str, fpath: Path) -> list:
        df = pd.read_csv(fpath)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        rows = []
        lookback = 100
        c_sig = CandleSignals()
        t_sig = TradeSignals()
        f_sig = FuturesSignals()
        d_sig = DerivedSignals()

        for i in range(lookback, len(df)):
            window = df.iloc[i - lookback: i + 1].reset_index(drop=True)
            try:
                c_scores = c_sig.calculate(window)
                ob_scores = self._approx_orderbook_signals(window)
                t_scores = t_sig.calculate([], window)
                f_scores = f_sig.calculate(0.0, 1000, [], window)
                d_scores = d_sig.calculate(c_scores, t_scores)
                all_scores = {**ob_scores, **c_scores, **t_scores, **f_scores, **d_scores}
                row = {"timestamp": df.iloc[i]["timestamp"], "symbol": symbol}
                row.update(all_scores)
                rows.append(row)
            except Exception:
                continue
        return rows

    def _approx_orderbook_signals(self, candles: pd.DataFrame) -> dict:
        """Approximate order book signals from candle data when no live OB available."""
        last = candles.iloc[-1]
        spread_approx = (last["high"] - last["low"]) / last["close"] if last["close"] else 0
        ob = {
            "bids": [(last["close"] - i * 0.01, 10.0) for i in range(10)],
            "asks": [(last["close"] + i * 0.01, 10.0) for i in range(10)],
        }
        return OrderBookSignals().calculate(ob, trade_size_usd=50.0)
