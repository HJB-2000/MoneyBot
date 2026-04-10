"""
Feature generator — runs all 30 signals on historical candle data.
Streams output row-by-row to CSV to avoid OOM on large datasets.
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

# Process every Nth candle — reduces rows by 4x while keeping signal diversity
STRIDE = 4


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

        os.makedirs(os.path.dirname(output_path) or "data", exist_ok=True)
        t0 = time.time()
        total_rows = 0
        header_written = False

        for fpath in candle_files:
            symbol = fpath.stem.replace("_", "/")
            print(f"  Processing {symbol}…", flush=True)
            try:
                rows_written = self._process_symbol_stream(
                    symbol, fpath, output_path, header_written
                )
                header_written = True
                total_rows += rows_written
                print(f"  ✓ {symbol}: {rows_written} rows", flush=True)
            except Exception as e:
                print(f"  ✗ {symbol}: {e}", flush=True)

        elapsed = time.time() - t0
        print(f"[FeatureGenerator] {total_rows} rows → {output_path} ({elapsed:.1f}s)", flush=True)
        # Return empty df — caller uses the CSV path directly
        return pd.DataFrame()

    def _process_symbol_stream(self, symbol: str, fpath: Path,
                                output_path: str, header_written: bool) -> int:
        df = pd.read_csv(fpath)
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        lookback = 100
        c_sig = CandleSignals()
        t_sig = TradeSignals()
        f_sig = FuturesSignals()
        d_sig = DerivedSignals()

        rows_written = 0
        writer = None
        f_out = None

        try:
            indices = range(lookback, len(df), STRIDE)
            for i in indices:
                window = df.iloc[i - lookback: i + 1].reset_index(drop=True)
                try:
                    c_scores  = c_sig.calculate(window)
                    ob_scores = self._approx_orderbook_signals(window)
                    t_scores  = t_sig.calculate([], window)
                    f_scores  = f_sig.calculate(0.0, 1000, [], window)
                    d_scores  = d_sig.calculate(c_scores, t_scores)
                    all_scores = {**ob_scores, **c_scores, **t_scores, **f_scores, **d_scores}
                    row = {"timestamp": str(df.iloc[i]["timestamp"]),
                           "symbol": symbol, "close": float(df.iloc[i]["close"])}
                    row.update(all_scores)

                    if writer is None:
                        mode = "a" if header_written else "w"
                        f_out = open(output_path, mode, newline="")
                        writer = csv.DictWriter(f_out, fieldnames=list(row.keys()))
                        if not header_written:
                            writer.writeheader()

                    writer.writerow(row)
                    rows_written += 1

                    # Flush every 500 rows to keep memory low
                    if rows_written % 500 == 0:
                        f_out.flush()

                except Exception:
                    continue
        finally:
            if f_out:
                f_out.flush()
                f_out.close()

        return rows_written

    def _approx_orderbook_signals(self, candles: pd.DataFrame) -> dict:
        """Approximate order book signals from candle data when no live OB available."""
        last = candles.iloc[-1]
        ob = {
            "bids": [(last["close"] - i * 0.01, 10.0) for i in range(10)],
            "asks": [(last["close"] + i * 0.01, 10.0) for i in range(10)],
        }
        return OrderBookSignals().calculate(ob, trade_size_usd=50.0)
