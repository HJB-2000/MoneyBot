"""
Historical data downloader — fetches 2 years of Binance OHLCV data via ccxt.
No API key needed (public endpoints only).
"""
from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone

import ccxt


class DataDownloader:
    def __init__(self, exchange_id: str = "binance"):
        self._exchange = getattr(ccxt, exchange_id)({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })

    def download_all(self, symbols: list, output_dir: str = "data/historical") -> None:
        os.makedirs(output_dir, exist_ok=True)
        total = len(symbols)
        for idx, symbol in enumerate(symbols, 1):
            print(f"[{idx}/{total}] Downloading {symbol}…")
            try:
                self._download_candles(symbol, output_dir)
            except Exception as e:
                print(f"  ✗ Failed {symbol}: {e}")
                self._log_progress(symbol, "FAILED", str(e))
            else:
                print(f"  ✓ Done {symbol}")

    def _download_candles(self, symbol: str, output_dir: str) -> None:
        safe = symbol.replace("/", "_")
        out_path = os.path.join(output_dir, f"{safe}.csv")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = now_ms - int(2 * 365.25 * 24 * 3600 * 1000)

        rows_written = 0
        write_header = not os.path.exists(out_path)
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            cursor = start_ms
            while cursor < now_ms:
                retries = 0
                batch = None
                while retries < 3:
                    try:
                        batch = self._exchange.fetch_ohlcv(
                            symbol, "5m", since=cursor, limit=1000
                        )
                        break
                    except Exception as e:
                        retries += 1
                        if retries >= 3:
                            raise
                        time.sleep(2)
                if not batch:
                    break
                for row in batch:
                    writer.writerow(row)
                    rows_written += 1
                cursor = batch[-1][0] + 1
                time.sleep(0.5)
        self._log_progress(symbol, "OK", f"{rows_written} rows")

    def _log_progress(self, symbol: str, status: str, note: str):
        os.makedirs("data", exist_ok=True)
        log_path = "data/training_log.csv"
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "symbol", "status", "note"])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(), symbol, status, note
            ])
