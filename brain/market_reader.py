import time
import threading
import logging
from typing import Optional
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_TTL = 5  # seconds


class MarketReader:
    def __init__(self, config: dict):
        cfg = config["exchange"]
        self._exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._futures = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._latencies: list = []
        self._rate_limited_until: float = 0.0

    # ------------------------------------------------------------------ #
    #  Cache helpers                                                       #
    # ------------------------------------------------------------------ #

    def _cached(self, key: str):
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < CACHE_TTL:
            return entry["data"]
        return None

    def _store(self, key: str, data):
        with self._lock:
            self._cache[key] = {"data": data, "ts": time.time()}

    def _call(self, fn, *args, **kwargs):
        """Wrap an exchange call: measure latency, handle errors gracefully."""
        if time.time() < self._rate_limited_until:
            time.sleep(self._rate_limited_until - time.time())
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            latency_ms = (time.time() - t0) * 1000
            self._latencies.append(latency_ms)
            if len(self._latencies) > 100:
                self._latencies.pop(0)
            return result
        except ccxt.RateLimitExceeded:
            self._rate_limited_until = time.time() + 10
            logger.warning("Rate limit hit — pausing 10s")
            return None
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.warning(f"Exchange error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected error in market call: {e}")
            return None

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    # ------------------------------------------------------------------ #
    #  Public data methods                                                 #
    # ------------------------------------------------------------------ #

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[dict]:
        key = f"ob:{symbol}:{limit}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = self._call(self._exchange.fetch_order_book, symbol, limit)
        if data:
            self._store(key, data)
        return data

    def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 100) -> Optional[pd.DataFrame]:
        key = f"candles:{symbol}:{timeframe}:{limit}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        raw = self._call(self._exchange.fetch_ohlcv, symbol, timeframe, limit=limit)
        if raw is None:
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        self._store(key, df)
        return df

    def get_trades(self, symbol: str, limit: int = 200) -> Optional[list]:
        key = f"trades:{symbol}:{limit}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = self._call(self._exchange.fetch_trades, symbol, limit=limit)
        if data:
            self._store(key, data)
        return data

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        key = f"funding:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        try:
            data = self._call(self._futures.fetch_funding_rate, symbol)
            if data:
                rate = data.get("fundingRate", None)
                self._store(key, rate)
                return rate
        except Exception:
            pass
        return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        key = f"oi:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        try:
            data = self._call(self._futures.fetch_open_interest, symbol)
            if data:
                oi = data.get("openInterestAmount", data.get("openInterest", None))
                self._store(key, oi)
                return oi
        except Exception:
            pass
        return None

    def get_liquidations(self, symbol: str) -> list:
        # Binance does not expose a public liquidation endpoint in ccxt;
        # return empty list — sentinel signal will handle gracefully.
        return []

    def get_24h_volume(self, symbol: str) -> Optional[float]:
        key = f"vol24:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        ticker = self.get_ticker(symbol)
        if ticker:
            vol = ticker.get("quoteVolume", None)
            self._store(key, vol)
            return vol
        return None

    def get_ticker(self, symbol: str) -> Optional[dict]:
        key = f"ticker:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        data = self._call(self._exchange.fetch_ticker, symbol)
        if data:
            self._store(key, data)
        return data

    def get_exchange_inflows(self, symbol: str) -> float:
        # On-chain exchange flow data is not available via ccxt.
        # Returning 0.0 — whale_detector will treat this as neutral.
        return 0.0
