import time
import threading
import logging
from typing import Optional
import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_TTL = 5       # seconds — general cache TTL
OI_CACHE_TTL = 3600  # 1 hour — how long we keep previous OI for change calc


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
        # Bybit linear perpetuals — no geo-restriction, public endpoints
        self._bybit = ccxt.bybit({
            "enableRateLimit": True,
            "options": {"defaultType": "linear"},
        })
        self._cache: dict = {}
        self._lock = threading.Lock()
        self._latencies: list = []
        self._rate_limited_until: float = 0.0
        # Track previous OI values keyed by symbol for open_interest_change signal
        self._oi_prev: dict = {}   # symbol -> {"value": float, "ts": float}

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

    @staticmethod
    def _to_perp(symbol: str) -> str:
        """Convert spot symbol to Bybit linear perpetual format.
        e.g. 'SOL/USDT' -> 'SOL/USDT:USDT'
        """
        if ":" in symbol:
            return symbol
        base, quote = symbol.split("/")
        return f"{base}/{quote}:{quote}"

    def get_funding_rate(self, symbol: str) -> Optional[float]:
        key = f"funding:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        # Try Binance futures first
        try:
            data = self._call(self._futures.fetch_funding_rate, symbol)
            if data:
                rate = data.get("fundingRate")
                if rate is not None:
                    self._store(key, float(rate))
                    return float(rate)
        except Exception:
            pass

        # Fallback: Bybit linear perpetual
        try:
            perp = self._to_perp(symbol)
            data = self._call(self._bybit.fetch_funding_rate, perp)
            if data:
                rate = data.get("fundingRate")
                if rate is not None:
                    self._store(key, float(rate))
                    logger.debug(f"Funding rate for {symbol} from Bybit: {rate}")
                    return float(rate)
        except Exception:
            pass

        return None

    def get_open_interest(self, symbol: str) -> Optional[float]:
        key = f"oi:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached

        oi = None

        # Try Binance futures first
        try:
            data = self._call(self._futures.fetch_open_interest, symbol)
            if data:
                oi = data.get("openInterestAmount", data.get("openInterest"))
                if oi is not None:
                    oi = float(oi)
        except Exception:
            pass

        # Fallback: Bybit linear perpetual
        if oi is None:
            try:
                perp = self._to_perp(symbol)
                data = self._call(self._bybit.fetch_open_interest, perp)
                if data:
                    val = data.get("openInterestAmount", data.get("openInterest"))
                    if val is not None:
                        oi = float(val)
                        logger.debug(f"OI for {symbol} from Bybit: {oi}")
            except Exception:
                pass

        if oi is not None:
            self._store(key, oi)
            # Update 1h-ago tracker
            prev = self._oi_prev.get(symbol)
            if prev is None or (time.time() - prev["ts"]) >= OI_CACHE_TTL:
                self._oi_prev[symbol] = {"value": oi, "ts": time.time()}

        return oi

    def get_oi_1h_ago(self, symbol: str) -> Optional[float]:
        """Return the OI value recorded ~1h ago, or None if not available yet."""
        prev = self._oi_prev.get(symbol)
        if prev and (time.time() - prev["ts"]) >= OI_CACHE_TTL:
            return prev["value"]
        return None

    def get_liquidations(self, symbol: str) -> list:
        """Bybit exposes forced liquidation trades via recent trades with a filter.
        We approximate: fetch recent trades from Bybit and flag unusually large ones
        as proxy liquidations. Returns list of {"side": "long"/"short", "amount": float}.
        """
        key = f"liq:{symbol}"
        cached = self._cached(key)
        if cached is not None:
            return cached
        try:
            perp = self._to_perp(symbol)
            trades = self._call(self._bybit.fetch_trades, perp, limit=100)
            if not trades:
                return []
            amounts = [t.get("amount", 0) for t in trades]
            if not amounts:
                return []
            avg = sum(amounts) / len(amounts)
            threshold = avg * 5  # trades 5x avg size = likely liquidation
            liq = []
            for t in trades:
                if t.get("amount", 0) >= threshold:
                    side = "long" if t.get("side") == "sell" else "short"
                    liq.append({"side": side, "amount": t.get("amount", 0)})
            self._store(key, liq)
            return liq
        except Exception:
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
