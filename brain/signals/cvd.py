import numpy as np
import pandas as pd


class CVDSignal:
    def __init__(self):
        self.cvd_divergence = False

    def calculate(self, trades: list, candles: pd.DataFrame) -> float:
        try:
            if not trades or candles is None or len(candles) < 5:
                return 0.0
            cvd = self._build_cvd(trades, candles)
            if cvd is None or len(cvd) < 3:
                return 0.0
            closes = candles["close"].values[-len(cvd):]

            trend_score = self._cvd_trend(cvd, closes) * 0.75
            exhaust_score = self._cvd_exhaustion(cvd, closes) * 0.10
            absorb_score = self._absorption(cvd, closes) * 0.15

            result = trend_score + exhaust_score + absorb_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _build_cvd(self, trades: list, candles: pd.DataFrame):
        """Build CVD series aligned to candles using trade side classification."""
        if not trades:
            return None
        buy_vol = sum(t.get("amount", 0) for t in trades if t.get("side") == "buy")
        sell_vol = sum(t.get("amount", 0) for t in trades if t.get("side") == "sell")

        # Build a simple per-candle delta using candle OHLC heuristic when trade
        # timestamps aren't reliable: if close > open → net buy candle, else net sell.
        closes = candles["close"].values.astype(float)
        opens = candles["open"].values.astype(float)
        volumes = candles["volume"].values.astype(float)

        delta = np.where(closes > opens, volumes, np.where(closes < opens, -volumes, 0.0))
        cvd = np.cumsum(delta)
        return cvd

    def _cvd_trend(self, cvd: np.ndarray, prices: np.ndarray) -> float:
        if len(cvd) < 3:
            return 0.0
        cvd_rising = cvd[-1] > cvd[-3]
        price_rising = prices[-1] > prices[-3]

        cvd_falling = cvd[-1] < cvd[-3]
        price_falling = prices[-1] < prices[-3]

        if cvd_rising and price_rising:
            self.cvd_divergence = False
            return 0.5   # confirmed uptrend
        elif cvd_falling and price_falling:
            self.cvd_divergence = False
            return -0.5  # confirmed downtrend
        elif cvd_rising and not price_rising:
            self.cvd_divergence = True
            return 0.7   # bullish divergence — strongest signal
        elif cvd_falling and price_rising:
            self.cvd_divergence = True
            return -0.7  # bearish divergence
        self.cvd_divergence = False
        return 0.0

    def _cvd_exhaustion(self, cvd: np.ndarray, prices: np.ndarray) -> float:
        if len(cvd) < 5:
            return 0.0
        cvd_surge = abs(cvd[-1] - cvd[-5]) > abs(cvd[-6:-1].mean()) * 2 if len(cvd) >= 6 else False
        price_stalled = abs(prices[-1] - prices[-3]) / (prices[-3] + 1e-9) < 0.001

        if cvd_surge and price_stalled:
            session_high = prices.max()
            session_low = prices.min()
            price = prices[-1]
            if abs(price - session_high) / (session_high + 1e-9) < 0.005:
                return -0.5   # exhaustion near high → reversal
            elif abs(price - session_low) / (session_low + 1e-9) < 0.005:
                return 0.5    # exhaustion near low → reversal up
        return 0.0

    def _absorption(self, cvd: np.ndarray, prices: np.ndarray) -> float:
        if len(cvd) < 4:
            return 0.0
        price_dropping = prices[-1] < prices[-4]
        cvd_rising = cvd[-1] > cvd[-4]
        if price_dropping and cvd_rising:
            return 0.6   # buyers absorbing selling pressure
        return 0.0
