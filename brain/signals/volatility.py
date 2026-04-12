import numpy as np
import pandas as pd


class VolatilitySignal:
    def __init__(self):
        self.atr_ratio = 1.0
        self.vol_spike = False
        self.is_squeezing = False

    def calculate(self, candles: pd.DataFrame) -> float:
        try:
            if candles is None or len(candles) < 22:
                return 0.0
            atr_score = self._atr(candles) * 0.45
            bb_score = self._bollinger(candles) * 0.35
            rv_score = self._realized_vol(candles) * 0.20
            result = atr_score + bb_score + rv_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _wilder_atr(self, candles: pd.DataFrame, period: int = 14) -> np.ndarray:
        highs = candles["high"].values.astype(float)
        lows = candles["low"].values.astype(float)
        closes = candles["close"].values.astype(float)
        tr = np.zeros(len(candles))
        for i in range(1, len(candles)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr[i] = max(hl, hc, lc)
        tr[0] = highs[0] - lows[0]

        atr = np.zeros(len(tr))
        atr[period] = tr[1:period+1].mean()
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr

    def _atr(self, candles: pd.DataFrame, period: int = 14) -> float:
        atr = self._wilder_atr(candles, period)
        current_atr = atr[-1]
        avg_atr = atr[-20:].mean()
        if avg_atr == 0:
            return 0.0
        self.atr_ratio = current_atr / avg_atr

        if self.atr_ratio < 0.7:
            return 0.6
        elif self.atr_ratio <= 1.3:
            return 0.1
        elif self.atr_ratio <= 2.0:
            return -0.3
        else:
            return -0.9  # VOLATILE override

    def _bollinger(self, candles: pd.DataFrame, period: int = 20, std: float = 2.0) -> float:
        closes = candles["close"].values.astype(float)
        rolling_mean = pd.Series(closes).rolling(period).mean().values
        rolling_std = pd.Series(closes).rolling(period).std().values

        mid = rolling_mean[-1]
        std_val = rolling_std[-1]
        if mid == 0 or std_val == 0:
            return 0.0

        upper = mid + std * std_val
        lower = mid - std * std_val
        price = closes[-1]
        bandwidth = (upper - lower) / mid

        # Squeeze: band is tighter than average OR absolutely very tight (<0.5%)
        rm = pd.Series(closes).rolling(period).mean()
        rs = pd.Series(closes).rolling(period).std()
        bw_history = ((rm + 2 * rs - (rm - 2 * rs)) / rm).dropna().values
        if len(bw_history) >= 5:
            avg_bw = float(bw_history[:-1].mean())
            squeeze_vs_avg = (
                not np.isnan(avg_bw) and avg_bw > 0
                and bandwidth < avg_bw * 0.75
            )
        else:
            squeeze_vs_avg = False
        self.is_squeezing = bool(squeeze_vs_avg or bandwidth < 0.005)

        if self.is_squeezing:
            return -0.2
        if price > upper:
            return -0.5
        elif price > mid:
            return 0.2
        elif abs(price - mid) / std_val < 0.5:
            return 0.2
        elif price < lower:
            return 0.4   # oversold
        elif price < mid:
            return -0.2
        return 0.0

    def _realized_vol(self, candles: pd.DataFrame, window: int = 12) -> float:
        self.vol_spike = False  # always reset first so stale True never persists
        closes = candles["close"].values.astype(float)
        volumes = candles["volume"].values.astype(float)
        returns = np.diff(np.log(closes))
        if len(returns) < window:
            return 0.0

        # Volume-based spike detection — threshold 10x to avoid false positives
        # (3x triggers during normal active-hours vs overnight baseline)
        if len(volumes) > window:
            recent_vol_avg = volumes[-window:].mean()
            baseline_vol_avg = volumes[:-window].mean()
            if baseline_vol_avg > 0 and recent_vol_avg / baseline_vol_avg > 10.0:
                self.vol_spike = True
                return -0.8

        daily_rv = returns.std()
        if daily_rv == 0:
            return 0.0

        recent_rv = returns[-window:].std()
        ratio = recent_rv / daily_rv
        self.vol_spike = bool(ratio > 5.0)  # 3x was too sensitive; 5x = genuine spike
        if self.vol_spike:
            return -0.8
        return 0.0
