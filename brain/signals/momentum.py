import numpy as np
import pandas as pd


class MomentumSignal:
    def calculate(self, candles: pd.DataFrame) -> float:
        try:
            if candles is None or len(candles) < 30:
                return 0.0
            rsi_score = self._rsi(candles) * 0.35
            macd_score = self._macd(candles) * 0.35
            ema_score = self._ema_cross(candles) * 0.15
            roc_score = self._rate_of_change(candles) * 0.15
            result = rsi_score + macd_score + ema_score + roc_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _ema(self, series: np.ndarray, period: int) -> np.ndarray:
        k = 2 / (period + 1)
        ema = np.zeros_like(series, dtype=float)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = series[i] * k + ema[i-1] * (1 - k)
        return ema

    def _rsi(self, candles: pd.DataFrame, period: int = 14) -> float:
        closes = candles["close"].values.astype(float)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        # Wilder smoothing
        avg_gain = gains[:period].mean()
        avg_loss = losses[:period].mean()
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        if rsi < 30:
            return 0.7
        elif rsi < 45:
            return 0.3
        elif rsi <= 55:
            return 0.0
        elif rsi <= 70:
            return -0.2
        else:
            return -0.6

    def _macd(self, candles: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        closes = candles["close"].values.astype(float)
        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)

        macd_now = macd_line[-1]
        macd_prev = macd_line[-2]
        sig_now = signal_line[-1]
        sig_prev = signal_line[-2]

        crossed_above = macd_prev < sig_prev and macd_now >= sig_now
        crossed_below = macd_prev > sig_prev and macd_now <= sig_now

        if crossed_above:
            return 0.5
        elif crossed_below:
            return -0.5
        elif macd_now > sig_now and macd_now > macd_prev:
            return 0.3
        elif macd_now < sig_now and macd_now < macd_prev:
            return -0.3
        else:
            return 0.2  # flat near zero = good for arb

    def _ema_cross(self, candles: pd.DataFrame, fast: int = 9, slow: int = 21) -> float:
        closes = candles["close"].values.astype(float)
        ema_fast = self._ema(closes, fast)
        ema_slow = self._ema(closes, slow)

        crossed_up = ema_fast[-2] < ema_slow[-2] and ema_fast[-1] >= ema_slow[-1]
        crossed_dn = ema_fast[-2] > ema_slow[-2] and ema_fast[-1] <= ema_slow[-1]

        if crossed_up:
            return 0.5
        elif crossed_dn:
            return -0.5
        elif ema_fast[-1] > ema_slow[-1]:
            return 0.3
        else:
            return -0.3

    def _rate_of_change(self, candles: pd.DataFrame, period: int = 10) -> float:
        closes = candles["close"].values.astype(float)
        if len(closes) < period + 1:
            return 0.0
        roc = (closes[-1] - closes[-period-1]) / closes[-period-1]
        if abs(roc) > 0.02:
            return -0.5  # too much movement, dangerous for arb
        elif abs(roc) < 0.005:
            return 0.4   # stable
        return 0.0
