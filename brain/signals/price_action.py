import numpy as np
import pandas as pd


class PriceActionSignal:
    def __init__(self):
        self.breakout_detected = False
        self.breakout_direction = None
        self.trend = "ranging"
        self.consolidating = False

    def calculate(self, candles: pd.DataFrame) -> float:
        try:
            if candles is None or len(candles) < 20:
                return 0.0
            scores = []
            scores.append(self._support_resistance_score(candles) * 0.30)
            scores.append(self._trend_structure(candles) * 0.35)
            scores.append(self._consolidation(candles) * 0.20)
            breakout_score = self._breakout(candles) * 0.15
            scores.append(breakout_score)
            result = sum(scores)
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _support_resistance_score(self, candles: pd.DataFrame, lookback: int = 200) -> float:
        df = candles.tail(lookback)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        price = closes[-1]
        tolerance = price * 0.002  # 0.2%

        levels = []
        for level in np.concatenate([highs, lows]):
            touches = np.sum(np.abs(np.concatenate([highs, lows]) - level) < tolerance)
            if touches >= 3:
                levels.append(level)

        if not levels:
            return 0.0

        levels = np.array(levels)
        supports = levels[levels < price]
        resistances = levels[levels > price]

        if len(supports) == 0 or len(resistances) == 0:
            return 0.0

        nearest_sup = supports.max()
        nearest_res = resistances.min()
        mid = (nearest_sup + nearest_res) / 2
        range_size = nearest_res - nearest_sup

        if range_size == 0:
            return 0.0

        dist_from_mid = abs(price - mid) / (range_size / 2)
        if dist_from_mid < 0.3:
            return 0.5   # near middle: good for arb
        elif price > nearest_res * 0.998:
            return -0.3  # near resistance
        elif price < nearest_sup * 1.002:
            return 0.3   # near support
        return 0.0

    def _trend_structure(self, candles: pd.DataFrame) -> float:
        highs = candles["high"].values[-10:]
        lows = candles["low"].values[-10:]

        # Simplified HH/HL / LH/LL over last 5 swing points
        hh = all(highs[i] > highs[i-1] for i in range(1, len(highs)))
        hl = all(lows[i] > lows[i-1] for i in range(1, len(lows)))
        lh = all(highs[i] < highs[i-1] for i in range(1, len(highs)))
        ll = all(lows[i] < lows[i-1] for i in range(1, len(lows)))

        if hh and hl:
            self.trend = "uptrend"
            return 0.5
        elif lh and ll:
            self.trend = "downtrend"
            return -0.5
        else:
            self.trend = "ranging"
            return 0.0

    def _consolidation(self, candles: pd.DataFrame, period: int = 20) -> float:
        df = candles.tail(period)
        high = df["high"].max()
        low = df["low"].min()
        mid = (high + low) / 2
        if mid == 0:
            return 0.0
        range_ratio = (high - low) / mid
        self.consolidating = range_ratio < 0.003

        if range_ratio < 0.003:
            return 0.6
        elif range_ratio < 0.01:
            return 0.1
        elif range_ratio > 0.03:
            return -0.3
        return 0.0

    def _breakout(self, candles: pd.DataFrame) -> float:
        if len(candles) < 21:
            return 0.0
        closes = candles["close"].values
        volumes = candles["volume"].values
        price = closes[-1]
        prev_high = candles["high"].values[-21:-1].max()
        prev_low = candles["low"].values[-21:-1].min()
        avg_vol = volumes[-20:-1].mean()
        current_vol = volumes[-1]

        if current_vol > 2 * avg_vol:
            if price > prev_high:
                self.breakout_detected = True
                self.breakout_direction = "up"
                return 0.7
            elif price < prev_low:
                self.breakout_detected = True
                self.breakout_direction = "down"
                return -0.7
        self.breakout_detected = False
        self.breakout_direction = None
        return 0.0
