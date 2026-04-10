"""
Candle signal group — 10 signals from OHLCV candle data.
All scores in [-1.0, +1.0]. Pandas + numpy only, no TA-Lib.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class CandleSignals:
    def __init__(self):
        self.atr_ratio: float = 1.0
        self.is_squeezing: bool = False
        self.breakout_detected: bool = False
        self.vol_spike: bool = False
        self.consolidating: bool = False
        self.trend: str = "ranging"
        self.realized_vol_ratio: float = 1.0

    def calculate(self, candles: pd.DataFrame) -> dict:
        zeros = {k: 0.0 for k in [
            "RSI", "MACD", "EMA_cross", "ATR_ratio", "bollinger_state",
            "rate_of_change", "support_resistance", "trend_structure",
            "consolidation_score", "realized_vol",
        ]}
        try:
            if candles is None or len(candles) < 30:
                return zeros
            closes = candles["close"].values.astype(float)
            highs  = candles["high"].values.astype(float)
            lows   = candles["low"].values.astype(float)
            vols   = candles["volume"].values.astype(float)
            return {
                "RSI":                  self._rsi(closes),
                "MACD":                 self._macd(closes),
                "EMA_cross":            self._ema_cross(closes),
                "ATR_ratio":            self._atr_ratio(highs, lows, closes),
                "bollinger_state":      self._bollinger(closes),
                "rate_of_change":       self._roc(closes),
                "support_resistance":   self._support_resistance(closes, highs, lows, vols),
                "trend_structure":      self._trend_structure(closes),
                "consolidation_score":  self._consolidation(closes),
                "realized_vol":         self._realized_vol(closes, vols),
            }
        except Exception:
            return zeros

    # ── Signal 9 — RSI ────────────────────────────────────────────────────
    def _rsi(self, closes: np.ndarray, period: int = 14) -> float:
        if len(closes) < period + 1:
            return 0.0
        deltas = np.diff(closes[-(period + 14):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
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
        if rsi < 25:   return 0.9
        elif rsi < 35: return 0.5
        elif rsi < 45: return 0.2
        elif rsi < 55: return 0.0
        elif rsi < 65: return -0.2
        elif rsi < 75: return -0.5
        return -0.9

    # ── Signal 10 — MACD ─────────────────────────────────────────────────
    def _macd(self, closes: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> float:
        if len(closes) < slow + sig:
            return 0.0
        def ema(arr, p):
            k = 2 / (p + 1)
            e = arr[:p].mean()
            for v in arr[p:]:
                e = v * k + e * (1 - k)
            return e
        def ema_series(arr, p):
            k = 2 / (p + 1)
            result = np.zeros(len(arr))
            result[p - 1] = arr[:p].mean()
            for i in range(p, len(arr)):
                result[i] = arr[i] * k + result[i - 1] * (1 - k)
            return result

        ema_fast = ema_series(closes, fast)
        ema_slow = ema_series(closes, slow)
        macd_line = ema_fast - ema_slow
        valid = macd_line[slow - 1:]
        if len(valid) < sig:
            return 0.0
        signal_line = ema_series(valid, sig)
        hist = valid - signal_line

        if len(hist) < 3:
            return 0.0
        h_curr = hist[-1]
        h_prev = hist[-2]
        h_prev2 = hist[-3]

        # Crossover detection
        if h_prev <= 0 < h_curr:
            return 0.7
        if h_prev >= 0 > h_curr:
            return -0.7
        if abs(h_curr) < 1e-9 * abs(closes[-1]):
            return 0.2  # flat near zero = arb friendly
        if h_curr > 0:
            return 0.5 if h_curr > h_prev else 0.1
        else:
            return -0.5 if h_curr < h_prev else -0.1

    # ── Signal 11 — EMA Cross ─────────────────────────────────────────────
    def _ema_cross(self, closes: np.ndarray, fast: int = 9, slow: int = 21) -> float:
        if len(closes) < slow + 2:
            return 0.0
        def ema_last2(arr, p):
            k = 2 / (p + 1)
            e = arr[:p].mean()
            prev = e
            for i, v in enumerate(arr[p:]):
                prev = e
                e = v * k + e * (1 - k)
            return prev, e
        f_prev, f_curr = ema_last2(closes, fast)
        s_prev, s_curr = ema_last2(closes, slow)
        crossed_up   = f_prev <= s_prev and f_curr > s_curr
        crossed_down = f_prev >= s_prev and f_curr < s_curr
        if crossed_up:   return 0.6
        if crossed_down: return -0.6
        gap = abs(f_curr - s_curr)
        gap_prev = abs(f_prev - s_prev)
        widening = gap > gap_prev
        if f_curr > s_curr:
            return 0.4 if widening else 0.1
        else:
            return -0.4 if widening else -0.1

    # ── Signal 12 — ATR Ratio ─────────────────────────────────────────────
    def _atr_ratio(self, highs, lows, closes, period: int = 14) -> float:
        n = len(closes)
        if n < period + 20:
            return 0.0
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
        tr[0] = highs[0] - lows[0]
        atr = np.zeros(n)
        atr[period] = tr[1:period + 1].mean()
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        current = atr[-1]
        avg = atr[-20:].mean()
        if avg == 0:
            return 0.0
        ratio = current / avg
        self.atr_ratio = float(ratio)
        if ratio < 0.5:   return 0.8
        elif ratio < 0.8: return 0.4
        elif ratio < 1.2: return 0.1
        elif ratio < 1.8: return -0.3
        elif ratio < 2.0: return -0.7
        return -1.0

    # ── Signal 13 — Bollinger ─────────────────────────────────────────────
    def _bollinger(self, closes: np.ndarray, period: int = 20, std: float = 2.0) -> float:
        if len(closes) < period:
            return 0.0
        s = pd.Series(closes)
        rm = s.rolling(period).mean().values
        rs = s.rolling(period).std().values
        mid = rm[-1]
        std_val = rs[-1]
        if mid == 0 or std_val == 0 or np.isnan(mid) or np.isnan(std_val):
            return 0.0
        upper = mid + std * std_val
        lower = mid - std * std_val
        bandwidth = (upper - lower) / mid
        self.is_squeezing = bool(bandwidth < 0.001)
        price = closes[-1]
        if self.is_squeezing:
            return -0.2
        avg_bw = 0.02  # approximate normal bandwidth
        if bandwidth > avg_bw * 1.5:
            return -0.4
        if price > upper:
            return -0.4
        elif price > mid:
            return 0.2
        elif abs(price - mid) / std_val < 0.5:
            return 0.2
        elif price < lower:
            return 0.5
        elif price < mid:
            return -0.2
        return 0.0

    # ── Signal 14 — Rate of Change ───────────────────────────────────────
    def _roc(self, closes: np.ndarray, period: int = 10) -> float:
        if len(closes) < period + 1:
            return 0.0
        roc = (closes[-1] - closes[-period - 1]) / (closes[-period - 1] + 1e-9)
        if abs(roc) > 0.03:
            return -0.7
        elif abs(roc) > 0.01:
            return -0.2
        elif abs(roc) < 0.005:
            return 0.5
        return 0.0

    # ── Signal 15 — Support / Resistance ─────────────────────────────────
    def _support_resistance(self, closes, highs, lows, vols, lookback: int = 200) -> float:
        c = closes[-min(lookback, len(closes)):]
        h = highs[-min(lookback, len(highs)):]
        l = lows[-min(lookback, len(lows)):]
        v = vols[-min(lookback, len(vols)):]
        price = closes[-1]

        # Pivot highs and lows
        levels = []
        for i in range(2, len(c) - 2):
            if h[i] >= max(h[i - 2], h[i - 1], h[i + 1], h[i + 2]):
                levels.append(h[i])
            if l[i] <= min(l[i - 2], l[i - 1], l[i + 1], l[i + 2]):
                levels.append(l[i])

        # Find strong levels (touched 3+ times)
        tolerance = price * 0.003
        strong = [lvl for lvl in levels
                  if sum(1 for x in c if abs(x - lvl) < tolerance) >= 3]

        self.breakout_detected = False
        if not strong:
            return 0.0

        nearest_above = [s for s in strong if s > price]
        nearest_below = [s for s in strong if s <= price]
        resistance = min(nearest_above) if nearest_above else None
        support = max(nearest_below) if nearest_below else None

        # Breakout: price just crossed above resistance with volume
        if resistance and price > resistance and v[-1] > v[-20:].mean() * 1.5:
            self.breakout_detected = True
            return -0.2

        if resistance and (resistance - price) / price < 0.005:
            return -0.3   # approaching strong resistance
        if support and (price - support) / price < 0.005:
            return 0.5    # bouncing off strong support
        # Price near center of range
        if support and resistance:
            mid = (support + resistance) / 2
            if abs(price - mid) / mid < 0.01:
                return 0.4
        return 0.0

    # ── Signal 16 — Trend Structure ──────────────────────────────────────
    def _trend_structure(self, closes: np.ndarray) -> float:
        if len(closes) < 20:
            return 0.0
        # Find local swing highs and lows over last 60 candles
        c = closes[-60:] if len(closes) >= 60 else closes
        pivots_high, pivots_low = [], []
        for i in range(2, len(c) - 2):
            if c[i] > c[i - 1] and c[i] > c[i - 2] and c[i] > c[i + 1] and c[i] > c[i + 2]:
                pivots_high.append(c[i])
            if c[i] < c[i - 1] and c[i] < c[i - 2] and c[i] < c[i + 1] and c[i] < c[i + 2]:
                pivots_low.append(c[i])

        if len(pivots_high) >= 2 and len(pivots_low) >= 2:
            hh = pivots_high[-1] > pivots_high[-2]
            hl = pivots_low[-1] > pivots_low[-2]
            lh = pivots_high[-1] < pivots_high[-2]
            ll = pivots_low[-1] < pivots_low[-2]
            if hh and hl:
                self.trend = "uptrend"
                return 0.5
            elif lh and ll:
                self.trend = "downtrend"
                return -0.5
            else:
                self.trend = "ranging"
                return 0.3
        self.trend = "ranging"
        return 0.3

    # ── Signal 17 — Consolidation ────────────────────────────────────────
    def _consolidation(self, closes: np.ndarray, period: int = 20) -> float:
        if len(closes) < period:
            return 0.0
        c = closes[-period:]
        mid = c.mean()
        if mid == 0:
            return 0.0
        hl_range = (c.max() - c.min()) / mid
        self.consolidating = bool(hl_range < 0.003)
        if hl_range < 0.003:
            return 0.8
        elif hl_range < 0.008:
            return 0.3
        elif hl_range < 0.02:
            return 0.0
        return -0.4

    # ── Signal 18 — Realized Volatility ──────────────────────────────────
    def _realized_vol(self, closes: np.ndarray, vols: np.ndarray, window: int = 12) -> float:
        if len(closes) < window + 1:
            return 0.0
        returns = np.diff(np.log(closes + 1e-9))
        if len(returns) < window:
            return 0.0
        recent_rv  = returns[-window:].std()
        overall_rv = returns.std()
        if overall_rv == 0:
            return 0.0
        ratio = recent_rv / overall_rv
        self.realized_vol_ratio = float(ratio)

        # Volume-based spike detection
        if len(vols) > window:
            recent_vol_avg   = vols[-window:].mean()
            baseline_vol_avg = vols[:-window].mean()
            if baseline_vol_avg > 0 and recent_vol_avg / baseline_vol_avg > 3.0:
                self.vol_spike = True
                return -1.0
        self.vol_spike = bool(ratio > 3.0)

        if ratio < 0.5:   return 0.6
        elif ratio < 1.0: return 0.2
        elif ratio < 2.0: return -0.2
        elif ratio < 3.0: return -0.6
        return -1.0
