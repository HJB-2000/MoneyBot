import numpy as np
import pandas as pd
from datetime import timezone


class VWAPSignal:
    def __init__(self):
        self.vwap_price = 0.0
        self.price_to_vwap_pct = 0.0

    def calculate(self, candles: pd.DataFrame, trades: list = None) -> float:
        try:
            if candles is None or len(candles) < 5:
                return 0.0
            vwap, upper, lower = self._calculate_vwap(candles)
            if vwap == 0:
                return 0.0
            self.vwap_price = vwap
            price = candles["close"].values[-1]
            self.price_to_vwap_pct = (price - vwap) / vwap

            reclaim_score = self._vwap_reclaim(candles, vwap)
            if reclaim_score != 0.0:
                return float(np.clip(reclaim_score, -1.0, 1.0))

            pos_score = self._vwap_position_score(price, vwap, upper, lower) * 0.40
            dist_score = self._distance_from_vwap(price, vwap) * 0.20

            result = pos_score + dist_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _calculate_vwap(self, candles: pd.DataFrame):
        df = candles.copy()
        # Reset at midnight UTC
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            today_start = pd.Timestamp.now('UTC').normalize()
            df = df[df["timestamp"] >= today_start]
            if len(df) == 0:
                df = candles  # fallback: use all

        typical = (df["high"] + df["low"] + df["close"]) / 3
        cumvol = df["volume"].cumsum()
        cumtpv = (typical * df["volume"]).cumsum()

        if cumvol.iloc[-1] == 0:
            return 0.0, 0.0, 0.0

        vwap = cumtpv.iloc[-1] / cumvol.iloc[-1]
        deviation = ((typical - vwap) ** 2 * df["volume"]).sum() / df["volume"].sum()
        std_dev = np.sqrt(deviation) if deviation > 0 else 0.0
        upper = vwap + 1.5 * std_dev
        lower = vwap - 1.5 * std_dev
        return float(vwap), float(upper), float(lower)

    def _vwap_position_score(self, price: float, vwap: float, upper: float, lower: float) -> float:
        if price > upper:
            return -0.4
        elif price > vwap:
            return 0.2
        elif abs(price - vwap) / vwap < 0.001:
            return 0.0   # at VWAP
        elif price < lower:
            return 0.4   # oversold vs VWAP
        else:
            return -0.2

    def _vwap_reclaim(self, candles: pd.DataFrame, vwap: float) -> float:
        closes = candles["close"].values.astype(float)
        volumes = candles["volume"].values.astype(float)
        avg_vol = volumes[-20:].mean() if len(volumes) >= 20 else volumes.mean()

        prev_price = closes[-2]
        curr_price = closes[-1]
        vol_now = volumes[-1]
        with_volume = vol_now > avg_vol * 1.2

        if prev_price < vwap and curr_price >= vwap and with_volume:
            return 0.6   # reclaim with volume: strong bullish
        elif prev_price > vwap and curr_price <= vwap and with_volume:
            return -0.6  # lost VWAP with volume: strong bearish
        return 0.0

    def _distance_from_vwap(self, price: float, vwap: float) -> float:
        if vwap == 0:
            return 0.0
        dist = abs(price - vwap) / vwap
        if dist < 0.002:
            return 0.3   # very close = best for arb
        elif dist > 0.005:
            return -0.2  # far from VWAP = trending conditions
        return 0.0
