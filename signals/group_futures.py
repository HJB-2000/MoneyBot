"""
Futures signal group — 5 signals from futures data + VWAP.
All scores in [-1.0, +1.0]. Gracefully returns zeros if data unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FuturesSignals:
    def __init__(self):
        self.funding_rate: float = 0.0
        self.liquidation_cascade: bool = False
        self.vwap_price: float = 0.0
        self.distance_from_vwap_pct: float = 0.0

    def calculate(self, funding, oi, liquidations, candles: pd.DataFrame,
                  oi_1h_ago=None) -> dict:
        zeros = {k: 0.0 for k in [
            "funding_rate", "open_interest_change",
            "liquidation_pressure", "vwap_position", "vwap_reclaim",
        ]}
        try:
            if candles is None or len(candles) < 5:
                return zeros
            # Guard: if all primary inputs are None treat as unavailable
            if funding is None and oi is None and liquidations is None:
                return zeros
            self.funding_rate = float(funding) if funding is not None else 0.0
            return {
                "funding_rate":          self._funding_rate(funding),
                "open_interest_change":  self._oi_change(oi, oi_1h_ago, candles),
                "liquidation_pressure":  self._liquidation_pressure(liquidations),
                "vwap_position":         self._vwap_position(candles),
                "vwap_reclaim":          self._vwap_reclaim(candles),
            }
        except Exception:
            return zeros

    # ── Signal 24 — Funding Rate ─────────────────────────────────────────
    def _funding_rate(self, rate) -> float:
        if rate is None:
            return 0.0
        r = float(rate)
        self.funding_rate = r
        if r > 0.001:    return 0.3
        elif r > 0.0005: return 0.1
        elif r >= -0.0002: return 0.0
        elif r >= -0.0005: return -0.1
        return -0.3

    # ── Signal 25 — Open Interest Change ────────────────────────────────
    def _oi_change(self, oi, oi_1h_ago, candles) -> float:
        if oi is None or oi_1h_ago is None or float(oi_1h_ago) == 0:
            return 0.0
        oi_now = float(oi)
        oi_ago = float(oi_1h_ago)
        change = (oi_now - oi_ago) / oi_ago
        closes = candles["close"].values.astype(float)
        price_rising = len(closes) >= 2 and closes[-1] > closes[-12] if len(closes) >= 12 else False
        if change > 0.05 and price_rising:
            return -0.1
        elif change > 0.05 and not price_rising:
            return -0.3
        elif change < -0.05:
            return 0.2
        return 0.1

    # ── Signal 26 — Liquidation Pressure ────────────────────────────────
    def _liquidation_pressure(self, liquidations) -> float:
        if liquidations is None or not liquidations:
            return 0.0
        long_liq  = sum(l.get("amount", 0) for l in liquidations if l.get("side") == "long")
        short_liq = sum(l.get("amount", 0) for l in liquidations if l.get("side") == "short")
        total = long_liq + short_liq
        if total == 0:
            return 0.0
        self.liquidation_cascade = long_liq / total > 0.8
        if self.liquidation_cascade:
            return -0.8
        if short_liq / total > 0.8:
            return 0.3
        return 0.0

    # ── Signal 27 — VWAP Position ────────────────────────────────────────
    def _vwap_position(self, candles: pd.DataFrame) -> float:
        vwap = self._calc_vwap(candles)
        if vwap == 0:
            return 0.0
        self.vwap_price = vwap
        price = float(candles["close"].iloc[-1])
        dist  = (price - vwap) / vwap
        self.distance_from_vwap_pct = dist
        if dist > 0.005:    return -0.4
        elif dist > 0.001:  return 0.1
        elif abs(dist) <= 0.001: return 0.3
        elif dist >= -0.005: return 0.2
        return 0.4

    # ── Signal 28 — VWAP Reclaim ─────────────────────────────────────────
    def _vwap_reclaim(self, candles: pd.DataFrame) -> float:
        if len(candles) < 3:
            return 0.0
        vwap = self._calc_vwap(candles)
        if vwap == 0:
            return 0.0
        closes = candles["close"].values.astype(float)
        vols   = candles["volume"].values.astype(float)
        avg_vol = vols[-20:].mean() if len(vols) >= 20 else vols.mean()
        prev  = closes[-2]
        curr  = closes[-1]
        vol_now = vols[-1]
        with_vol = vol_now > avg_vol * 1.5
        if prev < vwap and curr >= vwap and with_vol:
            return 0.7
        elif prev > vwap and curr <= vwap and with_vol:
            return -0.7
        return 0.0

    def _calc_vwap(self, candles: pd.DataFrame) -> float:
        df = candles.copy()
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            today_start = pd.Timestamp.now("UTC").normalize()
            today_df = df[df["timestamp"] >= today_start]
            if len(today_df) > 0:
                df = today_df
        typical  = (df["high"] + df["low"] + df["close"]) / 3
        cumvol   = df["volume"].cumsum()
        if cumvol.iloc[-1] == 0:
            return 0.0
        return float((typical * df["volume"]).cumsum().iloc[-1] / cumvol.iloc[-1])
