"""
Trade signal group — 5 signals from recent trade data.
All scores in [-1.0, +1.0].
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TradeSignals:
    def __init__(self):
        self.cvd_divergence: bool = False
        self.cvd_direction: str = "flat"
        self.volume_spike: bool = False
        self.whale_buying: bool = False
        self.whale_selling: bool = False

    def calculate(self, trades: list, candles: pd.DataFrame) -> dict:
        zeros = {k: 0.0 for k in [
            "CVD_divergence", "CVD_trend", "volume_spike_signal",
            "buy_sell_ratio", "large_trade_flow",
        ]}
        try:
            if not trades or candles is None or len(candles) < 5:
                return zeros
            closes = candles["close"].values.astype(float)
            opens  = candles["open"].values.astype(float)
            vols   = candles["volume"].values.astype(float)
            return {
                "CVD_divergence":     self._cvd_divergence(trades, closes, opens, vols),
                "CVD_trend":          self._cvd_trend(closes, opens, vols),
                "volume_spike_signal": self._volume_spike(vols),
                "buy_sell_ratio":     self._buy_sell_ratio(trades),
                "large_trade_flow":   self._large_trade_flow(trades),
            }
        except Exception:
            return zeros

    def _build_cvd(self, closes, opens, vols):
        delta = np.where(closes > opens, vols, np.where(closes < opens, -vols, 0.0))
        return np.cumsum(delta)

    # ── Signal 19 — CVD Divergence ───────────────────────────────────────
    def _cvd_divergence(self, trades, closes, opens, vols) -> float:
        if len(closes) < 5:
            return 0.0
        cvd = self._build_cvd(closes, opens, vols)
        # buy/sell balance from actual trades
        buy_vol  = sum(t.get("amount", 0) for t in trades if t.get("side") == "buy")
        sell_vol = sum(t.get("amount", 0) for t in trades if t.get("side") == "sell")
        net_trade_flow = buy_vol - sell_vol  # positive = buying pressure

        price_falling = closes[-1] < closes[-4]
        price_rising  = closes[-1] > closes[-4]
        cvd_rising    = cvd[-1] > cvd[-4]
        cvd_falling   = cvd[-1] < cvd[-4]

        if price_falling and (cvd_rising or net_trade_flow > 0):
            self.cvd_divergence = True
            self.cvd_direction = "bullish"
            return 0.9
        elif price_rising and (cvd_falling or net_trade_flow < 0):
            self.cvd_divergence = True
            self.cvd_direction = "bearish"
            return -0.9
        elif price_rising and cvd_rising:
            self.cvd_divergence = False
            self.cvd_direction = "up"
            return 0.3
        elif price_falling and cvd_falling:
            self.cvd_divergence = False
            self.cvd_direction = "down"
            return -0.3
        self.cvd_divergence = False
        self.cvd_direction = "flat"
        return 0.0

    # ── Signal 20 — CVD Trend ────────────────────────────────────────────
    def _cvd_trend(self, closes, opens, vols, period: int = 12) -> float:
        if len(closes) < period:
            return 0.0
        cvd = self._build_cvd(closes, opens, vols)
        c = cvd[-period:]
        if len(c) < 2:
            return 0.0
        slope = np.polyfit(range(len(c)), c, 1)[0]
        std = c.std()
        if std == 0:
            return 0.1
        norm = slope / std
        if norm > 1.0:   return 0.5
        elif norm > 0.3: return 0.2
        elif norm < -1.0: return -0.5
        elif norm < -0.3: return -0.2
        return 0.1   # flat = ok for arb

    # ── Signal 21 — Volume Spike ─────────────────────────────────────────
    def _volume_spike(self, vols: np.ndarray) -> float:
        if len(vols) < 24:
            return 0.0
        current  = vols[-1]
        avg_24h  = vols[-288:].mean() if len(vols) >= 288 else vols.mean()
        if avg_24h == 0:
            return 0.0
        ratio = current / avg_24h
        self.volume_spike = bool(ratio > 3.0)
        if ratio > 5.0:
            return -0.3
        elif ratio >= 3.0:
            return 0.4
        elif ratio >= 1.5:
            return 0.1
        return 0.3  # calm = good for arb

    # ── Signal 22 — Buy/Sell Ratio ───────────────────────────────────────
    def _buy_sell_ratio(self, trades: list) -> float:
        if not trades:
            return 0.0
        buys  = sum(1 for t in trades if t.get("side") == "buy")
        total = len(trades)
        if total == 0:
            return 0.0
        ratio = buys / total
        if 0.45 <= ratio <= 0.55:
            return 0.5
        elif 0.55 < ratio <= 0.65:
            return 0.2
        elif 0.65 < ratio <= 0.80:
            return 0.0
        elif ratio > 0.80:
            return -0.2
        elif 0.35 <= ratio < 0.45:
            return -0.2
        elif 0.20 <= ratio < 0.35:
            return 0.0
        return 0.2

    # ── Signal 23 — Large Trade Flow ─────────────────────────────────────
    def _large_trade_flow(self, trades: list) -> float:
        if not trades:
            return 0.0
        def usd(t):
            return t.get("amount", 0) * t.get("price", 1.0)

        large = [t for t in trades if usd(t) > 50_000]
        if not large:
            self.whale_buying  = False
            self.whale_selling = False
            return 0.0

        large_buy_usd  = sum(usd(t) for t in large if t.get("side") == "buy")
        large_sell_usd = sum(usd(t) for t in large if t.get("side") == "sell")
        self.whale_buying  = large_buy_usd  > large_sell_usd and large_buy_usd > 0
        self.whale_selling = large_sell_usd > large_buy_usd  and large_sell_usd > 0

        if self.whale_buying:
            return 0.4
        elif self.whale_selling:
            return -0.4
        return 0.0
