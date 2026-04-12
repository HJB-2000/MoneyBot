import numpy as np


class SentimentSignal:
    def __init__(self):
        self.funding_rate = 0.0
        self.long_liquidation_cascade = False

    def calculate(self, funding: float, liquidations: list, open_interest: float,
                  open_interest_1h_ago: float = None) -> float:
        try:
            self.funding_rate = funding if funding is not None else 0.0
            funding_score = self._funding_rate_score(self.funding_rate) * 0.35
            liq_score = self._liquidation_pressure(liquidations) * 0.55
            oi_score = self._open_interest_trend(open_interest, open_interest_1h_ago) * 0.10
            result = funding_score + liq_score + oi_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _funding_rate_score(self, rate: float) -> float:
        if rate > 0.0008:
            return 0.3   # longs paying heavily
        elif rate > 0.0002:
            return 0.1
        elif rate >= -0.0002:
            return 0.0
        elif rate < -0.0005:
            return -0.2
        return 0.0

    def _liquidation_pressure(self, liquidations: list) -> float:
        if liquidations is None:
            self.long_liquidation_cascade = False
            return 0.0
        if not liquidations:
            self.long_liquidation_cascade = False
            return 0.0   # no liquidations = neutral

        long_liq = sum(l.get("amount", 0) for l in liquidations if l.get("side") == "long")
        short_liq = sum(l.get("amount", 0) for l in liquidations if l.get("side") == "short")
        total = long_liq + short_liq
        if total == 0:
            return 0.0  # liquidations present but no side data = neutral

        self.long_liquidation_cascade = long_liq / total > 0.8

        if self.long_liquidation_cascade:
            return -1.0   # cascade of long liq = price dropping fast
        elif short_liq / total > 0.8:
            return 0.3    # short squeeze
        return 0.0

    def _open_interest_trend(self, oi_now: float, oi_1h_ago: float) -> float:
        if oi_now is None or oi_1h_ago is None or oi_1h_ago == 0:
            return 0.0
        change = (oi_now - oi_1h_ago) / oi_1h_ago
        if change > 0.05:
            return -0.1   # new positions entering = caution
        elif change < -0.05:
            return 0.1    # positions closing
        return 0.0
