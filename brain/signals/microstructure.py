import time
import numpy as np


class MicrostructureSignal:
    def __init__(self):
        self.whale_detected = False
        self.spread_pct = 0.0
        self._prev_spread_pct = 0.0
        self._prev_spread_time = 0.0

    def calculate(self, orderbook: dict, trades: list) -> float:
        try:
            if not orderbook or not trades:
                return 0.0
            ofi = self._order_flow_imbalance(trades) * 0.20
            spread = self._spread_score(orderbook) * 0.40
            depth = self._depth_imbalance(orderbook) * 0.25
            whale = self._large_order_detection(trades) * 0.15
            result = ofi + spread + depth + whale
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _order_flow_imbalance(self, trades: list) -> float:
        if not trades:
            return 0.0
        buy_vol = sum(t["amount"] for t in trades if t.get("side") == "buy")
        sell_vol = sum(t["amount"] for t in trades if t.get("side") == "sell")
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        ofi = (buy_vol - sell_vol) / total
        if abs(ofi) <= 0.1:
            return 0.5   # balanced: perfect for arb
        elif ofi > 0.4:
            return 0.3   # buyers aggressive
        elif ofi < -0.4:
            return -0.3
        return 0.0

    def _spread_score(self, orderbook: dict) -> float:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return -0.5
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return 0.0
        self.spread_pct = (best_ask - best_bid) / mid

        now = time.time()
        if now - self._prev_spread_time < 300 and self._prev_spread_pct > 0:
            widening = self.spread_pct > self._prev_spread_pct * 1.5
        else:
            widening = False
        self._prev_spread_pct = self.spread_pct
        self._prev_spread_time = now

        if widening:
            return -0.7
        if self.spread_pct < 0.0005:
            return 0.6
        elif self.spread_pct < 0.002:
            return 0.2
        else:
            return -0.7

    def _depth_imbalance(self, orderbook: dict, levels: int = 10) -> float:
        bids = orderbook.get("bids", [])[:levels]
        asks = orderbook.get("asks", [])[:levels]
        if not bids or not asks:
            return 0.0
        sum_bids = sum(b[1] for b in bids)
        sum_asks = sum(a[1] for a in asks)
        total = sum_bids + sum_asks
        if total == 0:
            return 0.0
        ratio = sum_bids / total
        if 0.45 <= ratio <= 0.55:
            return 0.4   # balanced = best for arb
        elif ratio > 0.65:
            return 0.2   # bid heavy = supported
        elif ratio < 0.35:
            return -0.3
        return 0.0

    def _large_order_detection(self, trades: list) -> float:
        if not trades:
            return 0.0
        now = time.time()
        recent = [t for t in trades if now - t.get("timestamp", 0) / 1000 < 60]
        if not recent:
            recent = trades  # fallback: use all provided
        total_usd = sum(t.get("amount", 0) * t.get("price", 1.0) for t in recent)
        if total_usd == 0:
            return 0.0
        threshold = total_usd * 0.60  # single trade must be >60% of flow to qualify as whale

        whale_buys = sum(t["amount"] for t in recent
                         if t.get("amount", 0) * t.get("price", 1.0) > threshold and t.get("side") == "buy")
        whale_sells = sum(t["amount"] for t in recent
                          if t.get("amount", 0) * t.get("price", 1.0) > threshold and t.get("side") == "sell")

        self.whale_detected = (whale_buys + whale_sells) > 0
        score = 0.0
        if self.whale_detected:
            score -= 0.5  # any whale activity = caution
            if whale_buys > whale_sells:
                score += 0.2   # net whale buying
            elif whale_sells > whale_buys:
                score -= 0.3   # net whale selling
        return score
