"""
OrderBook signal group — 8 signals from order book data.
All scores are in [-1.0, +1.0].
"""
from __future__ import annotations

import time
from collections import defaultdict


class OrderBookSignals:
    def __init__(self):
        # State for iceberg and spoofing detection across calls
        self._level_refills: dict = defaultdict(list)   # price -> [timestamps]
        self._prev_snapshot: dict = {}
        self._prev_snapshot_time: float = 0.0
        self._trade_size_usd: float = 50.0  # default trade size for liquidity ratio

    def calculate(self, orderbook: dict, trade_size_usd: float = 50.0) -> dict:
        self._trade_size_usd = trade_size_usd
        try:
            if not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
                return self._zeros()
            return {
                "bid_ask_spread":      self._bid_ask_spread(orderbook),
                "depth_imbalance":     self._depth_imbalance(orderbook),
                "order_flow_imbalance": self._order_flow_imbalance(orderbook),
                "large_order_presence": self._large_order_presence(orderbook),
                "iceberg_detection":   self._iceberg_detection(orderbook),
                "spoofing_detection":  self._spoofing_detection(orderbook),
                "liquidity_score":     self._liquidity_score(orderbook),
                "book_pressure_ratio": self._book_pressure_ratio(orderbook),
            }
        except Exception:
            return self._zeros()

    def _zeros(self) -> dict:
        return {k: 0.0 for k in [
            "bid_ask_spread", "depth_imbalance", "order_flow_imbalance",
            "large_order_presence", "iceberg_detection", "spoofing_detection",
            "liquidity_score", "book_pressure_ratio",
        ]}

    # ── Signal 1 ──────────────────────────────────────────────────────────
    def _bid_ask_spread(self, ob: dict) -> float:
        bids, asks = ob["bids"], ob["asks"]
        best_bid, best_ask = bids[0][0], asks[0][0]
        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return 0.0
        spread_pct = (best_ask - best_bid) / mid
        if spread_pct < 0.0005:
            return 0.8
        elif spread_pct < 0.002:
            return 0.2
        elif spread_pct < 0.005:
            return -0.6
        return -1.0

    # ── Signal 2 ──────────────────────────────────────────────────────────
    def _depth_imbalance(self, ob: dict, levels: int = 10) -> float:
        bids = ob["bids"][:levels]
        asks = ob["asks"][:levels]
        sum_bids = sum(b[1] for b in bids)
        sum_asks = sum(a[1] for a in asks)
        total = sum_bids + sum_asks
        if total == 0:
            return 0.0
        ratio = sum_bids / total
        if ratio > 0.8 or ratio < 0.2:
            return -0.5
        elif 0.45 <= ratio <= 0.55:
            return 0.5
        elif ratio > 0.65:
            return 0.3
        elif ratio < 0.35:
            return -0.3
        return 0.0

    # ── Signal 3 ──────────────────────────────────────────────────────────
    def _order_flow_imbalance(self, ob: dict, levels: int = 10) -> float:
        bids = ob["bids"][:levels]
        asks = ob["asks"][:levels]
        buy_vol = sum(b[1] for b in bids)
        sell_vol = sum(a[1] for a in asks)
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0
        ofi = (buy_vol - sell_vol) / total
        if abs(ofi) <= 0.1:
            return 0.5
        elif ofi > 0.4:
            return 0.2
        elif ofi < -0.4:
            return -0.2
        return 0.0

    # ── Signal 4 ──────────────────────────────────────────────────────────
    def _large_order_presence(self, ob: dict, levels: int = 20) -> float:
        bids = ob["bids"][:levels]
        asks = ob["asks"][:levels]
        total_depth = sum(b[1] for b in bids) + sum(a[1] for a in asks)
        if total_depth == 0:
            return 0.0
        threshold = total_depth * 0.03
        large_bid = any(b[1] > threshold for b in bids)
        large_ask = any(a[1] > threshold for a in asks)
        if large_bid and not large_ask:
            return 0.3
        elif large_ask and not large_bid:
            return -0.3
        return 0.0

    # ── Signal 5 ──────────────────────────────────────────────────────────
    def _iceberg_detection(self, ob: dict) -> float:
        """
        Track bid/ask levels across calls. If same level appears 3+ times
        within 60s it's being refilled (iceberg).
        """
        now = time.time()
        cutoff = now - 60
        # record top bid and ask price levels seen
        if ob["bids"]:
            bp = round(ob["bids"][0][0], 6)
            self._level_refills[("bid", bp)].append(now)
            self._level_refills[("bid", bp)] = [
                t for t in self._level_refills[("bid", bp)] if t > cutoff
            ]
        if ob["asks"]:
            ap = round(ob["asks"][0][0], 6)
            self._level_refills[("ask", ap)].append(now)
            self._level_refills[("ask", ap)] = [
                t for t in self._level_refills[("ask", ap)] if t > cutoff
            ]
        bid_iceberg = any(
            len(v) >= 3 for k, v in self._level_refills.items() if k[0] == "bid"
        )
        ask_iceberg = any(
            len(v) >= 3 for k, v in self._level_refills.items() if k[0] == "ask"
        )
        if bid_iceberg:
            return 0.4   # buyer defending level = bullish
        if ask_iceberg:
            return -0.4
        return 0.0

    # ── Signal 6 ──────────────────────────────────────────────────────────
    def _spoofing_detection(self, ob: dict) -> float:
        """
        Compare current snapshot to previous. Large order that vanished = spoof.
        """
        now = time.time()
        bids = ob["bids"]
        asks = ob["asks"]

        if self._prev_snapshot and (now - self._prev_snapshot_time) < 2.0:
            prev_bids = self._prev_snapshot.get("bids", [])
            prev_asks = self._prev_snapshot.get("asks", [])
            # Check if a large bid disappeared (spoof on bid)
            if prev_bids and bids:
                prev_best_bid_size = prev_bids[0][1] if prev_bids else 0
                curr_best_bid_size = bids[0][1] if bids else 0
                avg_size = sum(b[1] for b in bids[:10]) / max(len(bids[:10]), 1)
                if (prev_best_bid_size > avg_size * 5 and
                        curr_best_bid_size < prev_best_bid_size * 0.3):
                    self._prev_snapshot = {"bids": bids, "asks": asks}
                    self._prev_snapshot_time = now
                    return -0.3  # spoof on bid = fake support
                if (prev_asks and prev_asks[0][1] > avg_size * 5 and
                        asks and asks[0][1] < prev_asks[0][1] * 0.3):
                    self._prev_snapshot = {"bids": bids, "asks": asks}
                    self._prev_snapshot_time = now
                    return 0.3   # spoof on ask = fake resistance

        self._prev_snapshot = {"bids": bids, "asks": asks}
        self._prev_snapshot_time = now
        return 0.0

    # ── Signal 7 ──────────────────────────────────────────────────────────
    def _liquidity_score(self, ob: dict) -> float:
        bids = ob["bids"]
        asks = ob["asks"]
        # Estimate available liquidity as total USD in top 10 bid levels
        mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else 1.0
        available_usd = sum(b[0] * b[1] for b in bids[:10])
        if self._trade_size_usd <= 0:
            return 0.0
        ratio = available_usd / self._trade_size_usd
        if ratio > 50:
            return 0.8
        elif ratio >= 20:
            return 0.5
        elif ratio >= 10:
            return 0.2
        elif ratio >= 5:
            return -0.1
        return -1.0

    # ── Signal 8 ──────────────────────────────────────────────────────────
    def _book_pressure_ratio(self, ob: dict) -> float:
        """
        Compare large orders ($10k+ equivalent) on bid vs ask side.
        """
        bids = ob["bids"]
        asks = ob["asks"]
        if not bids or not asks:
            return 0.0
        mid = (bids[0][0] + asks[0][0]) / 2
        wall_threshold = 10000 / mid if mid > 0 else 1.0  # 10k USD in units
        bid_walls = sum(b[1] for b in bids[:20] if b[1] > wall_threshold)
        ask_walls = sum(a[1] for a in asks[:20] if a[1] > wall_threshold)
        total = bid_walls + ask_walls
        if total == 0:
            return 0.0
        ratio = bid_walls / total
        if ratio > 0.65:
            return 0.4
        elif ratio < 0.35:
            return -0.4
        return 0.0
