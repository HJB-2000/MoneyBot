import time
import numpy as np


class WhaleDetectorSignal:
    SMALL_THRESHOLD_USD = 1_000
    LARGE_THRESHOLD_USD = 50_000

    def __init__(self):
        self.whale_buying = False
        self.whale_selling = False
        self.iceberg_detected = False
        self._ob_history: list = []  # ring buffer of recent orderbook snapshots

    def calculate(self, trades: list, orderbook: dict) -> float:
        try:
            if not trades or not orderbook:
                return 0.0

            # Snapshot orderbook for iceberg detection
            self._ob_history.append({
                "ts": time.time(),
                "bids": list(orderbook.get("bids", [])[:5]),
                "asks": list(orderbook.get("asks", [])[:5]),
            })
            if len(self._ob_history) > 20:
                self._ob_history.pop(0)

            accum_score = self._whale_accumulation(trades) * 0.35
            iceberg_score = self._iceberg_detection(orderbook, trades) * 0.25
            spoof_score = self._spoofing_detection() * 0.20
            inflow_score = 0.0  # exchange inflows not available via public API  * 0.20

            result = accum_score + iceberg_score + spoof_score + inflow_score
            return float(np.clip(result, -1.0, 1.0))
        except Exception:
            return 0.0

    def _classify_by_size(self, trades: list):
        """Estimate USD value per trade using price * amount."""
        small, medium, large = [], [], []
        for t in trades:
            price = t.get("price", 0)
            amount = t.get("amount", 0)
            usd_val = price * amount
            if usd_val < self.SMALL_THRESHOLD_USD:
                small.append(t)
            elif usd_val < self.LARGE_THRESHOLD_USD:
                medium.append(t)
            else:
                large.append(t)
        return small, medium, large

    def _whale_accumulation(self, trades: list) -> float:
        _, _, large = self._classify_by_size(trades)
        if not large:
            self.whale_buying = False
            self.whale_selling = False
            return 0.0
        buy_vol = sum(t.get("amount", 0) for t in large if t.get("side") == "buy")
        sell_vol = sum(t.get("amount", 0) for t in large if t.get("side") == "sell")
        self.whale_buying = buy_vol > sell_vol * 1.5
        self.whale_selling = sell_vol > buy_vol * 1.5
        if self.whale_buying:
            return 0.5
        elif self.whale_selling:
            return -0.5
        return 0.0

    def _iceberg_detection(self, orderbook: dict, trades: list) -> float:
        """Detect when a level is repeatedly filled and refilled."""
        if len(self._ob_history) < 3:
            return 0.0
        current = self._ob_history[-1]
        previous = self._ob_history[-3]

        # Check bid side
        if current["bids"] and previous["bids"]:
            curr_bid = current["bids"][0][0] if current["bids"] else None
            prev_bid = previous["bids"][0][0] if previous["bids"] else None
            curr_bid_size = current["bids"][0][1] if current["bids"] else 0
            prev_bid_size = previous["bids"][0][1] if previous["bids"] else 0
            if (curr_bid == prev_bid and curr_bid_size >= prev_bid_size * 0.9
                    and curr_bid_size > 0):
                self.iceberg_detected = True
                return 0.4  # bid iceberg = large buyer hiding

        # Check ask side
        if current["asks"] and previous["asks"]:
            curr_ask = current["asks"][0][0] if current["asks"] else None
            prev_ask = previous["asks"][0][0] if previous["asks"] else None
            curr_ask_size = current["asks"][0][1] if current["asks"] else 0
            prev_ask_size = previous["asks"][0][1] if previous["asks"] else 0
            if (curr_ask == prev_ask and curr_ask_size >= prev_ask_size * 0.9
                    and curr_ask_size > 0):
                self.iceberg_detected = True
                return -0.4  # ask iceberg = large seller hiding

        self.iceberg_detected = False
        return 0.0

    def _spoofing_detection(self) -> float:
        """A large order that appeared and disappeared quickly = spoof."""
        if len(self._ob_history) < 4:
            return 0.0
        h = self._ob_history

        # Look for large bid that appeared 2 snapshots ago and is now gone
        try:
            old_bids = {b[0]: b[1] for b in h[-3]["bids"]}
            new_bids = {b[0]: b[1] for b in h[-1]["bids"]}
            for price, size in old_bids.items():
                if size > 0 and price not in new_bids:
                    return -0.3   # bid spoof removed → real direction likely DOWN

            old_asks = {a[0]: a[1] for a in h[-3]["asks"]}
            new_asks = {a[0]: a[1] for a in h[-1]["asks"]}
            for price, size in old_asks.items():
                if size > 0 and price not in new_asks:
                    return 0.3    # ask spoof removed → real direction likely UP
        except Exception:
            pass
        return 0.0
