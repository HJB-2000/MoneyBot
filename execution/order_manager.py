import json
import os
import threading
from datetime import datetime, timezone

from strategies.base_strategy import Opportunity

ORDERS_FILE = "data/open_orders.json"


class OrderManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._positions: dict = self._load()

    def open(self, opp: Opportunity):
        with self._lock:
            self._positions[opp.id] = {
                "id": opp.id,
                "strategy": opp.strategy,
                "pair": opp.pair,
                "direction": opp.direction,
                "entry_price": opp.entry_price,
                "size_usd": opp.trade_size_usd,
                "target_price": opp.target_price,
                "stop_price": opp.stop_price,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "hold_max_seconds": opp.hold_max_seconds,
                "regime": opp.regime,
                "score": opp.score,
            }
            self._save()

    def close(self, opp_id: str) -> dict:
        with self._lock:
            pos = self._positions.pop(opp_id, None)
            self._save()
            return pos

    def get_all(self) -> dict:
        with self._lock:
            return dict(self._positions)

    # Trailing stop config: activate once price moves TRAIL_ACTIVATE in profit;
    # keep stop TRAIL_DISTANCE below (long) / above (short) the running best price.
    TRAIL_ACTIVATE = 0.003   # 0.3% profit before trailing kicks in
    TRAIL_DISTANCE = 0.005   # trail stop 0.5% from running best price

    def check_exits(self, market_reader) -> list:
        """
        Returns list of (position_dict, exit_reason) for positions
        that should be closed.  Also updates trailing stops in-place.
        """
        to_close = []
        dirty = False
        with self._lock:
            for opp_id, pos in list(self._positions.items()):
                pair = pos["pair"]
                ticker = market_reader.get_ticker(pair)
                if ticker is None:
                    continue
                price = ticker.get("last", 0)
                if price == 0:
                    continue

                opened_at = datetime.fromisoformat(pos["opened_at"])
                hold_secs = (datetime.now(timezone.utc) - opened_at).total_seconds()
                direction = pos["direction"]
                entry = pos.get("entry_price", price) or price

                # ── trailing stop update ─────────────────────────────────
                if entry > 0:
                    if direction == "long":
                        best = pos.get("trail_high", entry)
                        best = max(best, price)
                        pos["trail_high"] = best
                        # activate once we're TRAIL_ACTIVATE above entry
                        if best >= entry * (1 + self.TRAIL_ACTIVATE):
                            trail_stop = best * (1 - self.TRAIL_DISTANCE)
                            # only move stop up, never down
                            old_stop = pos.get("stop_price") or 0
                            if trail_stop > old_stop:
                                pos["stop_price"] = trail_stop
                                dirty = True
                    elif direction == "short":
                        best = pos.get("trail_low", entry)
                        best = min(best, price)
                        pos["trail_low"] = best
                        if best <= entry * (1 - self.TRAIL_ACTIVATE):
                            trail_stop = best * (1 + self.TRAIL_DISTANCE)
                            old_stop = pos.get("stop_price") or float("inf")
                            if trail_stop < old_stop:
                                pos["stop_price"] = trail_stop
                                dirty = True

                # ── exit checks ─────────────────────────────────────────
                reason = None
                if direction == "long":
                    if pos["target_price"] and price >= pos["target_price"]:
                        reason = "target"
                    elif pos["stop_price"] and price <= pos["stop_price"]:
                        reason = "stop"
                elif direction == "short":
                    if pos["target_price"] and price <= pos["target_price"]:
                        reason = "target"
                    elif pos["stop_price"] and price >= pos["stop_price"]:
                        reason = "stop"

                if hold_secs >= pos.get("hold_max_seconds", 3600):
                    reason = "time_stop"

                if reason:
                    to_close.append((pos, reason, price))

            if dirty:
                self._save()

        return to_close

    def _load(self) -> dict:
        if os.path.exists(ORDERS_FILE) and os.path.getsize(ORDERS_FILE) > 0:
            try:
                with open(ORDERS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        with open(ORDERS_FILE, "w") as f:
            json.dump(self._positions, f, default=str)
