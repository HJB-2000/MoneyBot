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

    def check_exits(self, market_reader) -> list:
        """
        Returns list of (position_dict, exit_reason) for positions
        that should be closed.
        """
        to_close = []
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

                reason = None
                direction = pos["direction"]

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
