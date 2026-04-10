from datetime import datetime, timezone
from .tracker import CapitalTracker


class Compounder:
    def __init__(self, tracker: CapitalTracker):
        self._tracker = tracker
        self._start_capital = tracker.get_capital()
        self._start_time = datetime.now(timezone.utc)
        self._compound_events = 0

    def compound(self, profit_usd: float, trade_meta: dict = None):
        """Add profit back into the capital pool (auto-compounding)."""
        if profit_usd == 0:
            return
        self._tracker.update(profit_usd, trade_meta)
        self._compound_events += 1

    def get_compound_rate(self) -> float:
        """Returns effective daily return % since bot started."""
        elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        days = elapsed / 86400
        if days < 0.01:
            return 0.0
        current = self._tracker.get_capital()
        if self._start_capital <= 0:
            return 0.0
        total_return = (current - self._start_capital) / self._start_capital
        daily_rate = (1 + total_return) ** (1 / days) - 1
        return daily_rate * 100  # percent

    def project(self, days: int) -> float:
        """Projects capital after N days at current compound rate."""
        rate = self.get_compound_rate() / 100
        if rate <= 0:
            return self._tracker.get_capital()
        current = self._tracker.get_capital()
        return current * ((1 + rate) ** days)
