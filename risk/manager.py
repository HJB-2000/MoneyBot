import time
from dataclasses import dataclass
from datetime import datetime, timezone

from strategies.base_strategy import Opportunity


@dataclass
class ApprovalResult:
    approved: bool
    size: float
    reason: str


class RiskManager:
    def __init__(self, config: dict, tracker, position_sizer):
        self._cfg = config["risk"]
        self._capital_cfg = config["capital"]
        self._tracker = tracker
        self._sizer = position_sizer
        self._open_trades: list = []
        self._cb1_active: bool = False
        self._cb3_active: bool = False
        self._cb2_halted_until: float = 0.0
        self._vol_pause_until: float = 0.0
        self._last_opp_time: datetime = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    #  Main approval gate                                                  #
    # ------------------------------------------------------------------ #

    def approve(self, opp: Opportunity, signal_objects: dict = None) -> ApprovalResult:
        capital = self._tracker.get_capital()

        # Gate 1 — capital floor
        if capital < self._capital_cfg["survival_floor"] + 0.01:
            return ApprovalResult(False, 0.0, "Capital below survival floor")

        # Gate 2 — daily loss limit
        daily_pnl = self._tracker.get_daily_pnl()
        if daily_pnl / (capital - daily_pnl + 1e-9) < -self._cfg["max_daily_loss_pct"]:
            return ApprovalResult(False, 0.0, "Daily loss limit reached")

        # Gate 3 — max open trades
        if len(self._open_trades) >= self._cfg["max_open_trades"]:
            return ApprovalResult(False, 0.0, "Max open trades reached")

        # Gate 4 — circuit breakers
        if time.time() < self._cb2_halted_until:
            return ApprovalResult(False, 0.0, "Circuit breaker CB2: daily drawdown")

        drawdown = self._tracker.get_drawdown()
        if drawdown > self._cfg["circuit_breaker_drawdown"]:
            self._cb2_halted_until = time.time() + 86400
            return ApprovalResult(False, 0.0, "Circuit breaker CB2 triggered: drawdown > 15%")

        losses = self._tracker.get_consecutive_losses()
        self._cb1_active = losses >= self._cfg["circuit_breaker_losses"]

        win_rate = self._tracker.get_win_rate(20)
        self._cb3_active = (win_rate < self._cfg["circuit_breaker_win_rate"]
                             and win_rate > 0)  # only if we have data

        # Gate 5 — volatility guard
        if time.time() < self._vol_pause_until:
            return ApprovalResult(False, 0.0, "Volatility pause active")

        # Gate 6 — API latency
        vol_obj = (signal_objects or {}).get("volatility")
        if vol_obj and hasattr(vol_obj, "_mr_latency"):
            pass  # latency check done via market_reader in executor

        # --- All gates passed: get size ---
        size = self._sizer.get_size(
            capital=capital,
            signal_objects=signal_objects,
            win_rate=win_rate,
            avg_win=0.01,    # default estimates until we have data
            avg_loss=0.01,
            regime_mult=1.0,  # applied later by engine
        )

        # Apply circuit breaker reductions
        if self._cb1_active and self._cb3_active:
            size *= 0.35
        elif self._cb1_active:
            size *= 0.5
        elif self._cb3_active:
            size *= 0.7

        size = max(size, 2.0)   # never go below $2
        opp.trade_size_usd = size
        return ApprovalResult(True, size, "approved")

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #

    def record_open(self, opp: Opportunity):
        self._open_trades.append(opp.id)
        self._last_opp_time = datetime.now(timezone.utc)

    def record_close(self, opp_id: str, result: str):
        if opp_id in self._open_trades:
            self._open_trades.remove(opp_id)

    def trigger_vol_pause(self):
        self._vol_pause_until = time.time() + self._cfg["volatility_pause_seconds"]

    @property
    def open_trade_count(self) -> int:
        return len(self._open_trades)

    @property
    def cb1_active(self) -> bool:
        return self._cb1_active

    @property
    def cb3_active(self) -> bool:
        return self._cb3_active
