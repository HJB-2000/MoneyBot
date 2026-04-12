import threading
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
        self._lock = threading.RLock()            # protects all mutable state below
        self._open_trades: list = []              # list of trade IDs
        self._open_pairs: set = set()             # pairs with at least one open position
        self._open_trade_pairs: dict = {}         # trade_id → pair
        self._open_trade_strategies: dict = {}    # trade_id → strategy name
        self._cb1_active: bool = False
        self._cb3_active: bool = False
        self._cb2_halted_until: float = 0.0
        self._vol_pause_until: float = 0.0
        self._last_opp_time: datetime = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    #  Main approval gate                                                  #
    # ------------------------------------------------------------------ #

    def approve(self, opp: Opportunity, signal_objects: dict = None) -> ApprovalResult:
        with self._lock:
            capital = self._tracker.get_capital()

            # Gate 1 — capital floor
            if capital < self._capital_cfg["survival_floor"] + 0.01:
                return ApprovalResult(False, 0.0, "Capital below survival floor")

            # Gate 2 — daily loss limit
            daily_pnl = self._tracker.get_daily_pnl()
            if daily_pnl / (capital - daily_pnl + 1e-9) < -self._cfg["max_daily_loss_pct"]:
                return ApprovalResult(False, 0.0, "Daily loss limit reached")

            # Gate 3a — max open trades (global)
            if len(self._open_trades) >= self._cfg["max_open_trades"]:
                return ApprovalResult(False, 0.0, "Max open trades reached")

            # Gate 3b — no duplicate pair (prevents LINK/USDT opened twice)
            if opp.pair in self._open_pairs:
                return ApprovalResult(False, 0.0, "Pair already has open position")

            # Gate 3c — per-strategy cap for stat_arb (max 1 position)
            if opp.strategy == "stat_arb":
                stat_arb_open = sum(
                    1 for s in self._open_trade_strategies.values()
                    if s == "stat_arb"
                )
                max_sa = self._cfg.get("max_stat_arb_positions", 1)
                if stat_arb_open >= max_sa:
                    return ApprovalResult(False, 0.0, "stat_arb position limit reached")

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

            # --- All gates passed: get size ---
            size = self._sizer.get_size(
                capital=capital,
                signal_objects=signal_objects,
                win_rate=win_rate,
                avg_win=0.01,
                avg_loss=0.01,
                regime_mult=1.0,
            )

            # Apply circuit breaker reductions
            if self._cb1_active and self._cb3_active:
                size *= 0.35
            elif self._cb1_active:
                size *= 0.5
            elif self._cb3_active:
                size *= 0.7

            size = max(size, 2.0)
            opp.trade_size_usd = size
            return ApprovalResult(True, size, "approved")

    # ------------------------------------------------------------------ #
    #  State updates                                                       #
    # ------------------------------------------------------------------ #

    def record_open(self, opp: Opportunity):
        with self._lock:
            self._open_trades.append(opp.id)
            self._open_pairs.add(opp.pair)
            self._open_trade_pairs[opp.id] = opp.pair
            self._open_trade_strategies[opp.id] = opp.strategy
            self._last_opp_time = datetime.now(timezone.utc)

    def record_close(self, opp_id: str, result: str):
        with self._lock:
            if opp_id in self._open_trades:
                self._open_trades.remove(opp_id)
            self._open_trade_strategies.pop(opp_id, None)
            pair = self._open_trade_pairs.pop(opp_id, None)
            if pair:
                # Only remove pair from open set if no other open trade holds it
                still_open = any(p == pair for p in self._open_trade_pairs.values())
                if not still_open:
                    self._open_pairs.discard(pair)

    def trigger_vol_pause(self):
        self._vol_pause_until = time.time() + self._cfg["volatility_pause_seconds"]

    @property
    def open_trade_count(self) -> int:
        with self._lock:
            return len(self._open_trades)

    @property
    def cb1_active(self) -> bool:
        return self._cb1_active

    @property
    def cb3_active(self) -> bool:
        return self._cb3_active
