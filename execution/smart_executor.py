from dataclasses import dataclass
from datetime import datetime, timezone

from strategies.base_strategy import Opportunity
from execution.fill_simulator import FillSimulator
from execution.order_manager import OrderManager


@dataclass
class ExecutionResult:
    executed: bool
    reason: str
    net_pnl_usd: float = 0.0
    fill: dict = None


class SmartExecutor:
    def __init__(self, market_reader, risk_manager, scorer,
                 tracker, order_manager: OrderManager,
                 config: dict, signal_objects_ref: dict = None):
        self._mr = market_reader
        self._risk = risk_manager
        self._scorer = scorer
        self._tracker = tracker
        self._orders = order_manager
        self._sim = FillSimulator()
        self._cfg = config
        self._signal_objects_ref = signal_objects_ref or {}

    def execute(self, opp: Opportunity, signals: dict,
                combiner_result, signal_objects: dict,
                regime: str) -> ExecutionResult:
        """
        6 pre-execution checks before paper fill.
        """
        # Check 1 — Re-fetch orderbook, recalculate profit
        ob = self._mr.get_orderbook(opp.pair.split("→")[0] + "/USDT"
                                     if "→" in opp.pair else opp.pair, 5)

        # Check 2 — price staleness
        if ob and ob.get("asks"):
            current_price = ob["asks"][0][0]
            if opp.entry_price > 0:
                price_drift = abs(current_price - opp.entry_price) / opp.entry_price
                if price_drift > 0.0015:
                    return ExecutionResult(False, "stale_price")

        # Check 3 — expiry
        age = (datetime.now(timezone.utc) - opp.detected_at).total_seconds()
        if age > opp.expiry_seconds:
            return ExecutionResult(False, "expired")

        # Check 4 — re-score
        new_score = self._scorer.score(
            opp, regime, signals, combiner_result, signal_objects
        )
        exec_threshold = self._cfg["scoring"]["execution_threshold"]
        if new_score < exec_threshold:
            return ExecutionResult(False, f"score_below_threshold:{new_score:.3f}")

        # Check 5 — re-run risk approval
        approval = self._risk.approve(opp, signal_objects)
        if not approval.approved:
            return ExecutionResult(False, f"risk:{approval.reason}")

        # Check 6 — API latency
        if self._mr.avg_latency_ms > self._cfg["risk"]["max_api_latency_ms"]:
            return ExecutionResult(False, "high_latency")

        # --- All checks passed: simulate fill ---
        fill = self._sim.simulate(opp)
        net_pnl = fill["net_pnl_usd"]

        # Record open position
        self._orders.open(opp)
        self._risk.record_open(opp)

        # For instant strategies (triangular arb, volume spike with short expiry),
        # close immediately and settle P&L.
        if opp.expiry_seconds <= 20 or opp.strategy == "triangular_arb":
            self._close_immediate(opp, fill)
            return ExecutionResult(True, "executed_and_closed", net_pnl, fill)

        # Longer-hold strategies: position stays open for monitor thread to close.
        return ExecutionResult(True, "executed_open", net_pnl, fill)

    def _close_immediate(self, opp: Opportunity, fill: dict):
        """For fast strategies: close instantly, update capital."""
        self._orders.close(opp.id)
        self._risk.record_close(opp.id, fill["result"])
        meta = {
            "strategy": opp.strategy,
            "pair": opp.pair,
            "direction": opp.direction,
            "entry_price": fill["entry_price"],
            "exit_price": fill["exit_price"],
            "size_usd": opp.trade_size_usd,
            "gross_profit_pct": fill["gross_profit_pct"],
            "fees_pct": fill["fees_pct"],
            "slippage_pct": fill["slippage_pct"],
            "net_profit_pct": fill["net_profit_pct"],
            "regime": opp.regime,
            "confidence": opp.confidence,
            "score": opp.score,
            "hold_seconds": 0,
            "result": fill["result"],
        }
        self._tracker.update(fill["net_pnl_usd"], meta)
