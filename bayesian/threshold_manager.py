"""
Dynamic threshold manager — adapts execution threshold to conditions.
"""
from __future__ import annotations


REGIME_ADJ = {
    "RANGING":       -0.05,
    "TRENDING_UP":   +0.03,
    "TRENDING_DOWN": +0.03,
    "BREAKOUT":      +0.05,
    "VOLATILE":      +0.99,
    "FUNDING_RICH":  -0.03,
    "CHOPPY":        +0.08,
    "WHALE_MOVING":  +0.05,
}


class ThresholdManager:
    def get_threshold(self, regime: str, capital: float,
                      win_rate: float, consecutive_losses: int) -> float:
        base = 0.65

        # Regime adjustment
        regime_adj = REGIME_ADJ.get(regime, 0.0)

        # Performance adjustment
        if win_rate > 0.70:
            perf_adj = -0.03
        elif win_rate >= 0.60:
            perf_adj = 0.00
        elif win_rate >= 0.50:
            perf_adj = +0.04
        else:
            perf_adj = +0.08

        # Capital adjustment
        if capital < 100:
            cap_adj = 0.00
        elif capital < 500:
            cap_adj = -0.02
        else:
            cap_adj = -0.04

        # Consecutive loss adjustment
        if consecutive_losses >= 5:
            loss_adj = +0.10
        elif consecutive_losses >= 3:
            loss_adj = +0.05
        else:
            loss_adj = 0.00

        final = base + regime_adj + perf_adj + cap_adj + loss_adj

        # VOLATILE regime effectively blocks all trades
        if regime == "VOLATILE":
            return 0.99

        return max(0.55, min(0.85, final))
