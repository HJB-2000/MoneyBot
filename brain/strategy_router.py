from dataclasses import dataclass, field
from typing import List, Optional


ROUTING_TABLE = {
    "RANGING": {
        "strategies": ["triangular_arb", "stat_arb", "grid_trader", "mean_reversion"],
        "size_mult": 1.0,
        "note": "Best conditions — full size all arb",
    },
    "TRENDING_UP": {
        "strategies": ["correlation_breakout", "stat_arb", "volume_spike"],
        "size_mult": 0.6,
        "stat_arb_bias": "long_only",
        "note": "Trend mode — momentum strategies",
    },
    "TRENDING_DOWN": {
        "strategies": ["mean_reversion", "stat_arb"],
        "size_mult": 0.6,
        "stat_arb_bias": "short_only",
        "note": "Downtrend mode",
    },
    "BREAKOUT": {
        "strategies": ["correlation_breakout", "volume_spike"],
        "size_mult": 0.5,
        "note": "Breakout mode — ride momentum",
    },
    "VOLATILE": {
        "strategies": [],
        "size_mult": 0.0,
        "note": "Capital protection — no trades",
    },
    "WHALE_MOVING": {
        "strategies": ["correlation_breakout"],
        "size_mult": 0.4,
        "direction": "follow_whale",
        "note": "Follow whale direction only",
    },
    "FUNDING_RICH": {
        "strategies": ["triangular_arb", "stat_arb", "funding_arb", "grid_trader"],
        "size_mult": 1.0,
        "funding_priority": True,
        "note": "High funding rate — arb + funding strategies",
    },
    "CHOPPY": {
        "strategies": ["triangular_arb", "grid_trader"],
        "size_mult": 0.3,
        "note": "Low confidence — only best setups",
    },
}


@dataclass
class RouteResult:
    strategies: List[str] = field(default_factory=list)
    size_mult: float = 0.0
    bias: Optional[str] = None
    note: str = ""
    regime: str = ""
    confidence: float = 0.0


class StrategyRouter:
    def route(self, regime: str, confidence: float) -> RouteResult:
        entry = ROUTING_TABLE.get(regime, ROUTING_TABLE["CHOPPY"])
        size_mult = entry["size_mult"]

        # Low confidence → halve the size regardless of regime
        if confidence < 0.4:
            size_mult *= 0.5

        return RouteResult(
            strategies=list(entry.get("strategies", [])),
            size_mult=round(size_mult, 4),
            bias=entry.get("stat_arb_bias") or entry.get("direction"),
            note=entry.get("note", ""),
            regime=regime,
            confidence=confidence,
        )
