import csv
import os
import time
from datetime import datetime, timedelta, timezone
from typing import List

from strategies.base_strategy import Opportunity

OPP_LOG = "data/opportunity_log.csv"


class OpportunityScorer:
    def __init__(self, config: dict):
        self._obs_threshold = config["scoring"]["observation_threshold"]
        self._exec_threshold = config["scoring"]["execution_threshold"]
        self._ensure_log()

    def _ensure_log(self):
        if not os.path.exists(OPP_LOG) or os.path.getsize(OPP_LOG) == 0:
            with open(OPP_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "strategy", "pair", "regime",
                    "net_profit_pct", "liquidity_ratio", "confidence",
                    "latency_ms", "score", "executed"
                ])

    def score(self, opp: Opportunity, regime: str,
              signals: dict, combiner_result, signal_objects: dict,
              pair_ranker=None) -> float:
        """Score opportunity 0.0–1.0. Returns 0.0 if hard rejected."""

        # ---- Hard rejects ----
        vol_obj = signal_objects.get("volatility")
        whale_obj = signal_objects.get("whale")

        if opp.net_profit_pct < 0.0003:
            return 0.0
        if opp.liquidity_ratio < 5.0:
            return 0.0
        if opp.exchange_latency_ms > 300:
            return 0.0
        if (datetime.now(timezone.utc) - opp.detected_at).total_seconds() > opp.expiry_seconds:
            return 0.0
        if vol_obj and getattr(vol_obj, "vol_spike", False):
            return 0.0

        # Hard reject strategy not valid in current regime
        _valid_regimes = {
            "triangular_arb":      ["RANGING", "FUNDING_RICH", "CHOPPY"],
            "stat_arb":            ["RANGING", "TRENDING_UP", "TRENDING_DOWN", "FUNDING_RICH"],
            "grid_trader":         ["RANGING", "FUNDING_RICH", "CHOPPY"],
            "mean_reversion":      ["RANGING", "TRENDING_DOWN"],
            "correlation_breakout":["TRENDING_UP", "BREAKOUT", "WHALE_MOVING"],
            "volume_spike":        ["TRENDING_UP", "BREAKOUT"],
            "funding_arb":         ["FUNDING_RICH"],
        }
        _allowed = _valid_regimes.get(opp.strategy)
        if _allowed is not None and regime not in _allowed:
            return 0.0

        if whale_obj:
            whale_dir = "buy" if getattr(whale_obj, "whale_buying", False) else \
                        "sell" if getattr(whale_obj, "whale_selling", False) else None
            if whale_dir == "sell" and opp.direction == "long":
                return 0.0
            if whale_dir == "buy" and opp.direction == "short":
                return 0.0

        # ---- Soft scoring ----

        # Filter 1: Net profit (30%)
        f1 = self._profit_score(opp.net_profit_pct)

        # Filter 2: Liquidity (20%)
        f2 = self._liquidity_score(opp.liquidity_ratio)

        # Filter 3: Signal consensus (20%)
        f3 = combiner_result.confidence

        # Filter 4: Execution speed (15%)
        f4 = self._latency_score(opp.exchange_latency_ms)

        # Filter 5: Historical consistency (10%)
        f5 = 0.5  # default neutral
        if pair_ranker:
            times_seen = pair_ranker.times_seen_24h(opp.pair)
            f5 = self._history_score(times_seen)

        # Filter 6: CVD alignment (3%)
        cvd_obj = signal_objects.get("cvd")
        f6 = self._cvd_alignment(opp.direction, cvd_obj, signals)

        # Filter 7: Regime alignment (2%)
        f7 = self._regime_alignment(opp.strategy, regime)

        final = (f1 * 0.30 + f2 * 0.20 + f3 * 0.20 + f4 * 0.15 +
                 f5 * 0.10 + f6 * 0.03 + f7 * 0.02)

        opp.score = round(final, 4)
        opp.confidence = combiner_result.confidence

        if opp.score >= self._obs_threshold:
            self._log(opp, executed=opp.score >= self._exec_threshold)

        return opp.score

    def _profit_score(self, net_pct: float) -> float:
        if net_pct >= 0.005:
            return 1.0
        elif net_pct >= 0.003:
            return 0.8
        elif net_pct >= 0.001:
            return 0.5
        elif net_pct >= 0.0003:
            return 0.2
        return 0.0

    def _liquidity_score(self, ratio: float) -> float:
        if ratio >= 50:
            return 1.0
        elif ratio >= 20:
            return 0.8
        elif ratio >= 10:
            return 0.6
        elif ratio >= 5:
            return 0.3
        return 0.0

    def _latency_score(self, latency_ms: float) -> float:
        if latency_ms < 50:
            return 1.0
        elif latency_ms < 100:
            return 0.7
        elif latency_ms < 200:
            return 0.3
        return 0.0

    def _history_score(self, times_seen: int) -> float:
        if times_seen >= 10:
            return 1.0
        elif times_seen >= 5:
            return 0.8
        elif times_seen >= 1:
            return 0.5
        return 0.3

    def _cvd_alignment(self, direction: str, cvd_obj, signals: dict) -> float:
        cvd_score = signals.get("cvd", 0)
        if direction == "long":
            if cvd_score > 0.1:
                return 1.0
            elif abs(cvd_score) <= 0.1:
                return 0.5
            else:
                return 0.0
        elif direction == "short":
            if cvd_score < -0.1:
                return 1.0
            elif abs(cvd_score) <= 0.1:
                return 0.5
            else:
                return 0.0
        return 0.5  # neutral

    def _regime_alignment(self, strategy: str, regime: str) -> float:
        primary = {
            "RANGING": ["triangular_arb", "stat_arb", "grid_trader", "mean_reversion"],
            "TRENDING_UP": ["correlation_breakout", "volume_spike", "stat_arb"],
            "TRENDING_DOWN": ["mean_reversion", "stat_arb"],
            "BREAKOUT": ["correlation_breakout", "volume_spike"],
            "FUNDING_RICH": ["funding_arb", "triangular_arb", "stat_arb", "grid_trader"],
            "WHALE_MOVING": ["correlation_breakout"],
            "CHOPPY": ["triangular_arb", "grid_trader"],
        }
        primaries = primary.get(regime, [])
        if strategy in primaries:
            return 1.0
        return 0.2

    def _log(self, opp: Opportunity, executed: bool):
        with open(OPP_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                opp.strategy,
                opp.pair,
                opp.regime,
                round(opp.net_profit_pct, 6),
                round(opp.liquidity_ratio, 2),
                round(opp.confidence, 4),
                round(opp.exchange_latency_ms, 1),
                opp.score,
                int(executed),
            ])
