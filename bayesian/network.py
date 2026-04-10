"""
Bayesian Signal Network — core engine.
Takes 30 signal scores + strategy + regime → P(trade_wins) in [0, 1].
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

from bayesian.prior_loader import PriorLoader
from bayesian.correlation_filter import CorrelationFilter

DECISION_LOG = "data/decision_log.csv"

STRATEGY_PRIMARY_SIGNALS = {
    "triangular_arb":       ["bid_ask_spread", "liquidity_score",
                             "ATR_ratio", "order_flow_imbalance"],
    "stat_arb":             ["RSI", "MACD", "CVD_divergence",
                             "depth_imbalance"],
    "funding_arb":          ["funding_rate", "open_interest_change"],
    "grid_trader":          ["consolidation_score", "ATR_ratio",
                             "bollinger_state", "depth_imbalance"],
    "mean_reversion":       ["RSI", "CVD_divergence", "volume_spike_signal",
                             "support_resistance"],
    "volume_spike":         ["volume_spike_signal", "CVD_trend",
                             "buy_sell_ratio"],
    "correlation_breakout": ["btc_correlation_momentum",
                             "EMA_cross", "rate_of_change"],
}


class BayesianNetwork:
    def __init__(self):
        loader = PriorLoader()
        self.likelihood_ratios = loader.load_likelihood_ratios()
        self.base_rates        = loader.load_base_rates()
        self._corr_filter      = CorrelationFilter()
        self._ensure_log()

    def reload(self):
        loader = PriorLoader()
        self.likelihood_ratios = loader.load_likelihood_ratios()
        self.base_rates        = loader.load_base_rates()
        self._corr_filter      = CorrelationFilter()

    # ── Main compute ──────────────────────────────────────────────────────
    def compute(self, signals: dict, strategy: str, regime: str,
                direction: str = "neutral") -> float:
        # Step 6 — Veto layer (checked first for speed)
        if self._veto(signals, regime, strategy, direction):
            return 0.0

        # Step 1 — Prior
        prior = self.base_rates.get(regime, 0.50)
        if prior <= 0: prior = 0.01
        if prior >= 1: prior = 0.99

        # Step 2 — Decorrelate
        decorr = self._corr_filter.filter(signals)

        # Step 3 — Primary signals for this strategy
        primaries = STRATEGY_PRIMARY_SIGNALS.get(strategy, [])

        # Step 4 — Bayesian update
        odds = prior / (1 - prior)
        strat_ratios = self._get_ratios(strategy)

        for sig_name in primaries:
            if sig_name not in decorr:
                continue
            lr_base = strat_ratios.get(sig_name, 1.0)
            score   = decorr[sig_name]
            if lr_base <= 0:
                lr_base = 1.0
            if score > 0:
                signal_lr = 1 + (lr_base - 1) * score
            else:
                inv_lr = 1 / max(lr_base, 0.01)
                signal_lr = 1 + (inv_lr - 1) * abs(score)
            signal_lr = max(0.01, signal_lr)
            odds *= signal_lr

        P = odds / (1 + odds)

        # Step 5 — Boost from supporting non-primary signals, penalty from opposing
        non_primary = [k for k in decorr if k not in primaries]
        boost, penalty = 0.0, 0.0
        for sig_name in non_primary:
            score = decorr.get(sig_name, 0)
            if score > 0.1:
                boost += 0.02
            elif score < -0.3:
                penalty += 0.02
        P = min(P + min(boost, 0.10) - min(penalty, 0.10), 0.95)

        return round(float(P), 4)

    # ── Explain ───────────────────────────────────────────────────────────
    def explain(self, signals: dict, strategy: str, regime: str,
                direction: str = "neutral") -> dict:
        vetoed = self._veto(signals, regime, strategy, direction)
        P = self.compute(signals, strategy, regime, direction)
        prior = self.base_rates.get(regime, 0.50)
        primaries = STRATEGY_PRIMARY_SIGNALS.get(strategy, [])
        strat_ratios = self._get_ratios(strategy)
        key_signals = []
        for sig in primaries:
            score = signals.get(sig, 0.0)
            lr    = strat_ratios.get(sig, 1.0)
            if abs(score) > 0.3:
                direction_str = "+" if score > 0 else "-"
                key_signals.append(f"{sig} ({direction_str}{abs(score):.1f}, LR={lr:.2f})")

        explanation = {
            "prior":             round(prior, 4),
            "final_probability": P,
            "strategy":          strategy,
            "regime":            regime,
            "key_signals":       key_signals[:5],
            "vetoes_checked":    "triggered" if vetoed else "none triggered",
            "decision":          "BLOCKED" if vetoed else ("EXECUTE" if P >= 0.65 else "OBSERVE"),
        }
        self._log_decision(explanation, signals)
        return explanation

    # ── Veto ──────────────────────────────────────────────────────────────
    def _veto(self, signals: dict, regime: str, strategy: str,
              direction: str) -> bool:
        if regime == "VOLATILE":
            return True
        if float(signals.get("ATR_ratio", 0)) > 2.0:
            return True
        if float(signals.get("realized_vol", 0)) <= -0.9:
            return True
        if signals.get("liquidation_cascade") is True:
            return True
        if signals.get("liquidation_pressure", 0) <= -0.79:
            return True
        if signals.get("whale_selling") is True and direction == "long":
            return True
        if signals.get("whale_buying") is True and direction == "short":
            return True
        return False

    # ── Helpers ───────────────────────────────────────────────────────────
    def _get_ratios(self, strategy: str) -> dict:
        ratios = self.likelihood_ratios
        if isinstance(ratios, dict):
            if strategy in ratios:
                return ratios[strategy]
            # Literature priors format: {signal: {strategy: lr}}
            result = {}
            for sig, val in ratios.items():
                if isinstance(val, dict):
                    result[sig] = val.get(strategy, val.get("all", val.get("default", 1.0)))
                else:
                    result[sig] = float(val)
            return result
        return {}

    def _ensure_log(self):
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(DECISION_LOG) or os.path.getsize(DECISION_LOG) == 0:
            with open(DECISION_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "strategy", "regime", "prior",
                    "final_probability", "decision", "key_signals", "vetoes",
                ])

    def _log_decision(self, explanation: dict, signals: dict):
        try:
            with open(DECISION_LOG, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    explanation.get("strategy", ""),
                    explanation.get("regime", ""),
                    explanation.get("prior", ""),
                    explanation.get("final_probability", ""),
                    explanation.get("decision", ""),
                    "; ".join(explanation.get("key_signals", [])),
                    explanation.get("vetoes_checked", ""),
                ])
        except Exception:
            pass
