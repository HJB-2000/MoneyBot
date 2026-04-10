"""
Likelihood engine — calculates and updates likelihood ratios from trade history.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone

from bayesian.prior_loader import PriorLoader, LITERATURE_PRIORS

LIKELIHOOD_FILE = "data/likelihood_ratios.json"
MIN_TRADES = 30
SLIDING_WINDOW_DAYS = 60

ALL_SIGNALS = [
    "RSI", "MACD", "EMA_cross", "ATR_ratio", "bollinger_state",
    "rate_of_change", "support_resistance", "trend_structure",
    "consolidation_score", "realized_vol", "CVD_divergence", "CVD_trend",
    "volume_spike_signal", "buy_sell_ratio", "large_trade_flow",
    "bid_ask_spread", "depth_imbalance", "order_flow_imbalance",
    "large_order_presence", "iceberg_detection", "spoofing_detection",
    "liquidity_score", "book_pressure_ratio", "funding_rate",
    "open_interest_change", "liquidation_pressure", "vwap_position",
    "vwap_reclaim", "btc_correlation_momentum", "regime_persistence",
]

STRATEGIES = [
    "triangular_arb", "stat_arb", "funding_arb", "grid_trader",
    "mean_reversion", "volume_spike", "correlation_breakout",
]


class LikelihoodEngine:
    def __init__(self):
        self._loader = PriorLoader()

    def load_literature_priors(self) -> dict:
        return self._loader.load_literature_priors()

    def calculate_from_history(self, trade_log, signal_log) -> dict:
        """
        trade_log: path to trade_log.csv  OR list of dicts
        signal_log: path to signal_log.csv OR list of dicts
        Returns likelihood_ratios dict or falls back to literature priors.
        """
        trades  = self._read(trade_log)
        signals = self._read(signal_log)

        if len(trades) < MIN_TRADES or len(signals) < MIN_TRADES:
            return self.load_literature_priors()

        # Build timestamp-indexed signal lookup
        sig_by_ts = {r.get("timestamp", ""): r for r in signals}
        ratios = {}
        for strategy in STRATEGIES:
            strat_trades = [t for t in trades if t.get("strategy") == strategy]
            if len(strat_trades) < MIN_TRADES:
                ratios[strategy] = self._strategy_priors(strategy)
                continue
            wins   = [t for t in strat_trades if t.get("result") == "WIN"]
            losses = [t for t in strat_trades if t.get("result") == "LOSS"]
            if not wins or not losses:
                ratios[strategy] = self._strategy_priors(strategy)
                continue
            strat_ratios = {}
            for sig_name in ALL_SIGNALS:
                pw = self._p_positive(sig_name, wins,   sig_by_ts)
                pl = self._p_positive(sig_name, losses, sig_by_ts)
                lr = pw / max(pl, 0.01)
                strat_ratios[sig_name] = round(lr, 4)
            ratios[strategy] = strat_ratios

        os.makedirs("data", exist_ok=True)
        with open(LIKELIHOOD_FILE, "w") as f:
            json.dump(ratios, f, indent=2)
        return ratios

    def update_weekly(self, new_trades) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=SLIDING_WINDOW_DAYS)).isoformat()
        if isinstance(new_trades, str):
            rows = self._read(new_trades)
        else:
            rows = new_trades
        recent = [r for r in rows if r.get("timestamp", "") >= cutoff]
        return self.calculate_from_history(recent, [])

    def train_sliding_window(self, trade_log: str, signal_log: str,
                             lookback_days: int = 60) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        trades  = [r for r in self._read(trade_log)  if r.get("timestamp", "") >= cutoff]
        signals = [r for r in self._read(signal_log) if r.get("timestamp", "") >= cutoff]
        if len(trades) < MIN_TRADES:
            return {}
        return self.calculate_from_history(trades, signals)

    def _p_positive(self, signal_name: str, trade_subset: list,
                    sig_by_ts: dict) -> float:
        count, positive = 0, 0
        for t in trade_subset:
            ts = t.get("timestamp", "")
            row = sig_by_ts.get(ts)
            if row is None:
                continue
            val = row.get(signal_name)
            if val is None:
                continue
            try:
                if float(val) > 0:
                    positive += 1
                count += 1
            except (TypeError, ValueError):
                pass
        return positive / count if count > 0 else 0.5

    def _strategy_priors(self, strategy: str) -> dict:
        priors = self.load_literature_priors()
        result = {}
        for sig in ALL_SIGNALS:
            sp = priors.get(sig, {})
            if isinstance(sp, dict):
                lr = sp.get(strategy, sp.get("all", sp.get("default", 1.0)))
            else:
                lr = float(sp)
            result[sig] = float(lr)
        return result

    def _read(self, source) -> list:
        if isinstance(source, list):
            return source
        if not source or not os.path.exists(str(source)):
            return []
        rows = []
        try:
            with open(source) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
        except Exception:
            pass
        return rows
