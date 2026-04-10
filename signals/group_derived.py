"""
Derived signal group — 2 signals calculated from cached results.
No extra data fetches needed.
"""
from __future__ import annotations


class DerivedSignals:
    def __init__(self):
        self.scores: dict = {}

    def calculate(self, candle_signals: dict, trade_signals: dict,
                  regime_cycles: int = 0, btc_roc: float = 0.0,
                  pair_btc_corr: float = 0.0) -> dict:
        try:
            btc = self._btc_correlation_momentum(
                candle_signals, btc_roc, pair_btc_corr
            )
            reg = self._regime_persistence(regime_cycles)
            self.scores = {
                "btc_correlation_momentum": btc,
                "regime_persistence": reg,
            }
            return self.scores
        except Exception:
            return {"btc_correlation_momentum": 0.0, "regime_persistence": 0.0}

    # ── Signal 29 — BTC Correlation Momentum ────────────────────────────
    def _btc_correlation_momentum(self, candle_signals: dict,
                                  btc_roc: float, pair_btc_corr: float) -> float:
        """
        If the pair has high BTC correlation (> 0.80) and BTC moved > 2%
        but this pair has not yet followed, signal the likely catch-up.
        """
        if pair_btc_corr < 0.80:
            return 0.0
        if abs(btc_roc) < 0.02:
            return 0.0
        # check if this pair's ROC is already close to BTC's move
        pair_roc = candle_signals.get("rate_of_change", 0.0)
        if abs(pair_roc) >= abs(btc_roc) * 0.7:
            return 0.0   # already followed
        direction = 1.0 if btc_roc > 0 else -1.0
        return float(max(-1.0, min(1.0, direction * 0.6)))

    # ── Signal 30 — Regime Persistence ──────────────────────────────────
    def _regime_persistence(self, cycles: int) -> float:
        if cycles <= 0:
            return 0.0
        elif cycles <= 3:
            return 0.0
        elif cycles <= 10:
            return 0.2
        elif cycles <= 30:
            return 0.4
        return 0.1   # may be about to change
