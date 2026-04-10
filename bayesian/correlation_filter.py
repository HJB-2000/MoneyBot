"""
Correlation filter — prevents double-counting correlated signals.
"""
from __future__ import annotations

import json
import os

CORRELATIONS_FILE = "data/signal_correlations.json"

CORRELATION_GROUPS = {
    "momentum":    ["RSI", "MACD", "EMA_cross", "rate_of_change"],
    "volatility":  ["ATR_ratio", "bollinger_state", "realized_vol"],
    "order_flow":  ["order_flow_imbalance", "buy_sell_ratio", "CVD_trend",
                    "depth_imbalance"],
    "institutional": ["large_trade_flow", "iceberg_detection", "book_pressure_ratio"],
}


class CorrelationFilter:
    def __init__(self):
        self._matrix = self._load_matrix()

    def _load_matrix(self) -> dict:
        if os.path.exists(CORRELATIONS_FILE):
            try:
                with open(CORRELATIONS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def filter(self, signals: dict) -> dict:
        """
        Reduce contribution of correlated signals within each group.
        Signals in the same group share their weight proportionally.
        """
        adjusted = dict(signals)
        for group_name, members in CORRELATION_GROUPS.items():
            present = [m for m in members if m in signals]
            if len(present) < 2:
                continue
            # Group score = average of member scores
            group_score = sum(signals[m] for m in present) / len(present)
            # Correlation within group (default 0.7 if not trained)
            avg_corr = self._avg_group_corr(present) or 0.7
            n = len(present)
            # Effective independent count: 1 + (n-1)*(1-corr)
            # Scale factor < 1 prevents correlated signals from double-counting
            effective_count = 1 + (n - 1) * (1 - avg_corr)
            scale = effective_count / n
            for m in present:
                # Blend toward group average (captures contradictions) then scale down
                blended = signals[m] * (1 - avg_corr) + group_score * avg_corr
                adjusted[m] = blended * scale
        return adjusted

    def _avg_group_corr(self, members: list) -> float:
        if not self._matrix:
            return 0.7
        pairs, total = 0, 0.0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if a in self._matrix and b in self._matrix[a]:
                    total += abs(self._matrix[a][b])
                    pairs += 1
        return total / pairs if pairs > 0 else 0.7

    def build_matrix(self, historical_signals: dict) -> dict:
        """
        Build Pearson correlation matrix from historical signal data.
        historical_signals: {signal_name: [list of values]}
        """
        import math
        names = list(historical_signals.keys())
        matrix = {n: {} for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                xa = historical_signals[a]
                xb = historical_signals[b]
                n = min(len(xa), len(xb))
                if n < 2:
                    continue
                xa, xb = xa[:n], xb[:n]
                mean_a = sum(xa) / n
                mean_b = sum(xb) / n
                num = sum((xa[k] - mean_a) * (xb[k] - mean_b) for k in range(n))
                da = math.sqrt(sum((xa[k] - mean_a) ** 2 for k in range(n)))
                db = math.sqrt(sum((xb[k] - mean_b) ** 2 for k in range(n)))
                if da == 0 or db == 0:
                    continue
                corr = num / (da * db)
                matrix[a][b] = round(corr, 4)
                matrix[b][a] = round(corr, 4)

        os.makedirs("data", exist_ok=True)
        with open(CORRELATIONS_FILE, "w") as f:
            json.dump(matrix, f, indent=2)
        return matrix
