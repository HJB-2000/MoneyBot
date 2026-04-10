import csv
import os
from collections import defaultdict
from datetime import datetime, timezone

PAIR_CSV = "data/pair_rankings.csv"


class PairPerformance:
    def __init__(self, config: dict):
        self._config = config

    def update(self, trades: list):
        by_pair = defaultdict(list)
        for t in trades:
            by_pair[t.get("pair", "")].append(t)

        rows = []
        for pair, ptrades in by_pair.items():
            wins = sum(1 for t in ptrades if t.get("result") == "WIN")
            losses = sum(1 for t in ptrades if t.get("result") == "LOSS")
            total = len(ptrades)
            pnl = sum(float(t.get("net_profit_pct", 0) or 0) for t in ptrades)
            wr = wins / total if total > 0 else 0.0
            rows.append({
                "pair": pair,
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate": round(wr, 4),
                "total_pnl_pct": round(pnl, 6),
                "avg_pnl_pct": round(pnl / total, 6) if total > 0 else 0.0,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })

        if not rows:
            return

        with open(PAIR_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
