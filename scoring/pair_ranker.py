import csv
import os
import sqlite3
from datetime import datetime, timedelta, timezone

PAIR_DB = "moneyBot.db"
PAIR_CSV = "data/pair_rankings.csv"


class PairRanker:
    def __init__(self, config: dict, pairs: list):
        self._min_trades = config["learning"]["min_trades_for_ranking"]
        self._conn = sqlite3.connect(PAIR_DB, check_same_thread=False)
        self._init_db(pairs)
        self._tier_cache: dict = {}
        self._last_tier_update = None

    def _init_db(self, pairs: list):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS pair_stats (
                pair TEXT PRIMARY KEY,
                total_scans INTEGER DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl_pct REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)
        self._conn.commit()
        for pair in pairs:
            self._conn.execute(
                "INSERT OR IGNORE INTO pair_stats (pair, last_updated) VALUES (?, ?)",
                (pair, datetime.now(timezone.utc).isoformat())
            )
        self._conn.commit()

    def get_tier(self, pair: str) -> str:
        self._maybe_update_tiers()
        return self._tier_cache.get(pair, "B")

    def times_seen_24h(self, pair: str) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        # Use opportunity log as source
        count = 0
        if os.path.exists("data/opportunity_log.csv"):
            with open("data/opportunity_log.csv") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("pair") == pair and row.get("timestamp", "") >= cutoff:
                        count += 1
        return count

    def win_rate(self, pair: str) -> float:
        row = self._conn.execute(
            "SELECT wins, total_trades FROM pair_stats WHERE pair=?", (pair,)
        ).fetchone()
        if not row or row[1] == 0:
            return 0.0
        return row[0] / row[1]

    def update(self, pair: str, result: str, pnl_pct: float = 0.0):
        """Update pair stats after a trade result."""
        win = 1 if result == "WIN" else 0
        loss = 1 if result == "LOSS" else 0
        self._conn.execute("""
            UPDATE pair_stats
            SET total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl_pct = total_pnl_pct + ?,
                last_updated = ?
            WHERE pair = ?
        """, (win, loss, pnl_pct, datetime.now(timezone.utc).isoformat(), pair))
        self._conn.commit()

    def increment_scan(self, pair: str):
        self._conn.execute(
            "UPDATE pair_stats SET total_scans = total_scans + 1 WHERE pair=?", (pair,)
        )
        self._conn.commit()

    def _maybe_update_tiers(self):
        if self._last_tier_update and \
                (datetime.now(timezone.utc) - self._last_tier_update) < timedelta(hours=1):
            return
        rows = self._conn.execute(
            "SELECT pair, wins, losses, total_trades FROM pair_stats"
        ).fetchall()

        tiers = {}
        for pair, wins, losses, trades in rows:
            if trades >= self._min_trades:
                wr = wins / trades
                if wr > 0.60:
                    tiers[pair] = "A"
                elif wr > 0.45:
                    tiers[pair] = "B"
                else:
                    tiers[pair] = "C"
            else:
                tiers[pair] = "B"  # new pairs start at Tier B

        self._tier_cache = tiers
        self._last_tier_update = datetime.now(timezone.utc)
        self._save_csv(rows)

    def _save_csv(self, rows):
        with open(PAIR_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair", "tier", "wins", "losses", "total_trades",
                             "win_rate", "last_updated"])
            for pair, wins, losses, trades in rows:
                wr = wins / trades if trades > 0 else 0.0
                tier = self._tier_cache.get(pair, "B")
                writer.writerow([pair, tier, wins, losses, trades,
                                 round(wr, 4), datetime.now(timezone.utc).isoformat()])
