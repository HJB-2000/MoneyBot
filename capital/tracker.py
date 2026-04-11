import sqlite3
import csv
import os
import threading
from datetime import datetime, date, timezone
from pathlib import Path


MILESTONES = [100, 250, 500, 1000, 2500, 5000]
DB_PATH = "moneyBot.db"
TRADE_LOG = "data/trade_log.csv"
REPORTS_DIR = "reports/daily"


class CapitalTracker:
    def __init__(self, config: dict):
        self._lock = threading.Lock()
        self._starting_capital = config["capital"]["starting_capital"]
        self._survival_floor = config["capital"]["survival_floor"]
        self._capital = self._load_or_init()
        self._peak_capital = self._capital
        self._daily_start = self._capital
        self._daily_start_date = date.today()
        self._milestones_crossed = set(self._load_milestones())
        self._ensure_trade_log()

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _get_conn(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS capital (
                id INTEGER PRIMARY KEY,
                balance REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                value REAL PRIMARY KEY
            )
        """)
        conn.commit()
        return conn

    def _load_or_init(self) -> float:
        conn = self._get_conn()
        row = conn.execute("SELECT balance FROM capital ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if row:
            print(f"[CapitalTracker] Loaded balance: ${row[0]:.2f}")
            return float(row[0])
        self._save_balance(self._starting_capital)
        print(f"[CapitalTracker] First run — initialised at ${self._starting_capital:.2f}")
        return self._starting_capital

    def _save_balance(self, balance: float):
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO capital (balance, updated_at) VALUES (?, ?)",
            (balance, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def _load_milestones(self) -> list:
        conn = self._get_conn()
        rows = conn.execute("SELECT value FROM milestones").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _save_milestone(self, value: float):
        conn = self._get_conn()
        conn.execute("INSERT OR IGNORE INTO milestones (value) VALUES (?)", (value,))
        conn.commit()
        conn.close()

    def _ensure_trade_log(self):
        if not os.path.exists(TRADE_LOG) or os.path.getsize(TRADE_LOG) == 0:
            with open(TRADE_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "strategy", "pair", "direction",
                    "entry_price", "exit_price", "size_usd",
                    "gross_profit_pct", "fees_pct", "slippage_pct", "net_profit_pct",
                    "net_pnl_usd", "balance_after",
                    "regime", "confidence", "score",
                    "hold_seconds", "result"
                ])

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def get_capital(self) -> float:
        with self._lock:
            return self._capital

    def log_open(self, opp) -> None:
        """
        Called when a trade is opened.
        Deducts trade size from capital and logs a RUNNING entry to trade_log.
        """
        with self._lock:
            self._roll_daily_if_needed()
            self._capital -= opp.trade_size_usd
            self._capital = max(self._capital, self._survival_floor)
            self._save_balance(self._capital)
            meta = {
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "strategy":    opp.strategy,
                "pair":        opp.pair,
                "direction":   opp.direction,
                "entry_price": opp.entry_price,
                "exit_price":  "",
                "size_usd":    opp.trade_size_usd,
                "gross_profit_pct": "",
                "fees_pct":    opp.fees_pct,
                "slippage_pct": opp.slippage_pct,
                "net_profit_pct": "",
                "regime":      opp.regime,
                "confidence":  opp.confidence,
                "score":       opp.score,
                "hold_seconds": "",
                "result":      "RUNNING",
                "trade_id":    opp.id,
            }
            self._log_trade(-opp.trade_size_usd, meta)
            print(f"  OPEN  {opp.strategy} {opp.pair} ${opp.trade_size_usd:.2f} → capital ${self._capital:.2f}")

    def update(self, pnl_usd: float, trade_meta: dict = None):
        """
        Called when a trade closes.
        pnl_usd = net P&L only (NOT including returned size).
        Adds back trade size + pnl and logs WIN/LOSS.
        """
        with self._lock:
            self._roll_daily_if_needed()
            size = float(trade_meta.get("size_usd", 0)) if trade_meta else 0.0
            self._capital += size + pnl_usd   # return the locked capital + pnl
            self._capital = max(self._capital, self._survival_floor)
            if self._capital > self._peak_capital:
                self._peak_capital = self._capital
            self._save_balance(self._capital)
            self._log_trade(pnl_usd, trade_meta)
            self._milestone_check()
            print(f"  CLOSE capital: ${self._capital:.2f} ({'+'if pnl_usd>=0 else ''}{pnl_usd:.4f})")

    def _roll_daily_if_needed(self):
        today = date.today()
        if today != self._daily_start_date:
            self._save_daily_report(self._daily_start_date)
            self._daily_start = self._capital
            self._daily_start_date = today

    def get_daily_pnl(self) -> float:
        with self._lock:
            return self._capital - self._daily_start

    def get_drawdown(self) -> float:
        """Returns current drawdown as a fraction (0.0 – 1.0) from peak."""
        with self._lock:
            if self._peak_capital == 0:
                return 0.0
            return max(0.0, (self._peak_capital - self._capital) / self._peak_capital)

    def get_win_rate(self, last_n: int = 20) -> float:
        results = self._read_last_results(last_n)
        if not results:
            return 0.0
        wins = sum(1 for r in results if r == "WIN")
        return wins / len(results)

    def get_consecutive_losses(self) -> int:
        results = self._read_last_results(50)
        count = 0
        for r in reversed(results):
            if r == "LOSS":
                count += 1
            else:
                break
        return count

    def _read_last_results(self, n: int) -> list:
        if not os.path.exists(TRADE_LOG):
            return []
        rows = []
        with open(TRADE_LOG, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = row.get("result", "")
                if r in ("WIN", "LOSS"):   # skip RUNNING entries
                    rows.append(r)
        return rows[-n:]

    def _milestone_check(self):
        for m in MILESTONES:
            if self._capital >= m and m not in self._milestones_crossed:
                self._milestones_crossed.add(m)
                self._save_milestone(m)
                print(f"\n{'='*50}")
                print(f"  MILESTONE REACHED: ${m:,}!")
                print(f"  Current balance: ${self._capital:.2f}")
                print(f"{'='*50}\n")

    def _log_trade(self, pnl_usd: float, meta: dict):
        if meta is None:
            return
        with open(TRADE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                meta.get("timestamp", datetime.now(timezone.utc).isoformat()),
                meta.get("strategy", ""),
                meta.get("pair", ""),
                meta.get("direction", ""),
                meta.get("entry_price", ""),
                meta.get("exit_price", ""),
                meta.get("size_usd", ""),
                meta.get("gross_profit_pct", ""),
                meta.get("fees_pct", ""),
                meta.get("slippage_pct", ""),
                meta.get("net_profit_pct", ""),
                round(pnl_usd, 6),
                round(self._capital, 4),
                meta.get("regime", ""),
                meta.get("confidence", ""),
                meta.get("score", ""),
                meta.get("hold_seconds", ""),
                meta.get("result", ""),
            ])

    def _save_daily_report(self, report_date: date):
        Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        path = f"{REPORTS_DIR}/{report_date}.txt"
        daily_pnl = self._capital - self._daily_start
        with open(path, "w") as f:
            f.write(f"MoneyBot Daily Report — {report_date}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Opening balance:  ${self._daily_start:.2f}\n")
            f.write(f"Closing balance:  ${self._capital:.2f}\n")
            f.write(f"Daily P&L:        ${daily_pnl:+.2f}\n")
            f.write(f"Daily P&L %:      {daily_pnl/self._daily_start*100:+.2f}%\n")
            f.write(f"Peak capital:     ${self._peak_capital:.2f}\n")
            f.write(f"Drawdown:         {self.get_drawdown()*100:.2f}%\n")
            f.write(f"Win rate (last20):{self.get_win_rate()*100:.1f}%\n")
        print(f"[CapitalTracker] Daily report saved: {path}")
