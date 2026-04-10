import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from learning.signal_optimizer import SignalOptimizer
from learning.strategy_optimizer import StrategyOptimizer
from learning.pair_performance import PairPerformance

TRADE_LOG = "data/trade_log.csv"
REPORTS_DIR = "reports/daily"
LEARNING_LOG = "data/learning_log.csv"


class TradeAnalyzer:
    def __init__(self, config: dict):
        self._config = config
        self._sig_opt = SignalOptimizer(config)
        self._strat_opt = StrategyOptimizer(config)
        self._pair_perf = PairPerformance(config)
        Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        self._ensure_learning_log()

    def _ensure_learning_log(self):
        if not os.path.exists(LEARNING_LOG) or os.path.getsize(LEARNING_LOG) == 0:
            with open(LEARNING_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["date", "analysis", "result"])

    def daily_analysis(self):
        trades = self._read_last_24h()
        if not trades:
            self._log("daily_analysis", "no_trades")
            return

        # Analysis 1 — regime performance
        self._analyze_regime(trades)

        # Analysis 2 — signal accuracy
        self._analyze_signals(trades)

        # Analysis 3 — pair performance
        self._pair_perf.update(trades)

        # Analysis 4 — strategy performance
        self._analyze_strategies(trades)

        # Analysis 5 — timing
        self._analyze_timing(trades)

        # Analysis 6 — threshold calibration
        self._calibrate_threshold(trades)

        # Generate daily report
        self._write_report(trades)
        self._log("daily_analysis", f"processed {len(trades)} trades")

    def _read_last_24h(self) -> list:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        trades = []
        if not os.path.exists(TRADE_LOG):
            return []
        with open(TRADE_LOG) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("timestamp", "") >= cutoff:
                    trades.append(row)
        return trades

    def _analyze_regime(self, trades: list):
        by_regime = defaultdict(list)
        for t in trades:
            by_regime[t.get("regime", "UNKNOWN")].append(t)

        params = self._load_strategy_params()
        for regime, regime_trades in by_regime.items():
            wins = sum(1 for t in regime_trades if t.get("result") == "WIN")
            if len(regime_trades) == 0:
                continue
            wr = wins / len(regime_trades)
            mult_key = f"regime_mult_{regime}"
            current = params.get(mult_key, 1.0)
            if wr > 0.65:
                params[mult_key] = min(1.5, current * 1.1)
            elif wr < 0.45:
                params[mult_key] = max(0.2, current * 0.9)
        self._save_strategy_params(params)

    def _analyze_signals(self, trades: list):
        """Compute signal accuracy per signal and update weights."""
        # Since we don't store per-signal scores per trade in this version,
        # use the dominant signal as a proxy.
        accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        for t in trades:
            regime = t.get("regime", "")
            result = t.get("result", "")
            net = float(t.get("net_profit_pct", 0) or 0)
            # Heuristic: credit the signal that matches regime direction
            if regime in ("TRENDING_UP", "RANGING") and result == "WIN":
                accuracy["momentum"]["correct"] += 1
            accuracy["momentum"]["total"] += 1

        signal_accuracy = {
            s: (v["correct"] / v["total"]) if v["total"] > 0 else 0.5
            for s, v in accuracy.items()
        }
        self._sig_opt.update_weights(signal_accuracy)

    def _analyze_strategies(self, trades: list):
        by_strat = defaultdict(list)
        for t in trades:
            by_strat[t.get("strategy", "")].append(t)

        params = self._load_strategy_params()
        for strat, strat_trades in by_strat.items():
            if len(strat_trades) < 5:
                continue
            wins = sum(1 for t in strat_trades if t.get("result") == "WIN")
            wr = wins / len(strat_trades)
            key = f"strategy_enabled_{strat}"
            if wr < 0.45:
                params[key] = False
                self._log(f"disable_{strat}", f"wr={wr:.2f}")
            else:
                params[key] = True
        self._save_strategy_params(params)

    def _analyze_timing(self, trades: list):
        by_hour = defaultdict(list)
        for t in trades:
            ts = t.get("timestamp", "")
            if ts:
                try:
                    hour = datetime.fromisoformat(ts).hour
                    by_hour[hour].append(t)
                except Exception:
                    pass

        hour_pnl = {}
        for hour, htrades in by_hour.items():
            pnl = sum(float(t.get("net_profit_pct", 0) or 0) for t in htrades)
            hour_pnl[hour] = pnl

        best_hours = sorted(hour_pnl, key=hour_pnl.get, reverse=True)[:8]
        params = self._load_strategy_params()
        params["peak_hours"] = best_hours
        self._save_strategy_params(params)

    def _calibrate_threshold(self, trades: list):
        """Adjust execution threshold based on 7-day win rate."""
        if len(trades) < 10:
            return
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        wr = wins / len(trades)
        params = self._load_strategy_params()
        threshold = params.get("execution_threshold", 0.72)
        if wr > 0.70:
            threshold = max(0.62, threshold - 0.01)
        elif wr < 0.55:
            threshold = min(0.82, threshold + 0.01)
        params["execution_threshold"] = threshold
        self._save_strategy_params(params)

    def _write_report(self, trades: list):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = f"{REPORTS_DIR}/{date_str}.txt"
        wins = sum(1 for t in trades if t.get("result") == "WIN")
        losses = sum(1 for t in trades if t.get("result") == "LOSS")
        total_pnl = sum(float(t.get("net_profit_pct", 0) or 0) for t in trades)
        with open(path, "w") as f:
            f.write(f"MoneyBot Daily Report — {date_str}\n{'='*40}\n")
            f.write(f"Total trades: {len(trades)}\n")
            f.write(f"Wins: {wins} | Losses: {losses}\n")
            f.write(f"Win rate: {wins/len(trades)*100:.1f}%\n" if trades else "")
            f.write(f"Total net P&L: {total_pnl*100:.4f}%\n")
        print(f"[TradeAnalyzer] Daily report: {path}")

    def _load_strategy_params(self) -> dict:
        path = "data/strategy_params.json"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_strategy_params(self, params: dict):
        with open("data/strategy_params.json", "w") as f:
            json.dump(params, f, indent=2)

    def _log(self, analysis: str, result: str):
        with open(LEARNING_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(), analysis, result
            ])
