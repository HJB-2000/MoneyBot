from __future__ import annotations

import json
from pathlib import Path

import pytest

import learning.pair_performance as pair_performance_module
import learning.signal_optimizer as signal_optimizer_module
import learning.strategy_optimizer as strategy_optimizer_module
import learning.trade_analyzer as trade_analyzer_module
from learning.pair_performance import PairPerformance
from learning.signal_optimizer import SignalOptimizer
from learning.strategy_optimizer import StrategyOptimizer
from learning.trade_analyzer import TradeAnalyzer


def patch_learning_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(signal_optimizer_module, "WEIGHTS_FILE", str(tmp_path / "data" / "signal_weights.json"), raising=False)
    monkeypatch.setattr(strategy_optimizer_module, "PARAMS_FILE", str(tmp_path / "data" / "strategy_params.json"), raising=False)
    monkeypatch.setattr(pair_performance_module, "PAIR_CSV", str(tmp_path / "data" / "pair_rankings.csv"), raising=False)
    monkeypatch.setattr(trade_analyzer_module, "TRADE_LOG", str(tmp_path / "data" / "trade_log.csv"), raising=False)
    monkeypatch.setattr(trade_analyzer_module, "REPORTS_DIR", str(tmp_path / "reports" / "daily"), raising=False)
    monkeypatch.setattr(trade_analyzer_module, "LEARNING_LOG", str(tmp_path / "data" / "learning_log.csv"), raising=False)


def write_trade_log(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("timestamp,strategy,pair,direction,entry_price,exit_price,size_usd,gross_profit_pct,fees_pct,slippage_pct,net_profit_pct,net_pnl_usd,balance_after,regime,confidence,score,hold_seconds,result\n")
        for row in rows:
            f.write(
                ",".join(str(row.get(key, "")) for key in [
                    "timestamp", "strategy", "pair", "direction", "entry_price", "exit_price", "size_usd",
                    "gross_profit_pct", "fees_pct", "slippage_pct", "net_profit_pct", "net_pnl_usd",
                    "balance_after", "regime", "confidence", "score", "hold_seconds", "result",
                ])
                + "\n"
            )


def test_signal_optimizer_normalizes_and_bounds(config, workspace, monkeypatch):
    patch_learning_paths(monkeypatch, workspace)
    path = workspace / "data" / "signal_weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "price_action": 0.15,
        "momentum": 0.15,
        "microstructure": 0.20,
        "volatility": 0.15,
        "sentiment": 0.10,
        "cvd": 0.15,
        "vwap": 0.05,
        "whale": 0.05,
    }))

    SignalOptimizer(config).update_weights({"momentum": 0.9, "price_action": 0.4})
    weights = json.loads(path.read_text())
    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert all(0.03 <= value <= 0.40 for value in weights.values())


def test_strategy_optimizer_adjusts_parameters(config, workspace, monkeypatch):
    patch_learning_paths(monkeypatch, workspace)
    params_path = workspace / "data" / "strategy_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps({
        "grid_spacing_atr_mult": 0.5,
        "z_score_entry": 2.0,
        "drop_threshold_pct": 0.02,
    }))

    StrategyOptimizer(config).update_params({"grid_trader": {"win_rate": 0.4}, "stat_arb": {"win_rate": 0.4}, "mean_reversion": {"win_rate": 0.4}})
    params = json.loads(params_path.read_text())
    assert params["grid_spacing_atr_mult"] > 0.5
    assert params["z_score_entry"] >= 2.1
    assert params["drop_threshold_pct"] < 0.02


def test_pair_performance_writes_csv(config, workspace, monkeypatch):
    patch_learning_paths(monkeypatch, workspace)
    trades = [
        {"pair": "SOL/USDT", "result": "WIN", "net_profit_pct": 0.01},
        {"pair": "SOL/USDT", "result": "LOSS", "net_profit_pct": -0.005},
    ]
    PairPerformance(config).update(trades)
    assert Path(workspace / "data" / "pair_rankings.csv").exists()


def test_trade_analyzer_daily_analysis_runs_and_updates(config, workspace, monkeypatch):
    from datetime import datetime, timezone, timedelta
    patch_learning_paths(monkeypatch, workspace)
    trade_log = workspace / "data" / "trade_log.csv"
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)
    rows = []
    for idx in range(12):
        ts = (base_time + timedelta(minutes=idx)).strftime("%Y-%m-%dT%H:%M:%S")
        rows.append(
            {
                "timestamp": ts,
                "strategy": "triangular_arb" if idx < 6 else "grid_trader",
                "pair": "SOL/USDT" if idx < 8 else "MATIC/USDT",
                "direction": "long",
                "entry_price": 100,
                "exit_price": 101,
                "size_usd": 5,
                "gross_profit_pct": 0.01,
                "fees_pct": 0.002,
                "slippage_pct": 0.001,
                "net_profit_pct": 0.008,
                "net_pnl_usd": 0.04,
                "balance_after": 50.04,
                "regime": "RANGING",
                "confidence": 0.8,
                "score": 0.9,
                "hold_seconds": 60,
                "result": "WIN" if idx < 9 else "LOSS",
            }
        )
    write_trade_log(trade_log, rows)

    analyzer = TradeAnalyzer(config)
    analyzer.daily_analysis()

    weights = json.loads((workspace / "data" / "signal_weights.json").read_text())
    params = json.loads((workspace / "data" / "strategy_params.json").read_text())

    assert pytest.approx(sum(weights.values()), rel=1e-6) == 1.0
    assert "execution_threshold" in params
    assert Path(workspace / "reports" / "daily").exists()
    assert Path(workspace / "data" / "learning_log.csv").exists()
