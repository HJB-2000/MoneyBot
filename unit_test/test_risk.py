from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import yaml

from risk.manager import RiskManager
from risk.position_sizer import PositionSizer
from strategies.base_strategy import Opportunity


def make_tracker(capital=500.0, daily_pnl=0.0, drawdown=0.0, win_rate=0.6, losses=0):
    return SimpleNamespace(
        get_capital=lambda: capital,
        get_daily_pnl=lambda: daily_pnl,
        get_drawdown=lambda: drawdown,
        get_win_rate=lambda last_n=20: win_rate,
        get_consecutive_losses=lambda: losses,
    )


def make_opp(**kwargs):
    defaults = dict(
        strategy="triangular_arb",
        pair="SOL/USDT",
        direction="neutral",
        entry_price=100.0,
        trade_size_usd=5.0,
        expected_profit_pct=0.001,
        net_profit_pct=0.0005,
        fees_pct=0.002,
        slippage_pct=0.001,
        liquidity_ratio=20.0,
        exchange_latency_ms=50.0,
        detected_at=datetime.now(timezone.utc),
        expiry_seconds=60,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


@pytest.fixture(scope="module")
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def test_approve_rejects_below_survival_floor(config):
    tracker = make_tracker(capital=0.5)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    assert risk.approve(make_opp()).approved is False


def test_approve_rejects_daily_loss_limit(config):
    tracker = make_tracker(capital=500.0, daily_pnl=-50.0)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    assert risk.approve(make_opp()).approved is False


def test_approve_rejects_max_open_trades(config):
    tracker = make_tracker()
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    risk._open_trades = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert risk.approve(make_opp()).approved is False


def test_approve_reduces_size_on_cb1(config):
    tracker = make_tracker(losses=3)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    result = risk.approve(make_opp())
    assert result.approved is True
    size_cb1 = result.size

    tracker2 = make_tracker(losses=0)
    risk2 = RiskManager(config, tracker2, PositionSizer(config, tracker2))
    size_normal = risk2.approve(make_opp()).size
    assert size_cb1 < size_normal


def test_approve_rejects_cb2_daily_drawdown(config):
    tracker = make_tracker(drawdown=0.16)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    assert risk.approve(make_opp()).approved is False


def test_approve_reduces_size_on_cb3(config):
    tracker = make_tracker(win_rate=0.5)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    result = risk.approve(make_opp())
    assert result.approved is True
    assert risk.cb3_active is True
    assert result.size > 0


def test_approve_passes_all_gates_clean(config):
    tracker = make_tracker(capital=500.0, win_rate=0.6, losses=0)
    risk = RiskManager(config, tracker, PositionSizer(config, tracker))
    result = risk.approve(make_opp())
    assert result.approved is True
    assert result.size > 0


def test_position_sizer_pre_kelly(config):
    tracker = make_tracker(capital=500.0)
    sizer = PositionSizer(config, tracker)
    sizer._estimate_trade_count = lambda: 30
    assert sizer.get_size(500.0) == pytest.approx(10.0)


def test_position_sizer_post_kelly(config):
    tracker = make_tracker(capital=1000.0)
    sizer = PositionSizer(config, tracker)
    sizer._estimate_trade_count = lambda: 60
    size = sizer.get_size(1000.0, win_rate=0.52, avg_win=0.02, avg_loss=0.01)
    assert size == pytest.approx(40.0, rel=1e-2)


def test_position_sizer_respects_tier_cap(config):
    tracker = make_tracker(capital=300.0)
    sizer = PositionSizer(config, tracker)
    sizer._estimate_trade_count = lambda: 60
    size = sizer.get_size(300.0, win_rate=0.9, avg_win=0.03, avg_loss=0.01)
    assert size <= 300.0 * config["capital_tiers"]["below_500"]


def test_position_sizer_atr_adjustment(config):
    tracker = make_tracker(capital=1000.0)
    sizer = PositionSizer(config, tracker)
    sizer._estimate_trade_count = lambda: 30
    base = sizer.get_size(1000.0)
    signal_objects = {"volatility": SimpleNamespace(atr_ratio=1.8)}
    adjusted = sizer.get_size(1000.0, signal_objects=signal_objects)
    assert adjusted < base
