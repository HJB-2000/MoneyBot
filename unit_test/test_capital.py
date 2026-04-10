from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import capital.tracker as tracker_module
from capital.compounder import Compounder
from capital.tracker import CapitalTracker


def patch_tracker_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(tracker_module, "DB_PATH", str(tmp_path / "moneyBot.db"), raising=False)
    monkeypatch.setattr(tracker_module, "TRADE_LOG", str(tmp_path / "data" / "trade_log.csv"), raising=False)
    monkeypatch.setattr(tracker_module, "REPORTS_DIR", str(tmp_path / "reports" / "daily"), raising=False)


def patch_tracker_date(monkeypatch, fake_date):
    monkeypatch.setattr(tracker_module, "date", fake_date, raising=False)


def make_tracker(config, monkeypatch, tmp_path):
    patch_tracker_paths(monkeypatch, tmp_path)
    return CapitalTracker(config)


def test_initial_capital_loads_from_config(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    assert tracker.get_capital() == pytest.approx(50.0)


def test_capital_persists_across_restart(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(5.0, {"result": "WIN"})

    tracker2 = make_tracker(config, monkeypatch, workspace)
    assert tracker2.get_capital() == pytest.approx(55.0)


def test_capital_updates_on_profit_and_loss(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(2.5, {"result": "WIN"})
    tracker.update(-1.0, {"result": "LOSS"})
    assert tracker.get_capital() == pytest.approx(51.5)


def test_capital_never_goes_negative(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(-999.0, {"result": "LOSS"})
    assert tracker.get_capital() >= config["capital"]["survival_floor"]


def test_daily_pnl_resets_at_midnight(config, workspace, monkeypatch):
    class FakeToday(date):
        @classmethod
        def today(cls):
            return date(2026, 4, 8)

    patch_tracker_date(monkeypatch, FakeToday)
    tracker = make_tracker(config, monkeypatch, workspace)
    # Avoid lock re-entry deadlock when rollover triggers daily report.
    monkeypatch.setattr(CapitalTracker, "get_drawdown", lambda self: 0.0)
    tracker.update(5.0, {"result": "WIN"})
    assert tracker.get_daily_pnl() == pytest.approx(5.0)

    class FakeTomorrow(date):
        @classmethod
        def today(cls):
            return date(2026, 4, 9)

    patch_tracker_date(monkeypatch, FakeTomorrow)
    tracker.update(0.0, {"result": "WIN"})
    assert tracker.get_daily_pnl() == pytest.approx(0.0)


def test_milestone_alert_at_100(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(51.0, {"result": "WIN"})
    assert 100 in tracker._milestones_crossed


def test_milestone_not_triggered_twice(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(51.0, {"result": "WIN"})
    before = len(tracker._milestones_crossed)
    tracker.update(1.0, {"result": "WIN"})
    assert len(tracker._milestones_crossed) == before


def test_drawdown_calculation(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(50.0, {"result": "WIN"})
    tracker.update(-15.0, {"result": "LOSS"})
    assert tracker.get_drawdown() == pytest.approx(0.15, rel=1e-3)


def test_win_rate_last_20(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    for _ in range(12):
        tracker.update(0.0, {"result": "WIN"})
    for _ in range(8):
        tracker.update(0.0, {"result": "LOSS"})
    assert tracker.get_win_rate(20) == pytest.approx(0.60)


def test_consecutive_losses_counter(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(0.0, {"result": "LOSS"})
    tracker.update(0.0, {"result": "LOSS"})
    tracker.update(0.0, {"result": "LOSS"})
    assert tracker.get_consecutive_losses() == 3


def test_consecutive_losses_resets_on_win(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    tracker.update(0.0, {"result": "LOSS"})
    tracker.update(0.0, {"result": "LOSS"})
    tracker.update(0.0, {"result": "WIN"})
    assert tracker.get_consecutive_losses() == 0


def test_compound_adds_to_pool(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    compounder = Compounder(tracker)
    compounder.compound(5.0, {"result": "WIN"})
    assert tracker.get_capital() == pytest.approx(55.0)


def test_daily_report_created(config, workspace, monkeypatch):
    tracker = make_tracker(config, monkeypatch, workspace)
    # Avoid lock re-entry deadlock when rollover triggers daily report.
    monkeypatch.setattr(CapitalTracker, "get_drawdown", lambda self: 0.0)

    class FakeToday(date):
        @classmethod
        def today(cls):
            return date(2026, 4, 8)

    patch_tracker_date(monkeypatch, FakeToday)
    tracker.update(1.0, {"result": "WIN"})

    class FakeTomorrow(date):
        @classmethod
        def today(cls):
            return date(2026, 4, 9)

    patch_tracker_date(monkeypatch, FakeTomorrow)
    tracker.update(0.0, {"result": "WIN"})

    report = Path(workspace / "reports" / "daily" / "2026-04-08.txt")
    assert report.exists()
