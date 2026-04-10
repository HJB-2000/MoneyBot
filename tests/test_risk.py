import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
from unittest.mock import MagicMock
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer
from strategies.base_strategy import Opportunity
from datetime import datetime


def make_opp(**kwargs):
    defaults = dict(
        strategy="triangular_arb", pair="SOL/USDT", direction="neutral",
        entry_price=100.0, trade_size_usd=5.0, expected_profit_pct=0.001,
        net_profit_pct=0.0005, fees_pct=0.002, slippage_pct=0.001,
        liquidity_ratio=20.0, exchange_latency_ms=50.0,
        detected_at=datetime.utcnow(), expiry_seconds=60,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def test_risk():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    print("\n=== Risk Tests ===")

    # Tracker mock
    tracker = MagicMock()
    tracker.get_capital.return_value = 500.0  # large enough to see CB1 reduction
    tracker.get_daily_pnl.return_value = 0.0
    tracker.get_drawdown.return_value = 0.0
    tracker.get_win_rate.return_value = 0.6
    tracker.get_consecutive_losses.return_value = 0

    sizer = PositionSizer(config, tracker)
    risk = RiskManager(config, tracker, sizer)

    # Capital below floor
    tracker.get_capital.return_value = 0.5
    tracker.get_daily_pnl.return_value = 0.0
    result = risk.approve(make_opp())
    assert not result.approved, "Should reject below floor"
    print("  Capital floor        PASS")

    # Reset capital
    tracker.get_capital.return_value = 500.0

    # Daily loss limit
    tracker.get_daily_pnl.return_value = -50.0  # -10% of 500
    result = risk.approve(make_opp())
    assert not result.approved, "Should reject on daily loss"
    print("  Daily loss limit     PASS")
    tracker.get_daily_pnl.return_value = 0.0

    # Max open trades
    risk._open_trades = ["a", "b", "c"]
    result = risk.approve(make_opp())
    assert not result.approved, "Should reject at max open trades"
    print("  Max open trades      PASS")
    risk._open_trades = []

    # CB1 active → size halved
    tracker.get_consecutive_losses.return_value = 3
    result = risk.approve(make_opp())
    assert result.approved
    size_cb1 = result.size
    tracker.get_consecutive_losses.return_value = 0
    result2 = risk.approve(make_opp())
    assert result2.approved
    size_normal = result2.size
    assert size_cb1 < size_normal, f"CB1 size {size_cb1} should be < normal {size_normal}"
    print(f"  CB1 size reduction   PASS  (${size_cb1:.4f} < ${size_normal:.4f})")

    # All clear → approved
    result = risk.approve(make_opp())
    assert result.approved, "Should approve when all clear"
    assert result.size > 0
    print(f"  All clear → approved PASS  size=${result.size:.4f}")

    # Size never exceeds tier cap ($500 * 6% = $30)
    assert result.size <= 500.0 * config["capital_tiers"]["500_to_1000"]
    print(f"  Tier cap             PASS  (size ≤ ${500.0 * config['capital_tiers']['500_to_1000']:.2f})")

    print("=== All risk tests PASSED ===\n")


if __name__ == "__main__":
    test_risk()
