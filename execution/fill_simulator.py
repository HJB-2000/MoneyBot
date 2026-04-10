import random
from strategies.base_strategy import Opportunity

TAKER_FEE = 0.001  # 0.10%


class FillSimulator:
    """Realistic paper trade fill model."""

    def simulate(self, opp: Opportunity) -> dict:
        """
        Returns fill details: entry_price, exit_price, net_pnl_usd, result.
        85% fill at best price, 15% at 1 tick worse.
        """
        entry = opp.entry_price
        size = opp.trade_size_usd

        # Slight fill realism: 15% of the time get 1 tick worse
        if random.random() < 0.15:
            tick = entry * 0.0001
            entry = entry + tick if opp.direction in ("long", "neutral") else entry - tick

        # Simulate exit based on expected profit
        net_pct = opp.net_profit_pct
        slippage = opp.slippage_pct

        if opp.direction == "long":
            exit_price = entry * (1 + net_pct + 2 * TAKER_FEE + slippage)
        elif opp.direction == "short":
            exit_price = entry * (1 - net_pct - 2 * TAKER_FEE - slippage)
        else:
            # neutral (arb): profit is the net_pct directly
            exit_price = entry

        # Net PnL
        if opp.direction == "neutral":
            gross = size * net_pct
        elif opp.direction == "long":
            gross = size * ((exit_price - entry) / entry)
        else:
            gross = size * ((entry - exit_price) / entry)

        fees = size * (2 * TAKER_FEE)
        slip_cost = size * slippage
        net_pnl = gross - fees - slip_cost

        result = "WIN" if net_pnl > 0 else "LOSS" if net_pnl < -size * 0.001 else "SCRATCH"

        return {
            "entry_price": round(entry, 8),
            "exit_price": round(exit_price, 8),
            "gross_profit_pct": round(net_pct + 2 * TAKER_FEE, 6),
            "fees_pct": round(2 * TAKER_FEE, 6),
            "slippage_pct": round(slippage, 6),
            "net_profit_pct": round(net_pct, 6),
            "net_pnl_usd": round(net_pnl, 6),
            "result": result,
        }
