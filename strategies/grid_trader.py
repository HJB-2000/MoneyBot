import json
import os
from datetime import datetime, timezone
from typing import List

import numpy as np

from .base_strategy import BaseStrategy, Opportunity

GRIDS_FILE = "data/grids.json"


class GridTraderStrategy(BaseStrategy):
    def __init__(self):
        self._active_grids: dict = self._load_grids()

    @property
    def name(self) -> str:
        return "grid_trader"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["grid_trader"]["enabled"]:
            return []
        if regime == "VOLATILE":
            return []

        cfg = config["strategies"]["grid_trader"]
        max_grids = cfg["max_grids_running"]
        cap_pct = cfg["capital_per_grid_pct"]
        atr_mult = cfg["grid_spacing_atr_mult"]
        levels = cfg["grid_levels"]
        grid_pairs = config.get("grid_pairs", [])

        opportunities = []

        for sym in grid_pairs:
            if len(self._active_grids) >= max_grids:
                break

            # Pause if trending or volatile
            if regime in ("VOLATILE", "TRENDING_UP", "TRENDING_DOWN", "BREAKOUT"):
                continue

            if sym in self._active_grids:
                # Check if grid level was crossed → opportunity
                opp = self._check_grid_fill(sym, market_reader, regime, config)
                if opp:
                    opportunities.append(opp)
                continue

            # Set up new grid
            candles = market_reader.get_candles(sym)
            ob = market_reader.get_orderbook(sym, 5)
            if candles is None or ob is None or not ob.get("asks"):
                continue

            current_price = ob["asks"][0][0]
            atr = self._calc_atr(candles)
            if atr == 0:
                continue

            grid_range = atr * atr_mult * levels
            lower = current_price - grid_range / 2
            upper = current_price + grid_range / 2
            step = grid_range / levels
            grid_capital = capital * cap_pct

            self._active_grids[sym] = {
                "lower": lower,
                "upper": upper,
                "step": step,
                "levels": levels,
                "entry_price": current_price,
                "grid_capital": grid_capital,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "holdings": {},  # level → amount_held
            }
            self._save_grids()

        return opportunities

    def _check_grid_fill(self, sym: str, market_reader, regime: str, config: dict):
        grid = self._active_grids[sym]
        ob = market_reader.get_orderbook(sym, 5)
        if not ob or not ob.get("bids"):
            return None

        price = ob["bids"][0][0]

        # Check if price broke out of grid range
        if price < grid["lower"] or price > grid["upper"]:
            if regime in ("VOLATILE", "TRENDING_UP", "TRENDING_DOWN"):
                del self._active_grids[sym]
                self._save_grids()
                return None

        # Identify nearest buy level
        step = grid["step"]
        buy_level = grid["lower"] + step * int((price - grid["lower"]) / step)
        sell_level = buy_level + step
        fee = self.TAKER_FEE
        slip = self._slippage(grid["grid_capital"] / grid["levels"])
        round_trip_profit = step / price - 2 * fee - 2 * slip

        if round_trip_profit <= 0:
            return None

        trade_size = grid["grid_capital"] / grid["levels"]

        return Opportunity(
            strategy=self.name,
            pair=sym,
            direction="long",
            entry_price=buy_level,
            trade_size_usd=trade_size,
            expected_profit_pct=round_trip_profit + fee,
            net_profit_pct=round_trip_profit,
            fees_pct=2 * fee,
            slippage_pct=2 * slip,
            liquidity_ratio=self._liquidity_ratio(ob, trade_size, price),
            exchange_latency_ms=market_reader.avg_latency_ms,
            detected_at=datetime.now(timezone.utc),
            regime=regime,
            expiry_seconds=86400,
            target_price=sell_level,
            stop_price=grid["lower"],
            hold_max_seconds=86400,
        )

    def _calc_atr(self, candles, period: int = 14) -> float:
        highs = candles["high"].values.astype(float)
        lows = candles["low"].values.astype(float)
        closes = candles["close"].values.astype(float)
        tr = [max(highs[i] - lows[i],
                  abs(highs[i] - closes[i-1]),
                  abs(lows[i] - closes[i-1])) for i in range(1, len(candles))]
        return float(np.mean(tr[-period:])) if tr else 0.0

    def _load_grids(self) -> dict:
        if os.path.exists(GRIDS_FILE) and os.path.getsize(GRIDS_FILE) > 0:
            try:
                with open(GRIDS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_grids(self):
        with open(GRIDS_FILE, "w") as f:
            json.dump(self._active_grids, f)
