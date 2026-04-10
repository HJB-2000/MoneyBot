import json
import os
from datetime import datetime, timezone
from typing import List

from .base_strategy import BaseStrategy, Opportunity

POSITIONS_FILE = "data/funding_positions.json"


class FundingArbStrategy(BaseStrategy):
    def __init__(self):
        self.available = True
        self._positions: dict = self._load_positions()

    @property
    def name(self) -> str:
        return "funding_arb"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["funding_arb"]["enabled"]:
            return []
        if not self.available:
            return []
        if regime in ("VOLATILE", "BREAKOUT"):
            return []

        cfg = config["strategies"]["funding_arb"]
        min_rate = cfg["min_funding_rate"]
        trade_size = capital * config["capital"]["max_position_pct"] * size_mult
        trade_size = max(trade_size, 5.0)

        opportunities = []
        pairs = config["pairs"]["scan_universe"]
        spot_fee = self.TAKER_FEE
        futures_fee = 0.0004  # Binance futures taker fee

        for sym in pairs:
            try:
                rate = market_reader.get_funding_rate(sym)
                if rate is None or abs(rate) < min_rate:
                    continue

                ob_spot = market_reader.get_orderbook(sym, 5)
                if not ob_spot or not ob_spot.get("bids"):
                    continue

                entry_price = ob_spot["bids"][0][0]
                liq = self._liquidity_ratio(ob_spot, trade_size, entry_price)
                if liq < 15:
                    continue

                cost = 2 * spot_fee + 2 * futures_fee
                net_per_8h = abs(rate) - cost
                if net_per_8h <= 0.0004:
                    continue

                slip = self._slippage(trade_size)
                return opportunities.append(Opportunity(
                    strategy=self.name,
                    pair=sym,
                    direction="neutral",  # delta-neutral: long spot + short futures
                    entry_price=entry_price,
                    trade_size_usd=trade_size,
                    expected_profit_pct=abs(rate),
                    net_profit_pct=net_per_8h - slip,
                    fees_pct=cost,
                    slippage_pct=slip,
                    liquidity_ratio=liq,
                    exchange_latency_ms=market_reader.avg_latency_ms,
                    detected_at=datetime.now(timezone.utc),
                    regime=regime,
                    expiry_seconds=3600,
                    hold_max_seconds=8 * 3600,
                ))
            except Exception:
                continue

        return opportunities

    def _load_positions(self) -> dict:
        if os.path.exists(POSITIONS_FILE) and os.path.getsize(POSITIONS_FILE) > 0:
            try:
                with open(POSITIONS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
