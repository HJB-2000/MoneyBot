import json
import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np

from .base_strategy import BaseStrategy, Opportunity

CORR_FILE = "data/correlation_matrix.json"


class CorrelationBreakoutStrategy(BaseStrategy):
    def __init__(self):
        self._btc_move_time: datetime = None
        self._btc_move_dir: str = None
        self._btc_move_pct: float = 0.0

    @property
    def name(self) -> str:
        return "correlation_breakout"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["correlation_breakout"]["enabled"]:
            return []
        if regime not in ("TRENDING_UP", "TRENDING_DOWN", "BREAKOUT", "WHALE_MOVING", "RANGING"):
            return []

        cfg = config["strategies"]["correlation_breakout"]
        btc_threshold = cfg["btc_move_threshold"]
        lag_window = cfg["lag_window_minutes"]
        min_corr = cfg["min_correlation"]
        stop_pct = cfg["stop_loss_pct"]
        trade_size = capital * config["capital"]["max_position_pct"] * size_mult
        trade_size = max(trade_size, 5.0)

        # Detect BTC move
        btc_candles = market_reader.get_candles("BTC/USDT", "5m", limit=10)
        if btc_candles is None or len(btc_candles) < 5:
            return []

        btc_close_now = btc_candles["close"].values[-1]
        btc_close_ago = btc_candles["close"].values[-5]
        if btc_close_ago == 0:
            return []
        btc_move = (btc_close_now - btc_close_ago) / btc_close_ago

        if abs(btc_move) >= btc_threshold:
            self._btc_move_time = datetime.now(timezone.utc)
            self._btc_move_dir = "up" if btc_move > 0 else "down"
            self._btc_move_pct = abs(btc_move)
        elif regime == "RANGING":
            return []  # no BTC move in ranging, skip

        # Check if BTC move is recent enough
        if self._btc_move_time is None:
            return []
        if (datetime.now(timezone.utc) - self._btc_move_time) > timedelta(minutes=lag_window):
            return []

        # Load correlation data
        corr_pairs = self._load_corr()
        opportunities = []
        pairs = config["pairs"]["scan_universe"]

        for sym in pairs:
            if sym == "BTC/USDT":
                continue
            # Check correlation with BTC
            btc_corr = self._get_btc_corr(sym, corr_pairs, min_corr, market_reader)
            if btc_corr < min_corr:
                continue

            # Check if this pair already moved
            candles = market_reader.get_candles(sym, "5m", limit=6)
            if candles is None or len(candles) < 5:
                continue
            pair_move = (candles["close"].values[-1] - candles["close"].values[-5]) / \
                        (candles["close"].values[-5] + 1e-9)

            # If pair already moved in same direction: lag captured, skip
            if self._btc_move_dir == "up" and pair_move > btc_threshold * 0.5:
                continue
            if self._btc_move_dir == "down" and pair_move < -btc_threshold * 0.5:
                continue

            ob = market_reader.get_orderbook(sym, 5)
            if not ob:
                continue

            direction = "long" if self._btc_move_dir == "up" else "short"
            entry_price = ob["asks"][0][0] if direction == "long" and ob.get("asks") \
                else ob["bids"][0][0] if ob.get("bids") else 0
            if entry_price == 0:
                continue

            expected_move = self._btc_move_pct * btc_corr * 0.7
            target_move = expected_move * 0.7

            fees_pct = 2 * self.TAKER_FEE
            slip = self._slippage(trade_size)
            liq = self._liquidity_ratio(ob, trade_size, entry_price)
            if liq < 5:
                continue

            opportunities.append(Opportunity(
                strategy=self.name,
                pair=sym,
                direction=direction,
                entry_price=entry_price,
                trade_size_usd=trade_size,
                expected_profit_pct=target_move,
                net_profit_pct=target_move - fees_pct - slip,
                fees_pct=fees_pct,
                slippage_pct=slip,
                liquidity_ratio=liq,
                exchange_latency_ms=market_reader.avg_latency_ms,
                detected_at=datetime.now(timezone.utc),
                regime=regime,
                expiry_seconds=60,
                target_price=entry_price * (1 + target_move) if direction == "long"
                             else entry_price * (1 - target_move),
                stop_price=entry_price * (1 - stop_pct) if direction == "long"
                           else entry_price * (1 + stop_pct),
                hold_max_seconds=lag_window * 60,
            ))

        return opportunities

    def _load_corr(self) -> dict:
        if os.path.exists(CORR_FILE) and os.path.getsize(CORR_FILE) > 0:
            try:
                with open(CORR_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_btc_corr(self, sym: str, corr_pairs: dict,
                      min_corr: float, market_reader) -> float:
        """Compute live Pearson correlation with BTC/USDT."""
        try:
            btc_c = market_reader.get_candles("BTC/USDT", "5m", limit=50)
            sym_c = market_reader.get_candles(sym, "5m", limit=50)
            if btc_c is None or sym_c is None:
                return 0.0
            n = min(len(btc_c), len(sym_c))
            if n < 20:
                return 0.0
            corr = float(np.corrcoef(btc_c["close"].values[-n:],
                                     sym_c["close"].values[-n:])[0, 1])
            return max(0.0, corr)
        except Exception:
            return 0.0
