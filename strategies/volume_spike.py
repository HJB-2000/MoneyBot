from datetime import datetime, timezone
from typing import List

from .base_strategy import BaseStrategy, Opportunity


class VolumeSpikeStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "volume_spike"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["volume_spike"]["enabled"]:
            return []
        if regime == "VOLATILE":
            return []

        cfg = config["strategies"]["volume_spike"]
        vol_mult = cfg["volume_multiplier"]
        window = cfg["window_minutes"]
        stop_pct = cfg["stop_loss_pct"]
        target_pct = 0.008

        cvd_obj = signal_objects.get("cvd")
        whale_obj = signal_objects.get("whale")
        trade_size = capital * config["capital"]["max_position_pct"] * size_mult * 0.6
        trade_size = max(trade_size, 5.0)

        pairs = config["pairs"]["scan_universe"]
        opportunities = []

        for sym in pairs:
            try:
                candles = market_reader.get_candles(sym, "5m", limit=50)
                if candles is None or len(candles) < window + 5:
                    continue

                current_vol = candles["volume"].values[-1]
                avg_vol = candles["volume"].values[-25:-1].mean()
                if avg_vol == 0 or current_vol < vol_mult * avg_vol:
                    continue

                close_now = candles["close"].values[-1]
                close_ago = candles["close"].values[-window]
                if close_ago == 0:
                    continue
                price_move = (close_now - close_ago) / close_ago

                if abs(price_move) < 0.005:
                    continue

                direction = "long" if price_move > 0 else "short"

                # CVD must confirm direction
                cvd_score = signals.get("cvd", 0)
                if direction == "long" and cvd_score < -0.2:
                    continue
                if direction == "short" and cvd_score > 0.2:
                    continue

                # Whale not moving opposite
                if direction == "long" and getattr(whale_obj, "whale_selling", False):
                    continue
                if direction == "short" and getattr(whale_obj, "whale_buying", False):
                    continue

                ob = market_reader.get_orderbook(sym, 5)
                if not ob:
                    continue
                entry_price = ob["asks"][0][0] if direction == "long" and ob.get("asks") \
                    else ob["bids"][0][0] if ob.get("bids") else 0
                if entry_price == 0:
                    continue

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
                    expected_profit_pct=target_pct,
                    net_profit_pct=target_pct - fees_pct - slip,
                    fees_pct=fees_pct,
                    slippage_pct=slip,
                    liquidity_ratio=liq,
                    exchange_latency_ms=market_reader.avg_latency_ms,
                    detected_at=datetime.now(timezone.utc),
                    regime=regime,
                    expiry_seconds=20,
                    target_price=entry_price * (1 + target_pct) if direction == "long"
                                 else entry_price * (1 - target_pct),
                    stop_price=entry_price * (1 - stop_pct) if direction == "long"
                               else entry_price * (1 + stop_pct),
                    hold_max_seconds=int(cfg["max_hold_minutes"] * 60),
                ))
            except Exception:
                continue

        return opportunities
