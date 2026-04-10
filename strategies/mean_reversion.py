from datetime import datetime, timezone
from typing import List

from .base_strategy import BaseStrategy, Opportunity


class MeanReversionStrategy(BaseStrategy):
    @property
    def name(self) -> str:
        return "mean_reversion"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["mean_reversion"]["enabled"]:
            return []
        if regime not in ("RANGING", "TRENDING_UP", "CHOPPY"):
            return []

        cfg = config["strategies"]["mean_reversion"]
        drop_threshold = cfg["drop_threshold_pct"]
        drop_window = cfg["drop_window_minutes"]
        target_pct = cfg["target_recovery_pct"]
        stop_pct = cfg["stop_loss_pct"]
        trade_size = capital * config["capital"]["max_position_pct"] * size_mult
        trade_size = max(trade_size, 5.0)

        cvd_obj = signal_objects.get("cvd")
        pairs = config["pairs"]["scan_universe"]
        opportunities = []

        for sym in pairs:
            try:
                candles = market_reader.get_candles(sym, "5m", limit=drop_window + 5)
                if candles is None or len(candles) < drop_window:
                    continue

                close_now = candles["close"].values[-1]
                close_ago = candles["close"].values[-drop_window]
                if close_ago == 0:
                    continue

                drop = (close_ago - close_now) / close_ago
                if drop < drop_threshold:
                    continue

                # Filter: BTC not also dropping
                btc = market_reader.get_ticker("BTC/USDT")
                if btc:
                    # Use 5m candles for BTC drop check
                    btc_c = market_reader.get_candles("BTC/USDT", "5m", limit=6)
                    if btc_c is not None and len(btc_c) >= drop_window:
                        btc_drop = (btc_c["close"].values[-drop_window] -
                                    btc_c["close"].values[-1]) / btc_c["close"].values[-drop_window]
                        if btc_drop > 0.01:
                            continue  # systemic move, skip

                # Filter: volume not spiking > 5x
                volumes = candles["volume"].values
                avg_vol = volumes[:-1].mean()
                if avg_vol > 0 and volumes[-1] > avg_vol * 5:
                    continue   # real news selling

                # CVD absorption check
                cvd_absorbing = getattr(cvd_obj, "cvd_divergence", False)

                ob = market_reader.get_orderbook(sym, 5)
                if not ob or not ob.get("asks"):
                    continue
                entry_price = ob["asks"][0][0]

                fees_pct = 2 * self.TAKER_FEE
                slip = self._slippage(trade_size)
                net = target_pct - fees_pct - slip
                liq = self._liquidity_ratio(ob, trade_size, entry_price)
                if liq < 5:
                    continue

                opportunities.append(Opportunity(
                    strategy=self.name,
                    pair=sym,
                    direction="long",
                    entry_price=entry_price,
                    trade_size_usd=trade_size,
                    expected_profit_pct=target_pct,
                    net_profit_pct=net,
                    fees_pct=fees_pct,
                    slippage_pct=slip,
                    liquidity_ratio=liq,
                    exchange_latency_ms=market_reader.avg_latency_ms,
                    detected_at=datetime.now(timezone.utc),
                    regime=regime,
                    expiry_seconds=30,
                    target_price=entry_price * (1 + target_pct),
                    stop_price=entry_price * (1 - stop_pct),
                    hold_max_seconds=int(cfg["max_hold_minutes"] * 60),
                ))
            except Exception:
                continue

        return opportunities
