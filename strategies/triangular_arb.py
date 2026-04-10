from datetime import datetime, timezone
from typing import List
from .base_strategy import BaseStrategy, Opportunity


class TriangularArbStrategy(BaseStrategy):
    """
    Scans triangular paths on Binance spot.
    Path format: [USDT, COIN_A, COIN_B, USDT]
    Leg1: buy COIN_A with USDT (use ask)
    Leg2: buy COIN_B with COIN_A (use ask)
    Leg3: sell COIN_B for USDT  (use bid)  ← direction='sell'
    """

    @property
    def name(self) -> str:
        return "triangular_arb"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["triangular_arb"]["enabled"]:
            return []
        if regime == "VOLATILE":
            return []

        # Only run in arb-friendly conditions
        combiner_result = signal_objects.get("_combiner_result")
        if combiner_result and not combiner_result.arb_friendly:
            return []

        cfg = config["strategies"]["triangular_arb"]
        min_profit = cfg["min_profit_pct"]
        paths = config.get("triangular_paths", [])
        trade_size = capital * config["capital"]["max_position_pct"] * size_mult
        trade_size = max(trade_size, 5.0)  # minimum $5

        opportunities = []
        for path in paths:
            try:
                opp = self._scan_path(path, trade_size, market_reader,
                                      min_profit, regime, config)
                if opp:
                    opportunities.append(opp)
            except Exception:
                continue

        return opportunities

    def _scan_path(self, path: list, trade_size: float, market_reader,
                   min_profit: float, regime: str, config: dict):
        if len(path) != 4:
            return None
        _, coin_a, coin_b, _ = path

        # Build symbols for the three legs
        sym_ab = f"{coin_a}/USDT"
        sym_cb = f"{coin_b}/{coin_a}"  # coin_b bought with coin_a
        sym_bc = f"{coin_b}/USDT"     # sell coin_b for USDT

        # Fetch orderbooks
        ob_ab = market_reader.get_orderbook(sym_ab, 5)
        ob_bc = market_reader.get_orderbook(sym_bc, 5)

        # Try both routing options and pick the profitable one
        for use_cross in [False, True]:
            result = self._simulate(
                trade_size, coin_a, coin_b,
                ob_ab, ob_bc,
                market_reader, use_cross, config
            )
            if result and result["net_profit_pct"] > min_profit:
                liq = min(
                    self._liquidity_ratio(ob_ab, trade_size, result["price_a"]),
                    self._liquidity_ratio(ob_bc, trade_size, result["price_b"]),
                )
                if liq < 15:
                    continue
                return Opportunity(
                    strategy=self.name,
                    pair=f"{coin_a}→{coin_b}→USDT",
                    direction="neutral",
                    entry_price=result["price_a"],
                    trade_size_usd=trade_size,
                    expected_profit_pct=result["gross_profit_pct"],
                    net_profit_pct=result["net_profit_pct"],
                    fees_pct=result["fees_pct"],
                    slippage_pct=result["slippage_pct"],
                    liquidity_ratio=liq,
                    exchange_latency_ms=market_reader.avg_latency_ms,
                    detected_at=datetime.now(timezone.utc),
                    regime=regime,
                    expiry_seconds=3,
                    path=path,
                )
        return None

    def _simulate(self, size_usd: float, coin_a: str, coin_b: str,
                  ob_ab: dict, ob_bc: dict, market_reader, use_cross: bool, config: dict):
        """Simulate: USDT → coin_A → coin_B → USDT"""
        if not ob_ab or not ob_bc:
            return None
        if not ob_ab.get("asks") or not ob_bc.get("bids"):
            return None

        slip = self._slippage(size_usd)
        fee = self.TAKER_FEE
        total_cost = fee + slip

        # Leg 1: buy coin_A with USDT
        ask_a = ob_ab["asks"][0][0] * (1 + slip)
        if ask_a == 0:
            return None
        amount_a = size_usd / ask_a

        # Leg 2: buy coin_B with coin_A (try coin_B/coin_A first, else use USDT prices)
        if use_cross:
            sym_cross = f"{coin_b}/{coin_a}"
            ob_cross = market_reader.get_orderbook(sym_cross, 5)
            if ob_cross and ob_cross.get("asks"):
                ask_cross = ob_cross["asks"][0][0] * (1 + slip)
                amount_b = (amount_a * (1 - fee)) / ask_cross if ask_cross > 0 else 0
            else:
                return None
        else:
            # Route through USDT: sell coin_A, buy coin_B — two extra legs
            # Simplified: estimate via USDT prices
            bid_a = ob_ab["bids"][0][0] * (1 - slip) if ob_ab.get("bids") else 0
            asks_b = ob_bc["asks"][0][0] * (1 + slip) if ob_bc.get("asks") else 0
            if bid_a == 0 or asks_b == 0:
                return None
            usdt_from_a = amount_a * (1 - fee) * bid_a
            amount_b = usdt_from_a / asks_b * (1 - fee)

        # Leg 3: sell coin_B for USDT (direction='sell' — CRITICAL)
        bid_b = ob_bc["bids"][0][0] * (1 - slip)
        if bid_b == 0 or amount_b == 0:
            return None
        usdt_out = amount_b * (1 - fee) * bid_b

        gross_profit_pct = (usdt_out - size_usd) / size_usd
        fees_pct = total_cost * 3   # 3 legs
        slippage_pct = slip * 3
        net_profit_pct = gross_profit_pct - fees_pct

        return {
            "gross_profit_pct": gross_profit_pct,
            "net_profit_pct": net_profit_pct,
            "fees_pct": fees_pct,
            "slippage_pct": slippage_pct,
            "price_a": ob_ab["asks"][0][0],
            "price_b": ob_bc["bids"][0][0],
        }
