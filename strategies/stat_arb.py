import json
import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy, Opportunity

CORR_MATRIX_FILE = "data/correlation_matrix.json"
POSITIONS_FILE = "data/stat_arb_positions.json"


class StatArbStrategy(BaseStrategy):
    def __init__(self):
        self._last_matrix_build = None
        self._corr_pairs: dict = {}
        self._positions: dict = self._load_positions()

    @property
    def name(self) -> str:
        return "stat_arb"

    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects, market_reader, capital: float, config: dict) -> List[Opportunity]:
        if not config["strategies"]["stat_arb"]["enabled"]:
            return []
        if regime == "VOLATILE":
            return []

        cfg = config["strategies"]["stat_arb"]
        bias = getattr(signal_objects.get("_route_result"), "bias", None)

        # Rebuild correlation matrix every hour
        if self._should_rebuild():
            self._build_matrix(config, market_reader)

        if not self._corr_pairs:
            return []

        trade_size = capital * config["capital"]["max_position_pct"] * size_mult
        trade_size = max(trade_size, 5.0)

        opportunities = []
        for (sym_a, sym_b), stats in self._corr_pairs.items():
            try:
                opp = self._check_divergence(
                    sym_a, sym_b, stats, trade_size,
                    market_reader, cfg, regime, bias
                )
                if opp:
                    opportunities.append(opp)
            except Exception:
                continue

        return opportunities

    def _should_rebuild(self) -> bool:
        if self._last_matrix_build is None:
            return True
        return (datetime.now(timezone.utc) - self._last_matrix_build) > timedelta(hours=1)

    def _build_matrix(self, config: dict, market_reader):
        """Fetch 48h of 5m candles and compute Pearson correlations."""
        pairs = config["pairs"]["scan_universe"]
        threshold = config["strategies"]["stat_arb"]["correlation_threshold"]
        price_data = {}

        for sym in pairs:
            candles = market_reader.get_candles(sym, "5m", limit=100)
            if candles is not None and len(candles) >= 30:
                price_data[sym] = candles["close"].values.astype(float)

        corr_pairs = {}
        syms = list(price_data.keys())
        for i in range(len(syms)):
            for j in range(i + 1, len(syms)):
                a, b = syms[i], syms[j]
                series_a = price_data[a]
                series_b = price_data[b]
                min_len = min(len(series_a), len(series_b))
                if min_len < 20:
                    continue
                sa = series_a[-min_len:]
                sb = series_b[-min_len:]
                corr = np.corrcoef(sa, sb)[0, 1]
                if corr >= threshold:
                    ratio = sa / sb
                    corr_pairs[(a, b)] = {
                        "correlation": float(corr),
                        "mean_ratio": float(ratio.mean()),
                        "std_ratio": float(ratio.std()),
                    }

        self._corr_pairs = corr_pairs
        self._last_matrix_build = datetime.now(timezone.utc)

        # Persist
        serializable = {f"{k[0]}|{k[1]}": v for k, v in corr_pairs.items()}
        with open(CORR_MATRIX_FILE, "w") as f:
            json.dump(serializable, f)

    def _check_divergence(self, sym_a: str, sym_b: str, stats: dict,
                           trade_size: float, market_reader, cfg: dict,
                           regime: str, bias: str):
        ticker_a = market_reader.get_ticker(sym_a)
        ticker_b = market_reader.get_ticker(sym_b)
        if not ticker_a or not ticker_b:
            return None

        price_a = ticker_a.get("last", 0)
        price_b = ticker_b.get("last", 0)
        if price_a == 0 or price_b == 0:
            return None

        current_ratio = price_a / price_b
        mean = stats["mean_ratio"]
        std = stats["std_ratio"]
        if std == 0:
            return None

        z_score = (current_ratio - mean) / std
        entry_z = cfg["z_score_entry"]

        # Bias filter
        if bias == "long_only" and z_score > -entry_z:
            return None  # only want to buy cheap pair
        if bias == "short_only" and z_score < entry_z:
            return None

        ob = None
        if abs(z_score) >= entry_z:
            # Determine which pair is cheap
            if z_score > entry_z:
                # sym_a expensive → buy sym_b
                pair = sym_b
                direction = "long"
                ob = market_reader.get_orderbook(sym_b, 5)
                entry_price = price_b
            else:
                # sym_b expensive → buy sym_a
                pair = sym_a
                direction = "long"
                ob = market_reader.get_orderbook(sym_a, 5)
                entry_price = price_a

            liq = self._liquidity_ratio(ob, trade_size, entry_price) if ob else 0
            if liq < 5:
                return None

            fees_pct = 2 * self.TAKER_FEE
            slip = self._slippage(trade_size)
            net = cfg["stop_loss_pct"] * 0.5 - fees_pct - slip  # conservative estimate

            return Opportunity(
                strategy=self.name,
                pair=pair,
                direction=direction,
                entry_price=entry_price,
                trade_size_usd=trade_size,
                expected_profit_pct=cfg["stop_loss_pct"] * 0.5,
                net_profit_pct=net,
                fees_pct=fees_pct,
                slippage_pct=slip,
                liquidity_ratio=liq,
                exchange_latency_ms=market_reader.avg_latency_ms,
                detected_at=datetime.now(timezone.utc),
                regime=regime,
                expiry_seconds=300,
                z_score=z_score,
                stop_price=entry_price * (1 - cfg["stop_loss_pct"]),
                target_price=entry_price * (1 + cfg["stop_loss_pct"] * 0.5),
                hold_max_seconds=int(cfg["max_hold_hours"] * 3600),
            )
        return None

    def _load_positions(self) -> dict:
        if os.path.exists(POSITIONS_FILE) and os.path.getsize(POSITIONS_FILE) > 0:
            try:
                with open(POSITIONS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
