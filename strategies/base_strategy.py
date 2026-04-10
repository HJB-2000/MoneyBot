from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Opportunity:
    strategy: str
    pair: str
    direction: str            # 'long', 'short', 'neutral'
    entry_price: float
    trade_size_usd: float
    expected_profit_pct: float
    net_profit_pct: float
    fees_pct: float
    slippage_pct: float
    liquidity_ratio: float
    exchange_latency_ms: float
    detected_at: datetime
    score: float = 0.0
    regime: str = ""
    confidence: float = 0.0
    expiry_seconds: int = 60
    # Extra fields used by specific strategies
    stop_price: float = 0.0
    target_price: float = 0.0
    path: Optional[list] = None
    z_score: float = 0.0
    hold_max_seconds: int = 3600
    id: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = f"{self.strategy}_{self.pair}_{self.detected_at.timestamp():.0f}"


class BaseStrategy:
    def scan(self, regime: str, size_mult: float, signals: dict,
             signal_objects: dict, market_reader, capital: float,
             config: dict) -> List[Opportunity]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                      #
    # ------------------------------------------------------------------ #

    TAKER_FEE = 0.001  # 0.10% Binance taker fee

    def _slippage(self, size_usd: float) -> float:
        if size_usd < 100:
            return 0.0020
        elif size_usd < 500:
            return 0.0010
        elif size_usd < 2000:
            return 0.0007
        else:
            return 0.0005

    def _liquidity_ratio(self, orderbook: dict, size_usd: float, price: float) -> float:
        if not orderbook or price == 0:
            return 0.0
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        bid_depth_usd = sum(b[0] * b[1] for b in bids[:10])
        ask_depth_usd = sum(a[0] * a[1] for a in asks[:10])
        available = min(bid_depth_usd, ask_depth_usd)
        return available / size_usd if size_usd > 0 else 0.0
