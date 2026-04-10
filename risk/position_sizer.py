class PositionSizer:
    def __init__(self, config: dict, tracker):
        self._cfg = config
        self._tracker = tracker

    def get_size(self, capital: float, signal_objects: dict = None,
                 win_rate: float = 0.5, avg_win: float = 0.01,
                 avg_loss: float = 0.01, regime_mult: float = 1.0) -> float:
        """
        Returns trade size in USD applying Kelly + ATR + tier cap.
        regime_mult is applied by the engine after this call.
        """
        trade_count = self._estimate_trade_count()

        if trade_count < self._cfg["capital"]["kelly_min_trades"]:
            # Pre-50 trades: flat 2%
            size = capital * 0.02
        else:
            # Kelly criterion
            if avg_loss == 0:
                size = capital * 0.02
            else:
                kelly_full = win_rate - (1 - win_rate) / (avg_win / avg_loss)
                kelly_full = max(0.0, kelly_full)
                size = kelly_full * self._cfg["capital"]["kelly_fraction"] * capital

        # ATR adjustment
        vol_obj = (signal_objects or {}).get("volatility")
        atr_ratio = getattr(vol_obj, "atr_ratio", 1.0)
        if atr_ratio > 1.5:
            size *= 0.7
        elif atr_ratio < 0.7:
            size *= 1.1

        # Capital tier cap
        cap = self._tier_cap(capital)
        size = min(size, capital * cap)

        return max(2.0, round(size, 4))

    def _tier_cap(self, capital: float) -> float:
        tiers = self._cfg["capital_tiers"]
        if capital < 500:
            return tiers["below_500"]
        elif capital < 1000:
            return tiers["500_to_1000"]
        elif capital < 2000:
            return tiers["1000_to_2000"]
        else:
            return tiers["above_2000"]

    def _estimate_trade_count(self) -> int:
        """Read total trades from trade log as a proxy."""
        import os, csv
        path = "data/trade_log.csv"
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path) as f:
            reader = csv.DictReader(f)
            for _ in reader:
                count += 1
        return count
