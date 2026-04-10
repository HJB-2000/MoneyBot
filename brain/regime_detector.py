import csv
import os
from datetime import datetime, timezone


REGIME_LOG = "data/regime_log.csv"


class RegimeDetector:
    def __init__(self, config: dict):
        self._cfg = config["regime"]
        self._current_regime = "CHOPPY"
        self._ensure_log()

    def _ensure_log(self):
        if not os.path.exists(REGIME_LOG) or os.path.getsize(REGIME_LOG) == 0:
            with open(REGIME_LOG, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "regime", "confidence", "score",
                                 "atr_ratio", "vol_spike", "funding_rate", "cvd_divergence",
                                 "whale_buying", "whale_selling", "breakout"])

    def classify(self, signals: dict, combiner_result, signal_objects: dict = None) -> str:
        objs = signal_objects or {}
        vol_obj = objs.get("volatility")
        whale_obj = objs.get("whale")
        pa_obj = objs.get("price_action")
        sent_obj = objs.get("sentiment")

        score = combiner_result.score
        atr_ratio = getattr(vol_obj, "atr_ratio", 1.0)
        vol_spike = getattr(vol_obj, "vol_spike", False)
        funding_rate = getattr(sent_obj, "funding_rate", 0.0)
        whale_buying = getattr(whale_obj, "whale_buying", False)
        whale_selling = getattr(whale_obj, "whale_selling", False)
        breakout = getattr(pa_obj, "breakout_detected", False)
        cvd_div = getattr(objs.get("cvd"), "cvd_divergence", False)

        # Priority 1 — VOLATILE (capital protection)
        if atr_ratio > float(self._cfg.get("volatile_atr_multiplier", 2.0)) or vol_spike:
            regime = "VOLATILE"

        # Priority 2 — WHALE_MOVING
        elif whale_buying or whale_selling:
            regime = "WHALE_MOVING"

        # Priority 3 — BREAKOUT
        elif breakout and pa_obj and pa_obj.breakout_detected:
            regime = "BREAKOUT"

        # Priority 4 — FUNDING_RICH
        elif funding_rate > float(self._cfg.get("funding_rich_rate", 0.0008)):
            regime = "FUNDING_RICH"

        # Priority 5 — RANGING (best for arb)
        elif (abs(score) <= 0.2
              and atr_ratio < 1.3
              and combiner_result.arb_friendly):
            regime = "RANGING"

        # Priority 6/7 — TRENDING
        elif score > float(self._cfg.get("trending_score_threshold", 0.4)):
            regime = "TRENDING_UP"

        elif score < -float(self._cfg.get("trending_score_threshold", 0.4)):
            regime = "TRENDING_DOWN"

        # Default — CHOPPY
        else:
            regime = "CHOPPY"

        self._log(regime, combiner_result, atr_ratio, vol_spike,
                  funding_rate, cvd_div, whale_buying, whale_selling, breakout)
        self._current_regime = regime
        return regime

    def _log(self, regime, cr, atr_ratio, vol_spike, funding_rate,
             cvd_div, whale_buying, whale_selling, breakout):
        with open(REGIME_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                regime,
                round(cr.confidence, 4),
                round(cr.score, 4),
                round(atr_ratio, 4),
                int(vol_spike),
                round(funding_rate, 6),
                int(cvd_div),
                int(whale_buying),
                int(whale_selling),
                int(breakout),
            ])

    @property
    def current_regime(self) -> str:
        return self._current_regime
