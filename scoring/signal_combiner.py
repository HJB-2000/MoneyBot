import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_WEIGHTS = {
    "price_action":   0.15,
    "momentum":       0.15,
    "microstructure": 0.20,
    "volatility":     0.15,
    "sentiment":      0.10,
    "cvd":            0.15,
    "vwap":           0.05,
    "whale":          0.05,
}

WEIGHTS_FILE = "data/signal_weights.json"


@dataclass
class ConsensusResult:
    score: float = 0.0
    confidence: float = 0.0
    dominant_signal: str = ""
    agreement_count: int = 0
    disagreement_flags: List[str] = field(default_factory=list)
    arb_friendly: bool = False


class SignalCombiner:
    def __init__(self, config: dict = None):
        self._weights = self._load_weights(config)

    def _load_weights(self, config: dict) -> dict:
        if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 0:
            try:
                with open(WEIGHTS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        if config and "signal_weights" in config:
            return dict(config["signal_weights"])
        return dict(DEFAULT_WEIGHTS)

    def reload_weights(self):
        self._weights = self._load_weights(None)

    def combine(self, signals: dict, signal_objects: dict = None) -> ConsensusResult:
        """
        signals: dict of signal_name → float score (-1 to +1)
        signal_objects: dict of signal_name → signal instance (for attribute access)
        """
        result = ConsensusResult()
        if not signals:
            return result

        weighted_sum = 0.0
        total_weight = 0.0
        directions = []

        for name, score in signals.items():
            w = self._weights.get(name, 0.05)
            weighted_sum += score * w
            total_weight += w
            directions.append(score)

        if total_weight > 0:
            result.score = weighted_sum / total_weight

        # Confidence = how much signals agree on direction (0 for perfect split)
        positive = sum(1 for s in directions if s > 0.05)
        negative = sum(1 for s in directions if s < -0.05)
        n = len(directions)
        result.agreement_count = max(positive, negative)
        result.confidence = abs(positive - negative) / n if n > 0 else 0.0

        # Disagreement flags
        majority_dir = "up" if positive > negative else "down"
        for name, score in signals.items():
            if majority_dir == "up" and score < -0.1:
                result.disagreement_flags.append(name)
            elif majority_dir == "down" and score > 0.1:
                result.disagreement_flags.append(name)

        # Dominant signal
        if signals:
            result.dominant_signal = max(signals, key=lambda k: abs(signals[k]))

        # Special overrides
        objs = signal_objects or {}
        cvd_obj = objs.get("cvd")
        whale_obj = objs.get("whale")
        vol_obj = objs.get("volatility")
        micro_obj = objs.get("microstructure")

        if cvd_obj and getattr(cvd_obj, "cvd_divergence", False):
            result.confidence = min(1.0, result.confidence + 0.15)

        if whale_obj and (getattr(whale_obj, "whale_buying", False) or
                          getattr(whale_obj, "whale_selling", False)):
            result.confidence = min(1.0, result.confidence + 0.10)

        if vol_obj and getattr(vol_obj, "vol_spike", False):
            result.confidence = max(0.0, result.confidence - 0.30)

        if micro_obj and getattr(micro_obj, "spread_pct", 0) > 0.002:
            result.confidence = max(0.0, result.confidence - 0.20)

        result.arb_friendly = self.is_arb_friendly(signals, signal_objects)
        return result

    def is_arb_friendly(self, signals: dict, signal_objects: dict = None) -> bool:
        """
        Returns True only when conditions are optimal for arbitrage strategies.
        """
        objs = signal_objects or {}
        vol_obj = objs.get("volatility")
        micro_obj = objs.get("microstructure")
        whale_obj = objs.get("whale")
        cvd_obj = objs.get("cvd")

        # Volatility must be low
        if vol_obj:
            if getattr(vol_obj, "atr_ratio", 1.0) >= 1.3:
                return False
            if getattr(vol_obj, "vol_spike", False):
                return False

        # Spread must be tight
        if micro_obj and getattr(micro_obj, "spread_pct", 0) > 0.002:
            return False

        # No whale activity
        if whale_obj:
            if getattr(whale_obj, "whale_buying", False) or getattr(whale_obj, "whale_selling", False):
                return False

        # OFI near zero (balanced)
        micro_score = signals.get("microstructure", 0)
        if abs(micro_score) > 0.5:
            return False

        # CVD flat (no strong directional pressure)
        if cvd_obj and getattr(cvd_obj, "cvd_divergence", False):
            return False

        return True
