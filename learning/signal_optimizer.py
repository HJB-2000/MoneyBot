import json
import os

WEIGHTS_FILE = "data/signal_weights.json"
MAX_CHANGE = 0.02
MIN_WEIGHT = 0.03
MAX_WEIGHT = 0.40

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


class SignalOptimizer:
    def __init__(self, config: dict):
        self._config = config

    def update_weights(self, signal_accuracy: dict):
        weights = self._load()

        for signal, accuracy in signal_accuracy.items():
            if signal not in weights:
                continue
            if accuracy > 0.65:
                weights[signal] = min(MAX_WEIGHT, weights[signal] + 0.01)
            elif accuracy < 0.50:
                weights[signal] = max(MIN_WEIGHT, weights[signal] - 0.01)

        # Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: round(v / total, 4) for k, v in weights.items()}
            # Fix floating-point rounding so sum is exactly 1.0
            diff = round(1.0 - sum(weights.values()), 4)
            if diff != 0.0:
                largest = max(weights, key=weights.get)
                weights[largest] = round(weights[largest] + diff, 4)

        with open(WEIGHTS_FILE, "w") as f:
            json.dump(weights, f, indent=2)

    def _load(self) -> dict:
        if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 0:
            try:
                with open(WEIGHTS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return dict(DEFAULT_WEIGHTS)
