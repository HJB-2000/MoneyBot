import json
import os

PARAMS_FILE = "data/strategy_params.json"
MAX_CHANGE_PCT = 0.05  # 5% max change per day


class StrategyOptimizer:
    def __init__(self, config: dict):
        self._config = config

    def update_params(self, strategy_performance: dict):
        params = self._load()

        for strat, perf in strategy_performance.items():
            wr = perf.get("win_rate", 0.5)

            if strat == "grid_trader":
                key = "grid_spacing_atr_mult"
                current = params.get(key, self._config["strategies"]["grid_trader"]["grid_spacing_atr_mult"])
                if wr < 0.45:
                    params[key] = round(current * (1 + MAX_CHANGE_PCT), 4)  # widen grid

            elif strat == "stat_arb":
                key = "z_score_entry"
                current = params.get(key, self._config["strategies"]["stat_arb"]["z_score_entry"])
                if wr < 0.45:
                    params[key] = round(min(3.0, current + 0.1), 2)  # raise entry threshold

            elif strat == "mean_reversion":
                key = "drop_threshold_pct"
                current = params.get(key, self._config["strategies"]["mean_reversion"]["drop_threshold_pct"])
                if wr < 0.45:
                    params[key] = round(max(0.01, current - 0.002), 4)  # lower drop threshold

        with open(PARAMS_FILE, "w") as f:
            json.dump(params, f, indent=2)

    def _load(self) -> dict:
        if os.path.exists(PARAMS_FILE) and os.path.getsize(PARAMS_FILE) > 0:
            try:
                with open(PARAMS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
