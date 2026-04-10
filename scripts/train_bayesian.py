"""
Training entry point — run ONCE before paper trading.
Usage: python scripts/train_bayesian.py
"""
from __future__ import annotations

import os
import sys
import json

# Ensure project root is in path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import yaml

from bayesian.prior_loader import PriorLoader
from training.data_downloader import DataDownloader
from training.feature_generator import FeatureGenerator
from training.label_generator import LabelGenerator
from training.likelihood_trainer import LikelihoodTrainer
from training.correlation_trainer import CorrelationTrainer

HIST_DIR = "data/historical"
FEATURES_PATH = "data/ml_features.csv"
STRATEGIES = [
    "triangular_arb", "stat_arb", "funding_arb", "grid_trader",
    "mean_reversion", "volume_spike", "correlation_breakout",
]


def main():
    print("=" * 60)
    print("  MoneyBot — Bayesian Training Pipeline")
    print("=" * 60)

    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    symbols = config.get("scan_universe", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

    # Step 1 — Download historical data if needed
    if not os.path.exists(HIST_DIR) or not os.listdir(HIST_DIR):
        print("\n[Step 1] Downloading 2 years of historical data…")
        print("  Estimated time: 30-40 minutes")
        DataDownloader().download_all(symbols, HIST_DIR)
    else:
        print(f"\n[Step 1] Historical data already present ({HIST_DIR}), skipping download.")

    # Step 2 — Generate features
    if not os.path.exists(FEATURES_PATH):
        print("\n[Step 2] Generating features from historical data…")
        print("  Estimated time: 10-15 minutes")
        FeatureGenerator().generate(HIST_DIR, FEATURES_PATH)
    else:
        print(f"\n[Step 2] Features already present ({FEATURES_PATH}), skipping.")

    # Step 3 — Generate labels
    import pandas as pd
    features_df = pd.read_csv(FEATURES_PATH)
    print(f"\n[Step 3] Generating labels for {len(features_df)} rows…")
    labeled_dfs = []
    for strategy in STRATEGIES:
        labeled = LabelGenerator().generate(features_df.copy(), strategy)
        labeled["strategy"] = strategy
        labeled_dfs.append(labeled)
    all_labeled = pd.concat(labeled_dfs, ignore_index=True)

    # Step 4 — Train likelihood ratios
    print("\n[Step 4] Training likelihood ratios…")
    ratios = LikelihoodTrainer().train(all_labeled)
    print(f"  Saved → data/likelihood_ratios.json ({len(ratios)} strategies)")

    # Step 5 — Train correlation matrix
    print("\n[Step 5] Building correlation matrix…")
    CorrelationTrainer().train(features_df)

    # Step 6 — Validate outputs
    print("\n[Step 6] Validation:")
    if os.path.exists("data/likelihood_ratios.json"):
        with open("data/likelihood_ratios.json") as f:
            lr = json.load(f)
        for strat, sigs in lr.items():
            if sigs:
                top = sorted(sigs.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  {strat}: top={top}")
    if os.path.exists("data/base_rates.json"):
        with open("data/base_rates.json") as f:
            br = json.load(f)
        print(f"  Base rates: {br}")
    if os.path.exists("data/signal_correlations.json"):
        print("  signal_correlations.json ✓")

    # Step 7 — Save literature priors as backup
    loader = PriorLoader()
    loader.save_literature_priors()
    loader.save_base_rates()
    print("  literature_priors.json ✓")
    print("  base_rates.json ✓")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE — ready to run: python run.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
