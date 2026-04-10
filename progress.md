# MoneyBot Progress Log

---

## 2026-04-10 — Codespace training setup added

STATUS: complete

BUILT:
  .devcontainer/devcontainer.json  ← Codespace machine config
  .devcontainer/start.sh           ← single command launcher
  scripts/server.py                ← keep-alive server (port 8080)
  scripts/notify.py                ← Gmail email notifier

HOW TO USE:
  1. Push repo to GitHub
  2. Open Codespace (Code → Codespaces → Create)
  3. Wait for machine to provision (~2 min)
  4. In terminal: bash .devcontainer/start.sh
  5. Make port 8080 public in Ports tab
  6. Set up cron-job.org ping on /health every 20 min
  7. Close laptop. Training runs for ~45-60 min.
  8. Receive Gmail notification when done.

GMAIL SETUP (one-time):
  myaccount.google.com → Security → 2-Step Verification →
  App passwords → name it "moneybot" → copy 16-char code
  Then in Codespace terminal:
    export GMAIL_SENDER="your@gmail.com"
    export GMAIL_PASSWORD="xxxx xxxx xxxx xxxx"
    export GMAIL_RECEIVER="your@gmail.com"
  Or add to GitHub → Settings → Codespaces → Secrets (permanent)

TESTED: existing 164 tests unaffected (no existing file modified)
NEXT: push to GitHub, open Codespace, run bash .devcontainer/start.sh

---

## 2026-04-08

### Step 1.1 complete — structure created
Created full directory tree: brain/, brain/signals/, strategies/, scoring/, risk/, execution/, learning/, capital/, data/, reports/daily/, tests/, config/
Created all __init__.py files and empty data files.

### Step 1.2 complete — requirements.txt installed
All packages installed and verified: ccxt, pandas, numpy, pyyaml, schedule, aiohttp, websockets

### Step 1.3 complete — config/config.yaml written
Full config with all strategies, signal weights, risk parameters, pairs, triangular paths.
NOTE: Replaced MATIC/USDT → POL/USDT (delisted orderbook), REEF/USDT → PEPE/USDT (empty ob)

### Step 2 complete — Capital tracker
capital/tracker.py: SQLite persistence, milestones, daily reports, win rate, drawdown, consecutive losses
capital/compounder.py: auto-compounding, project() method
TEST: $50 → 5x $0.50 = $52.50 ✓, restart loads $52.50 ✓

### Step 3 complete — Market data layer
brain/market_reader.py: ccxt Binance, 5s cache, latency tracking, graceful error handling
TEST: SOL/USDT orderbook + candles + trades fetched ✓, avg_latency_ms exposed ✓

### Step 4 complete — 8 signal generators
brain/signals/price_action.py, momentum.py, microstructure.py, volatility.py,
sentiment.py, cvd.py, vwap.py, whale_detector.py
ALL: return -1.0 to +1.0 ✓, complete in <500ms ✓, return 0.0 on missing data ✓
Exposed attributes: atr_ratio, vol_spike, funding_rate, cvd_divergence, whale_buying

### Step 5 complete — Signal combiner
scoring/signal_combiner.py: weighted consensus, confidence, CVD/whale/vol overrides
is_arb_friendly() method for arb strategy gating

### Step 6 complete — Regime detector
brain/regime_detector.py: 8 regimes in priority order, logs to data/regime_log.csv

### Step 7 complete — Strategy router
brain/strategy_router.py: maps regime → strategies + size_mult, low-confidence halving

### Step 8 complete — 7 trading strategies
strategies/base_strategy.py: BaseStrategy + Opportunity dataclass
strategies/triangular_arb.py: 3-leg triangular arb, leg3 direction='sell' ✓
strategies/stat_arb.py: correlation matrix, z-score divergence, position tracking
strategies/funding_arb.py: funding rate arb, delta-neutral
strategies/grid_trader.py: grid setup + level crossing, ATR-sized grids
strategies/mean_reversion.py: drop filter, BTC filter, CVD absorption check
strategies/volume_spike.py: 3x volume spike, CVD + whale confirmation
strategies/correlation_breakout.py: BTC lag detection, Pearson correlation

### Step 9 complete — Opportunity scorer + Pair ranker
scoring/opportunity_scorer.py: 7-filter 0.0–1.0 scoring, logs opportunities
scoring/pair_ranker.py: SQLite-backed Tier A/B/C, win rate, 24h frequency

### Step 10 complete — Risk manager + Position sizer
risk/manager.py: 6 gates, 3 circuit breakers, capital floor, daily loss limit
risk/position_sizer.py: pre-50 flat 2%, Kelly post-50, ATR adjustment, tier cap

### Step 11 complete — Smart executor + Fill simulator + Order manager
execution/fill_simulator.py: 85/15 fill realism, 2x fees, slippage
execution/order_manager.py: open/close/check_exits, persisted to JSON
execution/smart_executor.py: 6 pre-execution checks before paper fill

### Step 12 complete — Multithreaded master engine
brain/master_engine.py: 6 threads (Data/Brain/Scan/Execution/Monitor/Learning)
Threading: capital_lock, brain_queue(maxsize=1), opp_queue(maxsize=50)
Console output on every execution with full signal display

### Step 13 complete — Learning engine
learning/trade_analyzer.py: nightly analysis, regime/signal/strategy/timing/threshold
learning/signal_optimizer.py: weight update ±0.01/day, normalized, bounds [0.03,0.40]
learning/strategy_optimizer.py: parameter nudges ±5%/day max
learning/pair_performance.py: pair P&L tracking

### Step 14 complete — Tests
tests/test_signals.py: all 8 signals, bounds, speed, missing data, exposed attrs ✓
tests/test_risk.py: all 6 gates, CB1 size reduction, tier cap ✓
tests/test_strategies.py: scan() returns list, VOLATILE → [], fields populated ✓

### SYSTEM FULLY BUILT — Ready for 48-hour paper test
Run with: python run.py
Next step: 48-hour paper run (Step 14 from plan.md)

---

## 2026-04-10

### Bayesian upgrade complete — 164/164 tests pass

All files from update.md were created in a prior session:
- signals/group_orderbook.py, group_candles.py, group_trades.py, group_futures.py, group_derived.py
- bayesian/network.py, likelihood_engine.py, correlation_filter.py, threshold_manager.py, prior_loader.py
- training/data_downloader.py, feature_generator.py, label_generator.py, likelihood_trainer.py, correlation_trainer.py
- scripts/train_bayesian.py

3 test failures fixed this session:
1. test_high_atr_ratio_triggers_veto_score — make_volatile_candles redesigned: 40 stable + 20 spike candles
   so current ATR >> avg(last 20) → ratio > 1.8 → score ≤ -0.3
2. test_correlation_filter_reduces_correlated_impact — CorrelationFilter.filter() now applies effective-count
   scaling: adjusted = blended * (effective_count/n); equal correlated signals no longer double-count
3. test_bayesian_lower_probability_with_contradicting_signals — two fixes:
   a. depth_imbalance added to order_flow correlation group (pulls down decorrelated order_flow_imbalance)
   b. BayesianNetwork.compute() now applies penalty (−0.02 per signal < −0.3) for opposing non-primary signals

Result: 164 passed / 0 failed
