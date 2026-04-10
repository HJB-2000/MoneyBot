# MoneyBot — Bayesian Signal Network Upgrade
> Give this entire file to Claude Code.
> Read it completely before writing a single line of code.
> After every file created or modified, append to progress.md.

---

## IMPORTANT — READ BEFORE TOUCHING ANYTHING

This is an UPGRADE, not a rewrite.

DO NOT delete or rewrite any existing file.
DO NOT remove any existing functionality.
DO NOT change any file that is not explicitly listed below.

The only files that change are:
  brain/master_engine.py      ← small targeted update only
  learning/trade_analyzer.py  ← small addition only
  scoring/opportunity_scorer.py ← small targeted update only
  requirements.txt            ← additions only

Everything else is NEW files added alongside existing code.

The existing 117 unit tests MUST still pass after all changes.
Run this before starting: pytest unit_test/ -v
Run this after every task: pytest unit_test/ -v
If any existing test breaks: fix it before continuing.

Old signal files (brain/signals/*.py) are kept as-is.
They are no longer called by master_engine but are not deleted.

---

## What We Are Building

Replace the current 8-signal voting system with a scientifically
grounded Bayesian Signal Network using 30 signals.

The system works in 3 phases:
  Phase 1: Historical training (run once before paper trading)
  Phase 2: Live Bayesian decision making (runs every cycle)
  Phase 3: Weekly retraining on sliding window (automatic)

---

## New File Structure

```
moneyBot/
├── bayesian/
│   ├── __init__.py
│   ├── network.py              ← core Bayesian engine
│   ├── likelihood_engine.py    ← calculates likelihood ratios
│   ├── correlation_filter.py   ← removes correlated signal double-counting
│   ├── threshold_manager.py    ← dynamic threshold per regime
│   └── prior_loader.py         ← loads literature-based starting priors
├── signals/                    ← expand from 8 to 30 signals
│   ├── group_orderbook.py      ← 8 signals from order book data
│   ├── group_candles.py        ← 10 signals from candle data
│   ├── group_trades.py         ← 5 signals from trade data
│   ├── group_futures.py        ← 5 signals from futures data
│   └── group_derived.py        ← 2 derived signals (no extra fetch)
├── training/
│   ├── __init__.py
│   ├── data_downloader.py      ← downloads 2 years Binance history
│   ├── feature_generator.py    ← calculates all 30 signals on history
│   ├── label_generator.py      ← generates trade outcome labels
│   ├── likelihood_trainer.py   ← calculates real likelihood ratios
│   └── correlation_trainer.py  ← builds signal correlation matrix
└── scripts/
    └── train_bayesian.py       ← entry point: run once before live
```

---

## TASK 1 — The 30 Signals (4 Parallel Groups)

### Group 1: group_orderbook.py (8 signals)
Data needed: order book (fetch once per cycle)

```
CLASS: OrderBookSignals
METHOD: calculate(orderbook) -> dict of 8 scores

Signal 1 — bid_ask_spread
  spread_pct = (best_ask - best_bid) / mid_price
  Fuzzy membership:
    < 0.0005: score = +0.8
    0.0005-0.002: score = +0.2
    > 0.002: score = -0.6
    > 0.005: score = -1.0

Signal 2 — depth_imbalance
  total_bids = sum of top 10 bid sizes
  total_asks = sum of top 10 ask sizes
  ratio = total_bids / (total_bids + total_asks)
  0.45-0.55 (balanced): score = +0.5
  > 0.65 (bid heavy): score = +0.3
  < 0.35 (ask heavy): score = -0.3
  Extreme (>0.8 or <0.2): score = -0.5

Signal 3 — order_flow_imbalance
  OFI = (buy_volume - sell_volume) / total_volume
  Near 0 (±0.1): score = +0.5 (best for arb)
  > 0.4: score = +0.2 (directional)
  < -0.4: score = -0.2 (directional)

Signal 4 — large_order_presence
  Any single order > 3% of total depth = large order present
  Large bid present: score = +0.3
  Large ask present: score = -0.3
  None: score = 0.0

Signal 5 — iceberg_detection
  Same price level refilled 3+ times in last 60s = iceberg
  Iceberg on bid: score = +0.4
  Iceberg on ask: score = -0.4
  None detected: score = 0.0

Signal 6 — spoofing_detection
  Large order appears then disappears within 2s = spoof
  Spoof on bid (fake support): score = -0.3
  Spoof on ask (fake resistance): score = +0.3
  None: score = 0.0

Signal 7 — liquidity_score
  available_liquidity / our_trade_size
  > 50x: score = +0.8
  20-50x: score = +0.5
  10-20x: score = +0.2
  5-10x: score = -0.1
  < 5x: score = -1.0 (hard reject)

Signal 8 — book_pressure_ratio
  Compare bid wall strength vs ask wall strength
  Sum of large orders (>$10k) on each side
  Strong bid walls vs weak ask: score = +0.4
  Strong ask walls vs weak bid: score = -0.4
  Balanced: score = 0.0
```

### Group 2: group_candles.py (10 signals)
Data needed: 100 candles at 5m (fetch once per cycle)

```
CLASS: CandleSignals
METHOD: calculate(candles) -> dict of 10 scores

All implementations from scratch using pandas + numpy only.
No TA-Lib or external indicator libraries.

Signal 9 — RSI (period=14)
  Wilder RSI implementation
  < 25: score = +0.9 (strongly oversold)
  25-35: score = +0.5
  35-45: score = +0.2
  45-55: score = 0.0 (neutral, good for arb)
  55-65: score = -0.2
  65-75: score = -0.5
  > 75: score = -0.9 (strongly overbought)

Signal 10 — MACD (12/26/9)
  MACD line, signal line, histogram
  Histogram positive and growing: score = +0.5
  Histogram positive and shrinking: score = +0.1
  Just crossed above signal: score = +0.7
  Just crossed below signal: score = -0.7
  Histogram negative and growing: score = -0.5
  Flat near zero: score = +0.2 (best for arb)

Signal 11 — EMA Cross (9/21)
  Fast EMA above slow, widening gap: score = +0.4
  Fast EMA above slow, narrowing: score = +0.1
  Fast EMA below slow, widening: score = -0.4
  Fast EMA below slow, narrowing: score = -0.1
  Just crossed: score = ±0.6

Signal 12 — ATR Ratio
  current_atr / rolling_20_period_avg_atr
  < 0.5: score = +0.8 (very calm)
  0.5-0.8: score = +0.4
  0.8-1.2: score = +0.1 (normal)
  1.2-1.8: score = -0.3
  1.8-2.0: score = -0.7
  > 2.0: score = -1.0 (VETO trigger)
  EXPOSE: atr_ratio as attribute

Signal 13 — Bollinger Band State
  bandwidth = (upper - lower) / middle
  Squeezing (bandwidth < 50% of 20p avg): score = -0.2 (breakout coming)
  Wide bands (bandwidth > 150% of avg): score = -0.4 (too volatile)
  Price near lower band: score = +0.5
  Price near upper band: score = -0.4
  Price at middle: score = +0.2
  EXPOSE: is_squeezing as bool

Signal 14 — Rate of Change (10 period)
  ROC = (close - close_10ago) / close_10ago
  Abs(ROC) > 0.03: score = -0.7 (too fast)
  Abs(ROC) 0.01-0.03: score = -0.2
  Abs(ROC) < 0.005: score = +0.5 (stable, good for arb)

Signal 15 — Support/Resistance Score
  Find levels touched 3+ times in last 200 candles
  Price near center of range: score = +0.4
  Price approaching strong resistance: score = -0.3
  Price bouncing off strong support: score = +0.5
  Price just broke S/R with volume: score = -0.2 (breakout)
  EXPOSE: breakout_detected as bool

Signal 16 — Trend Structure
  Analyze last 5 swing highs and lows
  Clear HH + HL (uptrend): score = +0.5
  Clear LH + LL (downtrend): score = -0.5
  Mixed (ranging): score = +0.3 (good for arb)
  EXPOSE: trend as str

Signal 17 — Consolidation Score
  high_low_range of last 20 candles / mid_price
  < 0.003 (very tight): score = +0.8
  0.003-0.008: score = +0.3
  0.008-0.02: score = 0.0
  > 0.02: score = -0.4
  EXPOSE: consolidating as bool

Signal 18 — Realized Volatility (12 period)
  Std dev of last 12 returns vs 24h average
  Ratio < 0.5: score = +0.6
  Ratio 0.5-1.0: score = +0.2
  Ratio 1.0-2.0: score = -0.2
  Ratio 2.0-3.0: score = -0.6
  Ratio > 3.0: score = -1.0 (VETO trigger)
  EXPOSE: vol_spike as bool
  EXPOSE: realized_vol_ratio as float
```

### Group 3: group_trades.py (5 signals)
Data needed: last 200 trades (fetch once per cycle)

```
CLASS: TradeSignals
METHOD: calculate(trades, candles) -> dict of 5 scores
NOTE: candles required for CVD calculation

Signal 19 — CVD Divergence
  cvd = cumulative sum of (buy_vol - sell_vol) per candle
  Price falling + CVD rising: score = +0.9 (bullish divergence)
  Price rising + CVD falling: score = -0.9 (bearish divergence)
  Both moving same direction: score = ±0.3 (confirmation)
  CVD flat: score = 0.0
  EXPOSE: cvd_divergence as bool
  EXPOSE: cvd_direction as str

Signal 20 — CVD Trend
  Is CVD trending up or down over last 12 periods?
  Strong uptrend: score = +0.5
  Strong downtrend: score = -0.5
  Flat: score = +0.1 (neutral, ok for arb)

Signal 21 — Volume Spike Detection
  current_5min_volume vs rolling_24h_avg_5min_volume
  ratio = current / avg
  > 5x: score = -0.3 (too extreme, caution)
  3-5x with price move: score = +0.4 (valid breakout)
  1.5-3x: score = +0.1
  < 1x: score = +0.3 (calm, good for arb)
  EXPOSE: volume_spike as bool

Signal 22 — Buy/Sell Ratio
  Classify each trade as buy (hit ask) or sell (hit bid)
  buy_count / total_count
  0.45-0.55: score = +0.5 (balanced, best for arb)
  0.55-0.65: score = +0.2 (slight buy pressure)
  0.65-0.80: score = 0.0 (trending)
  > 0.80: score = -0.2 (extreme, unsustainable)
  Mirror for sell side

Signal 23 — Large Trade Flow
  Classify trades by size: small<$1k, medium$1k-50k, large>$50k
  Net flow of large trades:
  Large buyers dominant: score = +0.4
  Large sellers dominant: score = -0.4
  Mixed or absent: score = 0.0
  EXPOSE: whale_buying as bool
  EXPOSE: whale_selling as bool
```

### Group 4: group_futures.py (5 signals)
Data needed: futures API (fetch once per cycle, cache 5 min)

```
CLASS: FuturesSignals
METHOD: calculate(funding, oi, liquidations, candles) -> dict of 5 scores

If futures API unavailable: return all scores as 0.0 gracefully

Signal 24 — Funding Rate
  > 0.001 (0.1% per 8h): score = +0.3 (market very bullish)
  0.0005-0.001: score = +0.1
  -0.0002 to 0.0002: score = 0.0 (neutral)
  -0.0005 to -0.0002: score = -0.1
  < -0.0005: score = -0.3 (market very bearish)
  EXPOSE: funding_rate as float

Signal 25 — Open Interest Change
  oi_change_pct = (oi_now - oi_1h_ago) / oi_1h_ago
  Rising OI + rising price: score = -0.1 (trend strengthening)
  Rising OI + falling price: score = -0.3 (shorts building)
  Falling OI: score = +0.2 (positions closing, ranging likely)
  Stable OI: score = +0.1

Signal 26 — Liquidation Pressure
  Large long liquidations cascade: score = -0.8
  Large short liquidations cascade: score = +0.3 (short squeeze)
  Small liquidations: score = 0.0
  No liquidations: score = +0.2
  EXPOSE: liquidation_cascade as bool

Signal 27 — VWAP Position
  vwap = sum(typical_price * volume) / sum(volume) since midnight
  distance_pct = (price - vwap) / vwap
  Price far above VWAP (>0.5%): score = -0.4
  Price slightly above VWAP (0.1-0.5%): score = +0.1
  Price at VWAP (±0.1%): score = +0.3 (neutral, good for arb)
  Price slightly below VWAP (-0.5 to -0.1%): score = +0.2
  Price far below VWAP (<-0.5%): score = +0.4
  EXPOSE: vwap_price as float
  EXPOSE: distance_from_vwap_pct as float

Signal 28 — VWAP Reclaim
  Price crossed above VWAP with volume > 1.5x average: score = +0.7
  Price crossed below VWAP with volume > 1.5x average: score = -0.7
  No recent cross: score = 0.0
```

### Group 5: group_derived.py (2 signals)
No extra data needed — calculated from cached results above

```
CLASS: DerivedSignals
METHOD: calculate(candle_signals, trade_signals) -> dict of 2 scores

Signal 29 — BTC Correlation Momentum
  If this pair has correlation > 0.80 with BTC:
    Get BTC momentum score from candle_signals
    btc_roc = BTC rate of change last 5 minutes
    If BTC moved > 2% and this pair has not: score = btc_direction * 0.6
    If BTC moved > 2% and this pair already followed: score = 0.0
    If BTC flat: score = 0.0

Signal 30 — Regime Persistence
  How many consecutive cycles has the regime been the same?
  0-3 cycles: score = 0.0 (regime just changed, uncertain)
  3-10 cycles: score = +0.2 (regime stable, trust it)
  10-30 cycles: score = +0.4 (regime very stable)
  > 30 cycles: score = +0.1 (may be about to change)
```

---

## TASK 2 — The Bayesian Network

### bayesian/network.py

```
CLASS: BayesianNetwork

PURPOSE:
  Takes 30 signal scores + strategy name
  Returns P(trade_wins) as float 0.0-1.0
  Uses real likelihood ratios from historical training
  Accounts for signal correlations to avoid double counting

ATTRIBUTES:
  likelihood_ratios: dict loaded from data/likelihood_ratios.json
  correlation_matrix: dict loaded from data/signal_correlations.json
  base_rates: dict loaded from data/base_rates.json
    {regime: historical_win_rate_in_that_regime}

METHOD: compute(signals_dict, strategy, regime) -> float

ALGORITHM:

Step 1 — Load base rate (prior probability)
  prior = base_rates.get(regime, 0.50)
  This is the starting probability before any signals

Step 2 — Decorrelate signals
  Call correlation_filter.filter(signals_dict)
  Returns adjusted signal scores that account for correlation
  Prevents double-counting when two signals measure same thing

Step 3 — Get strategy-relevant signals
  STRATEGY_PRIMARY_SIGNALS = {
    'triangular_arb':       ['bid_ask_spread','liquidity_score',
                             'atr_ratio','order_flow_imbalance'],
    'stat_arb':             ['RSI','MACD','cvd_divergence',
                             'depth_imbalance'],
    'funding_arb':          ['funding_rate','open_interest_change'],
    'grid_trader':          ['consolidation_score','atr_ratio',
                             'bollinger_state','depth_imbalance'],
    'mean_reversion':       ['RSI','cvd_divergence','volume_spike',
                             'support_resistance'],
    'volume_spike':         ['volume_spike_signal','cvd_trend',
                             'buy_sell_ratio'],
    'correlation_breakout': ['btc_correlation_momentum',
                             'EMA_cross','rate_of_change'],
  }
  Only primary signals update the probability for their strategy

Step 4 — Bayesian update loop
  odds = prior / (1 - prior)  ← convert to odds for calculation

  For each primary signal of this strategy:
    lr = likelihood_ratios[strategy][signal_name]
    adjusted_score = decorrelated_scores[signal_name]

    If adjusted_score > 0 (signal positive):
      signal_lr = 1 + (lr - 1) * adjusted_score
      (scale the likelihood ratio by signal strength)
    Else:
      signal_lr = 1 / (1 + (lr - 1) * abs(adjusted_score))
      (inverse for negative signals)

    odds = odds * signal_lr  ← update odds

  P = odds / (1 + odds)  ← convert back to probability

Step 5 — Apply boost from supporting non-primary signals
  For each non-primary signal that is positive:
    boost = 0.02 per signal (small, capped at +0.10 total)
  P = min(P + boost, 0.95)

Step 6 — Apply veto layer
  VETO CONDITIONS (any = return 0.0 immediately):
    signals['atr_ratio'] > 2.0
    signals['realized_vol_ratio'] > 3.0
    signals['liquidation_cascade'] == True
    signals['whale_selling'] == True AND strategy direction is long
    signals['whale_buying'] == True AND strategy direction is short
    regime == 'VOLATILE'

Step 7 — Return final probability
  return round(P, 4)

METHOD: explain(signals_dict, strategy, regime) -> dict
  Returns human-readable explanation of decision:
  {
    'prior': 0.52,
    'final_probability': 0.74,
    'key_signals': ['RSI strongly oversold (+0.9)',
                    'CVD absorbing (+0.7)'],
    'vetoes_checked': 'none triggered',
    'decision': 'EXECUTE'
  }
  Log this explanation to data/decision_log.csv
```

### bayesian/likelihood_engine.py

```
CLASS: LikelihoodEngine

PURPOSE:
  Calculates and updates likelihood ratios from trade history

METHOD: calculate_from_history(trade_log_path, feature_log_path) -> dict

  For each completed trade in trade history:
    Look up what all 30 signal scores were at trade entry time
    Record: signal_score, strategy, result (WIN/LOSS)

  For each signal × strategy combination:
    wins_with_positive_signal = count where signal > 0 AND result = WIN
    losses_with_positive_signal = count where signal > 0 AND result = LOSS
    P_signal_given_win = wins_with_positive_signal / total_wins
    P_signal_given_loss = losses_with_positive_signal / total_losses
    likelihood_ratio = P_signal_given_win / P_signal_given_loss

  Minimum data requirement: 30 trades per strategy before ratio is trusted
  If < 30 trades: use literature_priors.json instead

METHOD: update_weekly(new_trades) -> dict
  Sliding window: use only last 60 days of trades
  Recalculate likelihood ratios
  Save to data/likelihood_ratios.json
  Log changes to data/learning_log.csv

LITERATURE PRIORS (from quantitative research):
  Store in data/literature_priors.json
  These are starting values before we have real trade history

  Based on published crypto trading research:
  {
    "RSI": {"mean_reversion": 1.8, "triangular_arb": 1.1},
    "CVD_divergence": {"mean_reversion": 2.3, "stat_arb": 1.9},
    "bid_ask_spread": {"triangular_arb": 2.8, "grid_trader": 2.1},
    "volume_spike": {"volume_spike": 1.6, "correlation_breakout": 1.4},
    "funding_rate": {"funding_arb": 3.2},
    "atr_ratio": {"all": 0.4},
    "liquidation_cascade": {"all": 0.1}
  }
```

### bayesian/correlation_filter.py

```
CLASS: CorrelationFilter

PURPOSE:
  Prevents double-counting when correlated signals
  fire together

METHOD: filter(signals_dict) -> adjusted_signals_dict

  Load correlation_matrix from data/signal_correlations.json

  KNOWN CORRELATION GROUPS (signals measuring similar things):
  Group A (price momentum): RSI, MACD, EMA_cross, rate_of_change
  Group B (volatility): ATR_ratio, bollinger_state, realized_vol
  Group C (order flow): OFI, buy_sell_ratio, CVD_trend
  Group D (institutional): large_trade_flow, iceberg, book_pressure

  For each group:
    Calculate group_score = weighted average of member scores
    Replace individual scores with:
      group_representative = group_score
      other members = group_score * (1 - correlation_with_representative)

  Result: correlated signals contribute proportionally
  not independently

METHOD: build_matrix(historical_signals) -> dict
  Calculate Pearson correlation between every pair of signals
  over historical data
  Save to data/signal_correlations.json
  Run during training phase only
```

### bayesian/threshold_manager.py

```
CLASS: ThresholdManager

PURPOSE:
  Dynamic execution threshold that adapts to conditions

METHOD: get_threshold(regime, capital, win_rate, consecutive_losses) -> float

  base = 0.65

  REGIME ADJUSTMENT:
    RANGING:       -0.05 (ideal conditions, easier)
    TRENDING_UP:   +0.03
    TRENDING_DOWN: +0.03
    BREAKOUT:      +0.05
    VOLATILE:      +0.99 (effectively blocks all trades)
    FUNDING_RICH:  -0.03
    CHOPPY:        +0.08
    WHALE_MOVING:  +0.05

  PERFORMANCE ADJUSTMENT:
    win_rate > 0.70 (last 20): -0.03 (system working, be more active)
    win_rate 0.60-0.70:         0.00
    win_rate 0.50-0.60:        +0.04
    win_rate < 0.50:           +0.08 (struggling, be selective)

  CAPITAL ADJUSTMENT:
    capital < $100:   0.00 (need activity to grow)
    capital $100-500: -0.02
    capital > $500:   -0.04 (can afford to be more active)

  CONSECUTIVE LOSS ADJUSTMENT:
    0-2 losses:  0.00
    3-4 losses: +0.05
    5+ losses:  +0.10

  final = base + regime_adj + performance_adj + capital_adj + loss_adj
  return max(0.55, min(0.85, final))
```

---

## TASK 3 — Historical Training System

### training/data_downloader.py

```
CLASS: DataDownloader

PURPOSE:
  Downloads 2 years of historical data from Binance
  Completely free via ccxt public API (no API key needed)

METHOD: download_all(symbols, output_dir) -> None

  For each symbol in config scan_universe:
    Download 5-minute candles for last 2 years:
      end_time = now
      start_time = now - (2 * 365 * 24 * 60 * 60 * 1000)  ← ms

    Use ccxt pagination to get all candles:
      while start_time < end_time:
        batch = exchange.fetch_ohlcv(symbol, '5m',
                                     since=start_time,
                                     limit=1000)
        save batch to data/historical/{symbol_safe}.csv
        start_time = batch[-1][0] + 1  ← next batch

    Also download trade data for CVD calculation:
      Save to data/historical/{symbol_safe}_trades.csv

  Progress bar showing download status
  Estimated time: 20-40 minutes for all pairs
  Total data size: approximately 2-4 GB

  Handle rate limits: sleep 0.5s between requests
  Handle errors: retry 3 times, skip if fails after 3
  Log progress to data/training_log.csv
```

### training/feature_generator.py

```
CLASS: FeatureGenerator

PURPOSE:
  Runs all 30 signals on historical data
  Generates feature matrix for training

METHOD: generate(historical_dir, output_path) -> DataFrame

  For each symbol and each 5-minute window in history:
    Load orderbook snapshot (approximated from candle data)
    Load candle data (100 candle lookback)
    Load trade data

    Run all 30 signals:
      ob_signals = OrderBookSignals().calculate_from_candles(candles)
      c_signals  = CandleSignals().calculate(candles)
      t_signals  = TradeSignals().calculate(trades)
      f_signals  = FuturesSignals().calculate_from_candles(candles)
      d_signals  = DerivedSignals().calculate(c_signals, t_signals)

    Combine into feature row:
      {timestamp, symbol, signal_1, signal_2, ..., signal_30}

  Save to data/ml_features.parquet
  Log: total rows generated, time taken
```

### training/label_generator.py

```
CLASS: LabelGenerator

PURPOSE:
  For each historical moment, determine if a trade
  would have been profitable

METHOD: generate(features_df, strategy) -> DataFrame with labels

  For each row in features:
    Look ahead N candles (strategy dependent):
      triangular_arb:       5 candles (25 minutes)
      stat_arb:             48 candles (4 hours)
      mean_reversion:       12 candles (60 minutes)
      volume_spike:         6 candles (30 minutes)
      correlation_breakout: 4 candles (20 minutes)
      grid_trader:          96 candles (8 hours)

    Calculate if price moved favorably:
      For long strategies: price_future > price_entry * (1 + min_profit)
      min_profit = 0.001 (0.1% after fees)

    Label: 1 = would have won, 0 = would have lost

  Add label column to features DataFrame
  Log: win rate in historical data per strategy
```

### training/likelihood_trainer.py

```
CLASS: LikelihoodTrainer

PURPOSE:
  Calculates real likelihood ratios from labeled historical data

METHOD: train(features_with_labels, output_path) -> dict

  For each strategy:
    Filter rows relevant to that strategy
    Separate into wins and losses

    For each of the 30 signals:
      P_positive_given_win = 
        count(signal > 0 AND label = 1) / count(label = 1)
      P_positive_given_loss = 
        count(signal > 0 AND label = 0) / count(label = 0)
      
      lr = P_positive_given_win / max(P_positive_given_loss, 0.01)
      (avoid division by zero)

      Store: likelihood_ratios[strategy][signal] = lr

    Also calculate base_rate:
      base_rates[strategy][regime] = 
        count(wins in regime) / count(all in regime)

  Validation:
    Print top 5 most predictive signals per strategy
    Print any signal with lr < 0.5 (negative predictor)
    These are worth reviewing

  Save to data/likelihood_ratios.json
  Save base_rates to data/base_rates.json
  Log summary to data/training_log.csv
```

### training/correlation_trainer.py

```
CLASS: CorrelationTrainer

PURPOSE:
  Builds signal correlation matrix to prevent double-counting

METHOD: train(features_df, output_path) -> dict

  Extract signal columns from features DataFrame
  Calculate Pearson correlation matrix for all 30 signals

  For each pair (signal_i, signal_j):
    if abs(correlation) > 0.70:
      flag as correlated pair
      Add to correlation groups

  Print: correlation heatmap summary in text format
  Print: top 10 most correlated pairs

  Save to data/signal_correlations.json
  Format: {signal_name: {other_signal: correlation_value}}
```

### scripts/train_bayesian.py

```
PURPOSE:
  Single entry point to run complete training pipeline
  Run this ONCE before starting paper trading

USAGE:
  python scripts/train_bayesian.py

STEPS:
  1. Check if data/historical/ exists and has data
     If not: run DataDownloader for all pairs
     Estimated time: 30-40 minutes

  2. Run FeatureGenerator on historical data
     Estimated time: 10-15 minutes

  3. Run LabelGenerator for all strategies
     Estimated time: 2-3 minutes

  4. Run LikelihoodTrainer
     Estimated time: 1-2 minutes
     Output: data/likelihood_ratios.json

  5. Run CorrelationTrainer
     Estimated time: 1 minute
     Output: data/signal_correlations.json

  6. Validate outputs:
     Print: likelihood ratios for top 5 signals per strategy
     Print: most correlated signal pairs
     Print: base win rates per regime
     Print: TRAINING COMPLETE — ready to run bot

  7. Copy literature_priors.json to data/ as backup
     Bot falls back to these if real ratios unavailable

Total estimated time: 45-60 minutes first run
Subsequent runs (weekly update): 5-10 minutes
```

---

## TASK 4 — Integration into Master Engine

### Modify brain/master_engine.py

```
Replace the old signal scoring system with BayesianNetwork.

In BrainThread:
  OLD approach:
    Run 8 signals → average scores → compare to threshold

  NEW approach:
    1. Fetch 4 data sources in parallel (unchanged)

    2. Calculate 5 signal groups in parallel:
       with ThreadPoolExecutor(max_workers=5) as executor:
         f1 = executor.submit(OrderBookSignals().calculate, orderbook)
         f2 = executor.submit(CandleSignals().calculate, candles)
         f3 = executor.submit(TradeSignals().calculate, trades)
         f4 = executor.submit(FuturesSignals().calculate, funding, oi, liq, candles)
         ob, c, t, f = f1.result(), f2.result(), f3.result(), f4.result()
       d = DerivedSignals().calculate(c, t)
       all_signals = {**ob, **c, **t, **f, **d}  ← 30 signals

    3. Get regime (unchanged, uses all_signals)

    4. For each opportunity found by strategies:
       P = bayesian_network.compute(all_signals, opportunity.strategy, regime)
       threshold = threshold_manager.get_threshold(regime, capital, win_rate, losses)
       if P > threshold:
         opportunity.score = P
         qualified.append(opportunity)

    5. Execute best qualified opportunity (unchanged)

    6. Log all_signals snapshot to data/signal_log.csv
       (needed for weekly retraining)

    NOTE: Add to config.yaml under logging section:
      signal_log: data/signal_log.csv
```

---

## TASK 5 — Weekly Retraining Integration

### Modify learning/trade_analyzer.py

```
Add to daily_analysis() that runs at midnight:

  Every Sunday (day_of_week == 6):
    from training.likelihood_trainer import LikelihoodTrainer
    from training.correlation_trainer import CorrelationTrainer

    trainer = LikelihoodTrainer()
    new_ratios = trainer.train_sliding_window(
      trade_log='data/trade_log.csv',
      signal_log='data/signal_log.csv',
      lookback_days=60
    )

    Validation check:
      If new_ratios based on < 30 trades per strategy:
        Keep existing ratios, log warning
      If new_ratios based on >= 30 trades:
        Save to data/likelihood_ratios.json
        BayesianNetwork reloads on next cycle

    Log to data/learning_log.csv:
      Date, trades_used, ratios_updated, top_predictive_signals
```

---

## TASK 6 — Unit Tests

Create unit_test/test_bayesian.py and unit_test/test_signals_30.py

### unit_test/test_signals_30.py

```python
# ── Order Book Signals (8) ──────────────────────────────────

def test_bid_ask_spread_tight_positive():
    ob = make_orderbook(best_bid=100.0, best_ask=100.04)
    scores = OrderBookSignals().calculate(ob)
    assert scores['bid_ask_spread'] > 0.5

def test_bid_ask_spread_wide_negative():
    ob = make_orderbook(best_bid=100.0, best_ask=100.6)
    scores = OrderBookSignals().calculate(ob)
    assert scores['bid_ask_spread'] < 0.0

def test_depth_imbalance_balanced_positive():
    ob = make_balanced_orderbook()
    scores = OrderBookSignals().calculate(ob)
    assert scores['depth_imbalance'] > 0.3

def test_liquidity_score_hard_reject_below_5x():
    ob = make_orderbook(depth=4.0)
    scores = OrderBookSignals().calculate(ob)
    assert scores['liquidity_score'] == -1.0

def test_iceberg_detected_on_refill():
    ob = make_refilling_orderbook()
    scores = OrderBookSignals().calculate(ob)
    assert scores['iceberg_detection'] != 0.0

def test_spoofing_detected_on_disappearing_order():
    ob = make_spoofing_orderbook()
    scores = OrderBookSignals().calculate(ob)
    assert scores['spoofing_detection'] != 0.0

def test_all_orderbook_scores_in_range():
    ob = make_orderbook()
    scores = OrderBookSignals().calculate(ob)
    for k, v in scores.items():
        assert -1.0 <= v <= 1.0, f"{k} out of range: {v}"

def test_orderbook_completes_under_100ms():
    import time
    ob = make_orderbook()
    t = time.time()
    OrderBookSignals().calculate(ob)
    assert time.time() - t < 0.1

# ── Candle Signals (10) ──────────────────────────────────────

def test_rsi_oversold_strong_positive():
    candles = make_declining_candles(periods=20)
    scores = CandleSignals().calculate(candles)
    assert scores['RSI'] > 0.4

def test_rsi_overbought_strong_negative():
    candles = make_rising_candles(periods=20)
    scores = CandleSignals().calculate(candles)
    assert scores['RSI'] < -0.4

def test_atr_ratio_exposed_and_valid():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert hasattr(sig, 'atr_ratio')
    assert sig.atr_ratio > 0

def test_high_atr_ratio_triggers_veto_score():
    candles = make_volatile_candles()
    scores = CandleSignals().calculate(candles)
    assert scores['ATR_ratio'] <= -0.7

def test_consolidation_detected_on_flat_data():
    candles = make_flat_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert sig.consolidating == True

def test_bollinger_squeeze_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.is_squeezing, bool)

def test_breakout_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.breakout_detected, bool)

def test_vol_spike_exposed_as_bool():
    candles = make_candles()
    sig = CandleSignals()
    sig.calculate(candles)
    assert isinstance(sig.vol_spike, bool)

def test_all_candle_scores_in_range():
    candles = make_candles()
    scores = CandleSignals().calculate(candles)
    for k, v in scores.items():
        assert -1.0 <= v <= 1.0, f"{k} out of range: {v}"

def test_candle_signals_insufficient_data_returns_zeros():
    candles = make_candles(n=5)
    scores = CandleSignals().calculate(candles)
    assert all(v == 0.0 for v in scores.values())

# ── Trade Signals (5) ────────────────────────────────────────

def test_cvd_bullish_divergence_strong_positive():
    candles = make_declining_candles()
    trades = make_buy_trades()
    scores = TradeSignals().calculate(trades, candles)
    assert scores['CVD_divergence'] > 0.7

def test_cvd_bearish_divergence_strong_negative():
    candles = make_rising_candles()
    trades = make_sell_trades()
    scores = TradeSignals().calculate(trades, candles)
    assert scores['CVD_divergence'] < -0.7

def test_whale_buying_exposed_as_bool():
    trades = make_large_buy_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert sig.whale_buying == True

def test_whale_selling_exposed_as_bool():
    trades = make_large_sell_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert sig.whale_selling == True

def test_volume_spike_exposed_as_bool():
    trades = make_trades()
    sig = TradeSignals()
    sig.calculate(trades, make_candles())
    assert isinstance(sig.volume_spike, bool)

# ── Futures Signals (5) ──────────────────────────────────────

def test_positive_funding_gives_positive_score():
    scores = FuturesSignals().calculate(0.001, 1000, [], make_candles())
    assert scores['funding_rate'] > 0.0

def test_negative_funding_gives_negative_score():
    scores = FuturesSignals().calculate(-0.001, 1000, [], make_candles())
    assert scores['funding_rate'] < 0.0

def test_liquidation_cascade_exposed_as_bool():
    sig = FuturesSignals()
    sig.calculate(0.0, 1000, make_liquidations(large=True), make_candles())
    assert sig.liquidation_cascade == True

def test_futures_unavailable_returns_zeros():
    scores = FuturesSignals().calculate(None, None, None, make_candles())
    assert all(v == 0.0 for v in scores.values())

def test_vwap_price_exposed_and_positive():
    sig = FuturesSignals()
    sig.calculate(0.0, 1000, [], make_candles())
    assert sig.vwap_price > 0.0

# ── Derived Signals (2) ──────────────────────────────────────

def test_btc_correlation_score_in_range():
    c_signals = {'rate_of_change': 0.03}
    t_signals = {}
    scores = DerivedSignals().calculate(c_signals, t_signals)
    assert -1.0 <= scores['btc_correlation_momentum'] <= 1.0

def test_regime_persistence_increases_with_cycles():
    sig = DerivedSignals()
    sig.calculate({}, {}, regime_cycles=15)
    scores_15 = sig.scores['regime_persistence']
    sig.calculate({}, {}, regime_cycles=3)
    scores_3 = sig.scores['regime_persistence']
    assert scores_15 > scores_3
```

### unit_test/test_bayesian.py

```python
def test_bayesian_returns_probability_in_range():
    network = BayesianNetwork()
    signals = make_all_signals()
    P = network.compute(signals, 'triangular_arb', 'RANGING')
    assert 0.0 <= P <= 1.0

def test_bayesian_veto_on_volatile_regime():
    network = BayesianNetwork()
    signals = make_all_signals()
    P = network.compute(signals, 'triangular_arb', 'VOLATILE')
    assert P == 0.0

def test_bayesian_veto_on_high_atr():
    network = BayesianNetwork()
    signals = make_all_signals()
    signals['ATR_ratio'] = 2.5
    P = network.compute(signals, 'triangular_arb', 'RANGING')
    assert P == 0.0

def test_bayesian_veto_on_liquidation_cascade():
    network = BayesianNetwork()
    signals = make_all_signals()
    signals['liquidation_cascade'] = True
    P = network.compute(signals, 'mean_reversion', 'RANGING')
    assert P == 0.0

def test_bayesian_higher_probability_with_strong_signals():
    network = BayesianNetwork()
    weak_signals = make_neutral_signals()
    strong_signals = make_strong_positive_signals()
    P_weak = network.compute(weak_signals, 'triangular_arb', 'RANGING')
    P_strong = network.compute(strong_signals, 'triangular_arb', 'RANGING')
    assert P_strong > P_weak

def test_bayesian_lower_probability_with_contradicting_signals():
    network = BayesianNetwork()
    positive = make_strong_positive_signals()
    contradicted = positive.copy()
    contradicted['CVD_divergence'] = -0.9
    contradicted['depth_imbalance'] = -0.7
    P_pos = network.compute(positive, 'triangular_arb', 'RANGING')
    P_con = network.compute(contradicted, 'triangular_arb', 'RANGING')
    assert P_pos > P_con

def test_bayesian_uses_regime_base_rate():
    network = BayesianNetwork()
    signals = make_neutral_signals()
    P_ranging = network.compute(signals, 'triangular_arb', 'RANGING')
    P_volatile = network.compute(signals, 'triangular_arb', 'VOLATILE')
    assert P_ranging > P_volatile

def test_bayesian_explain_returns_dict():
    network = BayesianNetwork()
    signals = make_all_signals()
    explanation = network.explain(signals, 'triangular_arb', 'RANGING')
    assert 'prior' in explanation
    assert 'final_probability' in explanation
    assert 'decision' in explanation

def test_correlation_filter_reduces_correlated_impact():
    cf = CorrelationFilter()
    signals = {'RSI': 0.8, 'MACD': 0.8, 'EMA_cross': 0.8, 'rate_of_change': 0.8}
    filtered = cf.filter(signals)
    total_original = sum(signals.values())
    total_filtered = sum(filtered.values())
    assert total_filtered < total_original

def test_threshold_manager_volatile_blocks_all():
    tm = ThresholdManager()
    threshold = tm.get_threshold('VOLATILE', 100, 0.6, 0)
    assert threshold > 0.95

def test_threshold_manager_ranging_lowest():
    tm = ThresholdManager()
    t_ranging = tm.get_threshold('RANGING', 100, 0.6, 0)
    t_choppy = tm.get_threshold('CHOPPY', 100, 0.6, 0)
    assert t_ranging < t_choppy

def test_threshold_manager_bad_win_rate_raises():
    tm = ThresholdManager()
    t_good = tm.get_threshold('RANGING', 100, 0.70, 0)
    t_bad = tm.get_threshold('RANGING', 100, 0.45, 0)
    assert t_bad > t_good

def test_threshold_manager_consecutive_losses_raises():
    tm = ThresholdManager()
    t_0 = tm.get_threshold('RANGING', 100, 0.6, 0)
    t_5 = tm.get_threshold('RANGING', 100, 0.6, 5)
    assert t_5 > t_0

def test_threshold_always_in_bounds():
    tm = ThresholdManager()
    for regime in ['RANGING','VOLATILE','CHOPPY','TRENDING_UP']:
        for win_rate in [0.3, 0.5, 0.7, 0.9]:
            for losses in [0, 3, 7]:
                t = tm.get_threshold(regime, 100, win_rate, losses)
                assert 0.55 <= t <= 0.85

def test_likelihood_engine_calculates_from_history():
    engine = LikelihoodEngine()
    mock_trades = make_mock_trade_history(n=100)
    mock_signals = make_mock_signal_history(n=100)
    ratios = engine.calculate_from_history(mock_trades, mock_signals)
    assert 'triangular_arb' in ratios
    assert all(v > 0 for v in ratios['triangular_arb'].values())

def test_likelihood_engine_falls_back_to_priors():
    engine = LikelihoodEngine()
    mock_trades = make_mock_trade_history(n=5)
    mock_signals = make_mock_signal_history(n=5)
    ratios = engine.calculate_from_history(mock_trades, mock_signals)
    assert ratios == engine.load_literature_priors()

def test_parallel_signal_calculation_under_500ms():
    import time
    from concurrent.futures import ThreadPoolExecutor
    ob = make_orderbook()
    candles = make_candles()
    trades = make_trades()
    t = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        f1 = ex.submit(OrderBookSignals().calculate, ob)
        f2 = ex.submit(CandleSignals().calculate, candles)
        f3 = ex.submit(TradeSignals().calculate, trades, candles)
        f4 = ex.submit(FuturesSignals().calculate, 0.0, 1000, [], candles)
        results = [f.result() for f in [f1,f2,f3,f4]]
    elapsed = time.time() - t
    assert elapsed < 0.5
    assert all(isinstance(r, dict) for r in results)
```

---

## TASK 7 — Update progress.md After Every File

After creating each file, append to progress.md:

```
## [DATE TIME] — [filename created]
STATUS: complete
BUILT: [filename]
TESTED: [what was verified]
RESULT: [test output or confirmation]
NEXT: [next file to build]
```

---

## Build Order (Follow Exactly)

```
0.  Create empty __init__.py files first:
    bayesian/__init__.py
    training/__init__.py
    scripts/__init__.py
    signals/__init__.py (only if it does not already exist)

1.  signals/group_orderbook.py
2.  signals/group_candles.py
3.  signals/group_trades.py
4.  signals/group_futures.py
5.  signals/group_derived.py
6.  Run: pytest unit_test/test_signals_30.py -v
    All must pass before continuing

7.  bayesian/correlation_filter.py
8.  bayesian/prior_loader.py
9.  bayesian/likelihood_engine.py
10. bayesian/threshold_manager.py
11. bayesian/network.py
12. Run: pytest unit_test/test_bayesian.py -v
    All must pass before continuing

13. training/data_downloader.py
14. training/feature_generator.py
15. training/label_generator.py
16. training/likelihood_trainer.py
17. training/correlation_trainer.py
18. scripts/train_bayesian.py

19. Modify brain/master_engine.py
    Replace old scoring with BayesianNetwork
20. Modify learning/trade_analyzer.py
    Add weekly retraining on Sundays

21. Run: pytest unit_test/ -v
    All 117 original + all new tests must pass

22. Run: python scripts/train_bayesian.py
    Complete training pipeline on historical data

23. Run: python run.py
    Verify bot starts with Bayesian system active
    Verify dashboard shows probabilities not scores
```

---

## After Training Is Complete

Once train_bayesian.py finishes:

```
Check data/likelihood_ratios.json exists and has content
Check data/signal_correlations.json exists
Check data/base_rates.json exists

Then run paper trading:
  python run.py

The bot now:
  Downloads live market data (no API key needed)
  Calculates 30 signals in 4 parallel groups in ~120ms
  Runs Bayesian network with real trained likelihood ratios
  Uses dynamic threshold per regime and performance
  Updates likelihood ratios every Sunday automatically
  Makes probabilistic decisions, not static rule checks
```

---
*End of bayesian_upgrade.md*
*Feed this entire file to Claude Code.*
*Build order in Task 7 must be followed exactly.*
