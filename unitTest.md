# MoneyBot — Unit Test Plan
> Run every test in order. Bot only goes to production when ALL pass.
> Any failure = fix it, re-run from the beginning of that section.
> After every test session, append results to progress.md.

---

## The Testing Philosophy

We test in 4 stages:

```
Stage 1: Unit Tests       ← each component works in isolation
Stage 2: Integration Tests ← components work together
Stage 3: Paper Trading    ← full system works with real market data
Stage 4: Production Gate  ← final checklist before real money
```

Never skip a stage. Never go to production if any test fails.

---

## How to Run

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-timeout pytest-mock

# Run all unit tests
pytest tests/ -v --timeout=30

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a specific file
pytest tests/test_signals.py -v

# Run a specific test
pytest tests/test_signals.py::test_rsi_oversold -v
```

Target: 100% of tests passing before moving to Stage 2.

---

## Stage 1 — Unit Tests

---

### tests/test_capital.py

```python
"""
Tests for capital/tracker.py and capital/compounder.py
"""

def test_initial_capital_loads_from_config():
    # tracker starts at config.starting_capital ($50)
    # PASS if: tracker.get_capital() == 50.0

def test_capital_persists_across_restart():
    # simulate: add $5 profit, restart tracker, check balance
    # PASS if: capital loads as $55.0 after restart, not $50.0

def test_capital_updates_on_profit():
    # call tracker.update(+2.50)
    # PASS if: get_capital() increases by exactly 2.50

def test_capital_updates_on_loss():
    # call tracker.update(-1.00)
    # PASS if: get_capital() decreases by exactly 1.00

def test_capital_never_goes_negative():
    # call tracker.update(-999.00) on $50 capital
    # PASS if: capital floors at survival_floor (not negative)

def test_daily_pnl_resets_at_midnight():
    # simulate trades, advance time to midnight
    # PASS if: get_daily_pnl() resets to 0.0 at new day

def test_milestone_alert_at_100():
    # grow capital from $50 to $101
    # PASS if: milestone_check() triggers alert for $100 milestone

def test_milestone_not_triggered_twice():
    # cross $100 milestone, then update again
    # PASS if: same milestone does not trigger twice

def test_drawdown_calculation():
    # peak = $100, current = $85
    # PASS if: get_drawdown() == 0.15 (15%)

def test_win_rate_last_20():
    # log 20 trades: 12 wins, 8 losses
    # PASS if: get_win_rate(20) == 0.60

def test_consecutive_losses_counter():
    # log 3 losses in a row
    # PASS if: get_consecutive_losses() == 3

def test_consecutive_losses_resets_on_win():
    # log 2 losses then 1 win
    # PASS if: get_consecutive_losses() == 0

def test_compound_adds_to_pool():
    # compounder.compound(5.00) on $50 capital
    # PASS if: capital == 55.00

def test_daily_report_created():
    # trigger daily report generation
    # PASS if: file exists at reports/daily/YYYY-MM-DD.txt
```

---

### tests/test_market_reader.py

```python
"""
Tests for brain/market_reader.py
Uses real Binance API in paper mode — requires internet.
"""

def test_orderbook_returns_bids_and_asks():
    # fetch MATIC/USDT orderbook
    # PASS if: result has 'bids' and 'asks', each non-empty list

def test_orderbook_bids_sorted_descending():
    # fetch orderbook
    # PASS if: bids[0][0] > bids[1][0] (highest bid first)

def test_orderbook_asks_sorted_ascending():
    # PASS if: asks[0][0] < asks[1][0] (lowest ask first)

def test_candles_returns_correct_shape():
    # fetch 100 candles at 5m
    # PASS if: DataFrame shape == (100, 6)
    # columns: timestamp, open, high, low, close, volume

def test_candles_no_gaps():
    # fetch 100 candles
    # PASS if: timestamps are evenly spaced (no missing candles)

def test_funding_rate_returns_float():
    # fetch funding rate for a futures pair
    # PASS if: result is float between -0.01 and 0.01

def test_api_latency_tracked():
    # fetch any data
    # PASS if: market_reader.avg_latency_ms > 0

def test_cache_returns_same_data_within_5s():
    # call get_orderbook twice within 5 seconds
    # PASS if: both calls return identical data (cache working)

def test_graceful_handling_of_bad_symbol():
    # call get_orderbook('FAKE/USDT')
    # PASS if: returns None without raising exception

def test_graceful_handling_of_timeout():
    # mock a network timeout
    # PASS if: returns None without crashing
```

---

### tests/test_signals.py

```python
"""
Tests for all 8 signal generators.
Each signal must return float in [-1.0, 1.0] and complete in < 500ms.
"""

# ── Price Action ──────────────────────────────────────────

def test_price_action_score_in_range():
    # run on real MATIC/USDT candles
    # PASS if: -1.0 <= score <= 1.0

def test_price_action_returns_zero_on_empty_data():
    # pass empty DataFrame
    # PASS if: returns 0.0 without crashing

def test_price_action_exposes_breakout_detected():
    # run signal
    # PASS if: signal.breakout_detected is bool

def test_price_action_exposes_trend():
    # PASS if: signal.trend in ['uptrend','downtrend','ranging']

def test_price_action_consolidation_on_flat_data():
    # create candles with < 0.3% range
    # PASS if: signal.consolidating == True

def test_price_action_completes_under_500ms():
    # time the calculate() call
    # PASS if: execution_time < 0.5 seconds

# ── Momentum ──────────────────────────────────────────────

def test_momentum_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_rsi_oversold_gives_positive_score():
    # create candles with declining prices (RSI < 30)
    # PASS if: momentum score > 0.0

def test_rsi_overbought_gives_negative_score():
    # create candles with rising prices (RSI > 70)
    # PASS if: momentum score < 0.0

def test_macd_crossover_detected():
    # create candles with clear MACD crossover
    # PASS if: momentum score changes sign at crossover

def test_momentum_returns_zero_on_insufficient_data():
    # pass only 5 candles (need 26 for MACD)
    # PASS if: returns 0.0 without crashing

def test_momentum_completes_under_500ms():
    # PASS if: execution_time < 0.5 seconds

# ── Microstructure ────────────────────────────────────────

def test_microstructure_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_ofi_balanced_gives_positive_score():
    # mock trades: 50% buys, 50% sells
    # PASS if: score > 0.0 (balanced = good for arb)

def test_ofi_heavy_buying_detected():
    # mock trades: 80% buys
    # PASS if: signal detects buy pressure

def test_wide_spread_gives_negative_score():
    # mock orderbook with 0.5% spread
    # PASS if: spread_score < 0.0

def test_tight_spread_gives_positive_score():
    # mock orderbook with 0.02% spread
    # PASS if: spread_score > 0.0

def test_whale_detection_on_large_trade():
    # mock single trade = 10% of volume
    # PASS if: whale_detected == True

def test_whale_detection_absent_on_small_trades():
    # mock all trades < 1% of volume
    # PASS if: whale_detected == False

def test_microstructure_exposes_whale_detected():
    # PASS if: signal.whale_detected is bool

def test_microstructure_completes_under_500ms():
    # PASS if: execution_time < 0.5 seconds

# ── Volatility ────────────────────────────────────────────

def test_volatility_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_atr_ratio_exposed():
    # PASS if: signal.atr_ratio > 0.0

def test_high_atr_ratio_gives_negative_score():
    # mock candles with very wide ranges (ATR ratio > 2.0)
    # PASS if: score < -0.5

def test_low_atr_ratio_gives_positive_score():
    # mock candles with tight ranges (ATR ratio < 0.7)
    # PASS if: score > 0.3

def test_vol_spike_exposed():
    # PASS if: signal.vol_spike is bool

def test_vol_spike_true_on_sudden_expansion():
    # create candles where last 5min vol >> 24h average
    # PASS if: vol_spike == True

def test_bollinger_squeeze_detected():
    # create candles with tightening Bollinger bands
    # PASS if: signal.is_squeezing == True

def test_volatility_completes_under_500ms():
    # PASS if: execution_time < 0.5 seconds

# ── Sentiment ─────────────────────────────────────────────

def test_sentiment_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_positive_funding_gives_positive_score():
    # mock funding_rate = 0.001 (very positive)
    # PASS if: score > 0.0

def test_negative_funding_gives_negative_score():
    # mock funding_rate = -0.001
    # PASS if: score < 0.0

def test_funding_rate_exposed():
    # PASS if: signal.funding_rate is float

def test_liquidation_cascade_gives_negative_score():
    # mock large liquidation event
    # PASS if: score < -0.5

def test_sentiment_returns_zero_on_missing_data():
    # pass None for all inputs
    # PASS if: returns 0.0 without crashing

# ── CVD ───────────────────────────────────────────────────

def test_cvd_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_cvd_bullish_divergence_gives_strong_positive():
    # mock: price falling but CVD rising
    # PASS if: score > 0.5

def test_cvd_bearish_divergence_gives_strong_negative():
    # mock: price rising but CVD falling
    # PASS if: score < -0.5

def test_cvd_confirmed_uptrend():
    # mock: price rising AND CVD rising
    # PASS if: score > 0.0

def test_cvd_flat_gives_neutral_score():
    # mock: CVD flat, no trend
    # PASS if: -0.2 <= score <= 0.2

def test_cvd_completes_under_500ms():
    # PASS if: execution_time < 0.5 seconds

# ── VWAP ──────────────────────────────────────────────────

def test_vwap_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_vwap_price_exposed():
    # PASS if: signal.vwap_price > 0.0

def test_price_above_upper_band_negative_score():
    # mock price far above VWAP upper band
    # PASS if: score < -0.2

def test_price_below_lower_band_positive_score():
    # mock price far below VWAP lower band
    # PASS if: score > 0.2

def test_vwap_reclaim_gives_strong_positive():
    # mock price crossing above VWAP with volume
    # PASS if: score > 0.4

def test_vwap_resets_at_midnight():
    # simulate day boundary crossing
    # PASS if: vwap_price recalculates from zero

# ── Whale Detector ────────────────────────────────────────

def test_whale_detector_score_in_range():
    # PASS if: -1.0 <= score <= 1.0

def test_large_buy_orders_give_positive_score():
    # mock several trades > $50k each, all buys
    # PASS if: score > 0.0 and whale_buying == True

def test_large_sell_orders_give_negative_score():
    # mock several trades > $50k each, all sells
    # PASS if: score < 0.0 and whale_selling == True

def test_iceberg_bid_detected():
    # mock orderbook level refilling repeatedly
    # PASS if: iceberg_detected == True

def test_spoofing_detected_on_disappearing_order():
    # mock large bid appearing then vanishing in 1 second
    # PASS if: score adjusted for spoofing

def test_whale_detector_exposes_whale_buying():
    # PASS if: signal.whale_buying is bool

def test_whale_detector_completes_under_500ms():
    # PASS if: execution_time < 0.5 seconds
```

---

### tests/test_signal_combiner.py

```python
def test_combiner_score_in_range():
    # run combiner with all 8 signals
    # PASS if: -1.0 <= result.score <= 1.0

def test_confidence_high_when_signals_agree():
    # mock all 8 signals returning > 0.5
    # PASS if: result.confidence > 0.7

def test_confidence_low_when_signals_split():
    # mock 4 signals positive, 4 signals negative
    # PASS if: result.confidence < 0.4

def test_vol_spike_reduces_confidence():
    # mock vol_spike = True
    # PASS if: result.confidence reduced by at least 0.30

def test_cvd_divergence_boosts_confidence():
    # mock CVD divergence detected
    # PASS if: result.confidence increased by at least 0.10

def test_is_arb_friendly_true_when_conditions_met():
    # mock: low vol, tight spread, balanced OFI, no whales
    # PASS if: is_arb_friendly() == True

def test_is_arb_friendly_false_when_volatile():
    # mock: atr_ratio > 2.0
    # PASS if: is_arb_friendly() == False

def test_dominant_signal_identified():
    # PASS if: result.dominant_signal is str (one of 8 signal names)
```

---

### tests/test_regime_detector.py

```python
def test_volatile_regime_on_high_atr():
    # mock atr_ratio = 2.5
    # PASS if: regime == 'VOLATILE' (highest priority)

def test_volatile_overrides_everything():
    # mock atr_ratio = 2.5 AND positive funding rate
    # PASS if: regime == 'VOLATILE' (not FUNDING_RICH)

def test_breakout_detected_on_sr_break():
    # mock breakout_detected = True
    # PASS if: regime == 'BREAKOUT'

def test_funding_rich_on_high_rate():
    # mock funding_rate = 0.001, atr_ratio = 1.0
    # PASS if: regime == 'FUNDING_RICH'

def test_ranging_on_balanced_signals():
    # mock all signals near 0.0, low volatility
    # PASS if: regime == 'RANGING'

def test_trending_up_on_strong_positive_score():
    # mock combiner_score = 0.6
    # PASS if: regime == 'TRENDING_UP'

def test_trending_down_on_strong_negative_score():
    # mock combiner_score = -0.6
    # PASS if: regime == 'TRENDING_DOWN'

def test_choppy_when_mixed_signals():
    # mock combiner_score = 0.1 but confidence = 0.2
    # PASS if: regime == 'CHOPPY'

def test_regime_logged_to_csv():
    # run classifier
    # PASS if: new row added to data/regime_log.csv

def test_regime_returns_valid_string():
    # PASS if: regime in valid list of 8 regimes
```

---

### tests/test_strategies.py

```python
# ── Triangular Arb ────────────────────────────────────────

def test_triangular_last_leg_is_sell():
    # inspect path legs for any path
    # PASS if: legs[-1]['direction'] == 'sell'

def test_triangular_profitable_mock_data():
    # mock orderbook with known 0.5% spread across 3 legs
    # PASS if: opportunity.net_profit_pct > 0

def test_triangular_rejects_unprofitable_path():
    # mock orderbook with zero spread
    # PASS if: scan() returns empty list

def test_triangular_checks_liquidity():
    # mock orderbook with very low depth
    # PASS if: scan() returns empty list (fails 15x check)

def test_triangular_rejects_stale_opportunity():
    # create opportunity, wait 6 seconds
    # PASS if: opportunity rejected (expiry = 3s)

def test_triangular_pauses_in_volatile_regime():
    # run scan with regime = 'VOLATILE'
    # PASS if: scan() returns empty list

def test_triangular_dynamic_slippage_applied():
    # verify slippage is 0.002 for $50 trade, 0.001 for $200
    # PASS if: slippage varies correctly by size

# ── Statistical Arb ───────────────────────────────────────

def test_stat_arb_correlation_matrix_built():
    # run correlation fetch
    # PASS if: data/correlation_matrix.json exists and non-empty

def test_stat_arb_finds_correlated_pairs():
    # PASS if: at least 1 pair with correlation > 0.85 found

def test_stat_arb_z_score_triggers_entry():
    # mock price_ratio with z_score = 2.5
    # PASS if: opportunity created

def test_stat_arb_no_entry_below_threshold():
    # mock z_score = 1.5 (below 2.0)
    # PASS if: scan() returns empty list

def test_stat_arb_long_only_in_trending_up():
    # run with regime = 'TRENDING_UP', bias = 'long_only'
    # PASS if: all opportunities are direction='long'

def test_stat_arb_position_saved_to_json():
    # simulate opening a position
    # PASS if: data/stat_arb_positions.json contains the position

def test_stat_arb_exits_on_z_score_reversion():
    # mock z_score returning to 0.3
    # PASS if: position closed with profit

def test_stat_arb_exits_on_stop():
    # mock z_score reaching 4.0
    # PASS if: position closed with loss

def test_stat_arb_exits_on_time_stop():
    # mock position held for 5 hours
    # PASS if: position closed (max_hold_hours = 4)

# ── Grid Trader ───────────────────────────────────────────

def test_grid_creates_correct_number_of_levels():
    # setup grid with 10 levels
    # PASS if: 10 buy + 10 sell orders created

def test_grid_pauses_in_trending_regime():
    # run with regime = 'TRENDING_UP'
    # PASS if: grid paused (not closed)

def test_grid_stops_when_price_leaves_range():
    # mock price dropping below lower_bound
    # PASS if: grid positions closed

def test_grid_profit_per_roundtrip():
    # mock price oscillating through grid level
    # PASS if: net_profit > 0 after fees on each roundtrip

def test_max_grids_not_exceeded():
    # setup 3 grids when max is 2
    # PASS if: only 2 grids running simultaneously

# ── Mean Reversion ────────────────────────────────────────

def test_mean_reversion_triggers_on_2pct_drop():
    # mock 2.1% price drop in 5 minutes
    # PASS if: opportunity created

def test_mean_reversion_no_trigger_on_1pct_drop():
    # mock 1% drop
    # PASS if: scan() returns empty list

def test_mean_reversion_skips_when_btc_also_dropping():
    # mock target pair -2.5% AND BTC -1.5%
    # PASS if: scan() returns empty list (systemic move)

def test_mean_reversion_skips_high_volume():
    # mock drop with volume 6x average
    # PASS if: scan() returns empty list

def test_mean_reversion_exits_at_target():
    # mock price recovering 1%
    # PASS if: position closed with profit

def test_mean_reversion_exits_at_stop():
    # mock price dropping another 1.5%
    # PASS if: position closed with loss

# ── Volume Spike ──────────────────────────────────────────

def test_volume_spike_triggers_on_3x_volume():
    # mock 3.5x volume spike with 0.7% price move
    # PASS if: opportunity created

def test_volume_spike_requires_price_move():
    # mock 5x volume but only 0.3% price move
    # PASS if: scan() returns empty list

def test_volume_spike_direction_follows_price():
    # mock spike with price moving up
    # PASS if: opportunity.direction == 'long'

def test_volume_spike_pauses_in_volatile():
    # regime = 'VOLATILE'
    # PASS if: scan() returns empty list

# ── Correlation Breakout ──────────────────────────────────

def test_correlation_breakout_triggers_on_btc_move():
    # mock BTC move of 2.5% in 5 minutes
    # PASS if: at least one lagging pair opportunity created

def test_correlation_breakout_skips_if_pair_already_moved():
    # mock BTC +3%, target pair already +2.5%
    # PASS if: no opportunity (lag already closed)

def test_correlation_breakout_rejects_stale_btc_move():
    # mock BTC moved 15 minutes ago
    # PASS if: no opportunity (outside lag window)
```

---

### tests/test_scorer.py

```python
def test_scorer_hard_rejects_low_profit():
    # opportunity.net_profit_pct = 0.0001 (below 0.03%)
    # PASS if: score == 0.0

def test_scorer_hard_rejects_low_liquidity():
    # liquidity_ratio = 3.0 (below 5x)
    # PASS if: score == 0.0

def test_scorer_hard_rejects_high_latency():
    # exchange_latency_ms = 400
    # PASS if: score == 0.0

def test_scorer_hard_rejects_stale_opportunity():
    # detected_at = 10 seconds ago, expiry = 3 seconds
    # PASS if: score == 0.0

def test_scorer_hard_rejects_vol_spike():
    # vol_spike = True
    # PASS if: score == 0.0

def test_scorer_hard_rejects_wrong_regime():
    # opportunity.strategy = 'triangular_arb', regime = 'VOLATILE'
    # PASS if: score == 0.0

def test_scorer_returns_high_score_on_good_opportunity():
    # mock: 0.3% profit, 20x liquidity, 50ms latency, good signals
    # PASS if: score > 0.75

def test_scorer_score_in_range():
    # PASS if: 0.0 <= score <= 1.0

def test_scorer_logs_to_opportunity_log():
    # run scorer on valid opportunity
    # PASS if: new row in data/opportunity_log.csv with score column

def test_pair_ranker_updates_after_trade():
    # simulate WIN trade on MATIC/USDT
    # PASS if: pair_ranker.win_rate('MATIC/USDT') updated

def test_pair_tier_promotion():
    # simulate 10 wins on same pair
    # PASS if: pair promoted to Tier A

def test_pair_tier_demotion():
    # simulate 5 losses on Tier A pair
    # PASS if: pair demoted to Tier B or C
```

---

### tests/test_risk.py

```python
def test_approve_rejects_below_survival_floor():
    # capital = $0.50, floor = $1.00
    # PASS if: approve() == False

def test_approve_rejects_daily_loss_limit():
    # daily_pnl = -6% (limit = 5%)
    # PASS if: approve() == False

def test_approve_rejects_max_open_trades():
    # open_trades = 3 (max = 3)
    # PASS if: approve() == False

def test_approve_rejects_cb1_active():
    # 3 consecutive losses logged
    # PASS if: approve() returns True but size is halved

def test_approve_rejects_cb2_daily_drawdown():
    # drawdown = 16% (limit = 15%)
    # PASS if: approve() == False for rest of day

def test_approve_rejects_cb3_low_win_rate():
    # win_rate_last_20 = 50% (limit = 55%)
    # PASS if: approve() returns True but size * 0.7

def test_approve_passes_all_gates_clean():
    # all conditions green
    # PASS if: approve() == True with positive size

def test_position_sizer_pre_kelly():
    # total_trades = 30 (below 50)
    # PASS if: size == capital * 0.02

def test_position_sizer_post_kelly():
    # total_trades = 60, win_rate = 0.60, avg_win/loss = 1.5
    # PASS if: size == kelly_formula * 0.25 * capital

def test_position_sizer_respects_tier_cap():
    # capital = $300 (below $500, cap = 8%)
    # PASS if: size <= capital * 0.08

def test_position_sizer_atr_adjustment():
    # atr_ratio = 1.8
    # PASS if: size reduced by ~30%

def test_position_sizer_never_exceeds_25pct():
    # all circuit breakers clear, Kelly gives 30%
    # PASS if: size <= capital * 0.25

def test_volatility_guard_pauses_on_price_spike():
    # mock 2.1% price move in 60 seconds
    # PASS if: approve() == False for 300 seconds

def test_api_latency_guard_skips_slow_exchange():
    # mock avg_latency = 250ms
    # PASS if: approve() == False
```

---

### tests/test_executor.py

```python
def test_executor_cancels_stale_opportunity():
    # opportunity detected 6 seconds ago, expiry = 3s
    # PASS if: ExecutionResult.cancelled == True, reason='stale'

def test_executor_cancels_on_price_move():
    # price moved 0.2% since detection
    # PASS if: ExecutionResult.cancelled == True, reason='price_moved'

def test_executor_cancels_on_failed_rescore():
    # fresh score below threshold
    # PASS if: ExecutionResult.cancelled == True

def test_executor_updates_capital_on_win():
    # simulate winning paper trade
    # PASS if: capital increases by net_profit

def test_executor_updates_capital_on_loss():
    # simulate losing paper trade
    # PASS if: capital decreases by net_loss

def test_executor_logs_to_trade_log():
    # execute any paper trade
    # PASS if: new row in data/trade_log.csv

def test_executor_realistic_fill():
    # execute paper trade
    # PASS if: fill = 85% at best + 15% at 1 tick worse

def test_executor_deducts_fees():
    # verify taker fee deducted both sides
    # PASS if: gross_profit - fees == net_profit

def test_executor_deducts_slippage():
    # verify slippage deducted
    # PASS if: slippage_pct > 0 in trade log

def test_executor_updates_pair_ranker():
    # after WIN trade on MATIC/USDT
    # PASS if: pair_ranker records the win
```

---

### tests/test_ml.py

```python
def test_models_load_from_disk():
    # load all 3 trained models
    # PASS if: all 3 load without error

def test_predictor_returns_probability():
    # run predict() with mock features
    # PASS if: 0.0 <= ml_probability <= 1.0

def test_predictor_completes_under_100ms():
    # time the predict() call
    # PASS if: execution_time < 0.1 seconds

def test_feature_vector_correct_length():
    # build features from mock signals
    # PASS if: len(features) == 35

def test_ensemble_averages_three_models():
    # mock: XGB=0.7, LGBM=0.6, RF=0.65
    # PASS if: ml_probability == 0.65

def test_high_probability_boosts_score():
    # ml_probability = 0.80
    # PASS if: opportunity score increases vs no ML

def test_low_probability_reduces_score():
    # ml_probability = 0.30
    # PASS if: opportunity score reduced

def test_retrainer_requires_minimum_samples():
    # mock trade_log with only 100 rows
    # PASS if: retraining skipped (needs 500+)

def test_retrainer_updates_models_when_better():
    # mock new model accuracy > old model accuracy
    # PASS if: model files updated on disk
```

---

### tests/test_learning_engine.py

```python
def test_daily_analysis_runs_without_crash():
    # run trade_analyzer.daily_analysis()
    # PASS if: completes without exception

def test_signal_weights_updated():
    # run daily analysis with mock trade data
    # PASS if: data/signal_weights.json modified

def test_signal_weights_sum_to_one():
    # after any weight update
    # PASS if: sum(weights.values()) == 1.0

def test_signal_weights_bounded():
    # after any weight update
    # PASS if: all weights between 0.03 and 0.40

def test_max_weight_change_per_day():
    # run analysis for signal with 100% accuracy
    # PASS if: weight changed by at most 0.02

def test_pair_tier_updates_after_analysis():
    # run analysis with mock pair performance data
    # PASS if: pair_rankings.csv updated

def test_poor_strategy_gets_disabled():
    # mock strategy with win_rate 40% for 7 days
    # PASS if: strategy disabled in strategy_params.json

def test_disabled_strategy_reenabled_on_improvement():
    # mock disabled strategy now showing win_rate 60%
    # PASS if: strategy re-enabled

def test_score_threshold_raises_on_bad_win_rate():
    # mock 7-day win_rate = 50%
    # PASS if: execution_threshold increases by 0.01

def test_score_threshold_lowers_on_good_win_rate():
    # mock 7-day win_rate = 75%
    # PASS if: execution_threshold decreases by 0.01

def test_threshold_never_below_minimum():
    # mock excellent win_rate for 30 days
    # PASS if: threshold >= 0.62

def test_daily_report_contains_key_metrics():
    # generate report
    # PASS if: report file contains capital, pnl, win_rate, regime_dist
```

---

## Stage 2 — Integration Tests

### tests/test_integration.py

```python
def test_all_threads_start_successfully():
    # start master_engine
    # PASS if: all 6 threads alive after 10 seconds

def test_regime_classified_every_30_seconds():
    # run engine for 2 minutes
    # PASS if: regime_log.csv has 4+ entries, each 30s apart

def test_opportunities_logged_from_multiple_strategies():
    # run engine for 30 minutes
    # PASS if: opportunity_log has entries from 2+ different strategies

def test_capital_updates_after_paper_trade():
    # run engine until a paper trade executes
    # PASS if: capital != starting_capital after trade

def test_no_thread_crashes_in_30_minutes():
    # run engine 30 minutes
    # PASS if: all 6 threads still alive at end

def test_console_output_every_cycle():
    # capture stdout for 2 minutes
    # PASS if: output printed at least 4 times

def test_data_files_created_and_growing():
    # run engine 10 minutes
    # PASS if: opportunity_log.csv, regime_log.csv both have new rows

def test_kill_switch_activates_on_10pct_loss():
    # simulate 10% daily loss
    # PASS if: paper_mode switches to True automatically

def test_graceful_shutdown_on_keyboard_interrupt():
    # send KeyboardInterrupt to engine
    # PASS if: all positions saved, capital saved, no data loss

def test_open_positions_survive_restart():
    # open a paper position, restart engine
    # PASS if: position still tracked after restart

def test_ml_predictor_runs_in_live_cycle():
    # run engine with ML active
    # PASS if: ml_probability column appears in opportunity_log.csv

def test_circuit_breaker_halts_trading():
    # simulate 3 consecutive losses
    # PASS if: next approved trade has size halved

def test_regime_change_updates_active_strategies():
    # force regime change from RANGING to VOLATILE
    # PASS if: all strategies pause within one cycle
```

---

## Stage 3 — Paper Trading Test

### 48-Hour Paper Run Checklist

Run the full bot in paper mode with real Binance market data.
Do not proceed to Stage 4 until ALL items are checked.

```
SETUP:
□ config.yaml: paper_mode: true
□ All API keys set as environment variables
□ Starting capital confirmed at $50.00 in SQLite
□ All unit tests passed (Stage 1)
□ All integration tests passed (Stage 2)

AT 6 HOURS:
□ opportunity_log.csv has entries from at least 2 strategies
□ regime_log.csv shows at least 3 different regimes classified
□ No thread has crashed (check all 6 threads alive)
□ Console output printing every 30 seconds
□ No unhandled exceptions in any log file
□ ML predictor running (ml_probability column in log)

AT 24 HOURS:
□ At least 10 paper trades executed (check trade_log.csv)
□ Win rate on paper trades > 50%
□ Capital has changed from $50.00 (trades happened)
□ Pair rankings updated (pair_rankings.csv modified)
□ Daily report generated (reports/daily/ has file)
□ No strategy disabled due to bugs (only due to performance)
□ Average cycle time < 5 seconds (check console output)

AT 48 HOURS:
□ Learning engine ran at midnight (check learning_log.csv)
□ signal_weights.json updated from defaults
□ Daily report generated for both days
□ Total paper trades: at least 30
□ Overall win rate: > 52%
□ Maximum drawdown during 48h: < 15%
□ ML models still loading correctly
□ No memory leaks (check system memory usage)
□ Capital tracking consistent (no unexplained jumps)

PERFORMANCE BENCHMARKS (all must pass):
□ master_cycle completes in < 5 seconds average
□ Signal generation (all 8) completes in < 2 seconds
□ Opportunity scoring completes in < 1 second
□ ML prediction completes in < 100ms
□ Regime classification completes in < 500ms
```

---

## Stage 4 — Production Gate

### Final Checklist Before Real Money

Every single item must be checked before switching paper_mode to false.

```
CODE QUALITY:
□ All Stage 1 unit tests pass (0 failures)
□ All Stage 2 integration tests pass (0 failures)
□ 48-hour paper run completed with all items checked
□ No hardcoded API keys anywhere in codebase
□ API keys only in environment variables

CAPITAL SAFETY:
□ Kill switch tested and confirmed working
□ Survival floor confirmed ($1.00 blocks all trades)
□ Daily loss limit confirmed (5% halts trading)
□ Circuit breakers tested (all 4 confirmed working)
□ Position size cap confirmed (never exceeds 25%)

CONFIGURATION:
□ paper_mode: false set in config
□ max_position_pct: 0.05 (start tiny)
□ manual_approval: true (first 5 days)
□ All thresholds verified in config.yaml

DATA:
□ SQLite database clean and starting at real capital ($50)
□ ML models trained on 90 days of historical data
□ pair_rankings.csv initialized (all pairs Tier B)
□ signal_weights.json at default values
□ All data/ CSV files exist and writable
□ reports/daily/ directory exists and writable

MONITORING:
□ You know how to check: tail -f data/trade_log.csv
□ You know how to check: tail -20 progress.md
□ You know how to stop: KeyboardInterrupt (Ctrl+C)
□ You have read the daily report format and understand it
□ You know the kill switch will auto-activate at -10% daily

MENTAL CHECKLIST:
□ I will not touch the capital pool for 3 months
□ I will not increase position sizes faster than the weekly schedule
□ I will check progress.md every day for the first 2 weeks
□ I understand the bot may have losing days and that is normal
□ I have read and understood the risk manager rules
```

---

## Test Coverage Target

```
Component                  Target Coverage
─────────────────────────────────────────
capital/tracker.py              95%
capital/compounder.py           90%
brain/market_reader.py          80%
brain/signals/*.py              85%
scoring/signal_combiner.py      90%
brain/regime_detector.py        95%
brain/strategy_router.py        95%
strategies/*.py                 85%
scoring/opportunity_scorer.py   95%
scoring/pair_ranker.py          90%
risk/manager.py                 98%
risk/position_sizer.py          95%
execution/smart_executor.py     90%
learning/*.py                   80%
─────────────────────────────────────────
Overall target:                 88%+
```

Run coverage report:
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## The Rule

```
Stage 1 all green → Stage 2
Stage 2 all green → Stage 3
Stage 3 all checked → Stage 4
Stage 4 all checked → production

Any failure at any stage = fix and restart that stage from the top.
No exceptions. No shortcuts. Real money is on the line.
```

---
*End of unittestPlan.md*
*This file is read-only after creation.*
*All test results logged to progress.md*