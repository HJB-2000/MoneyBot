from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import brain.regime_detector as regime_module
import scoring.opportunity_scorer as scorer_module
import scoring.pair_ranker as pair_ranker_module
import scoring.signal_combiner as combiner_module
from brain.strategy_router import StrategyRouter
from scoring.opportunity_scorer import OpportunityScorer
from scoring.pair_ranker import PairRanker
from scoring.signal_combiner import ConsensusResult, SignalCombiner
from strategies.base_strategy import Opportunity


def patch_scoring_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(combiner_module, "WEIGHTS_FILE", str(tmp_path / "data" / "signal_weights.json"), raising=False)
    monkeypatch.setattr(regime_module, "REGIME_LOG", str(tmp_path / "data" / "regime_log.csv"), raising=False)
    monkeypatch.setattr(scorer_module, "OPP_LOG", str(tmp_path / "data" / "opportunity_log.csv"), raising=False)
    monkeypatch.setattr(pair_ranker_module, "PAIR_DB", str(tmp_path / "moneyBot.db"), raising=False)
    monkeypatch.setattr(pair_ranker_module, "PAIR_CSV", str(tmp_path / "data" / "pair_rankings.csv"), raising=False)


def make_signal_objects():
    return {
        "volatility": SimpleNamespace(atr_ratio=1.0, vol_spike=False),
        "microstructure": SimpleNamespace(spread_pct=0.0002),
        "whale": SimpleNamespace(whale_buying=False, whale_selling=False),
        "cvd": SimpleNamespace(cvd_divergence=False),
        "price_action": SimpleNamespace(breakout_detected=False),
        "sentiment": SimpleNamespace(funding_rate=0.0),
    }


def make_opportunity(**kwargs):
    defaults = dict(
        strategy="triangular_arb",
        pair="SOL/USDT",
        direction="long",
        entry_price=100.0,
        trade_size_usd=5.0,
        expected_profit_pct=0.001,
        net_profit_pct=0.001,
        fees_pct=0.002,
        slippage_pct=0.001,
        liquidity_ratio=20.0,
        exchange_latency_ms=50.0,
        detected_at=datetime.now(timezone.utc),
        expiry_seconds=60,
    )
    defaults.update(kwargs)
    return Opportunity(**defaults)


def test_combiner_score_in_range(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    result = combiner.combine({name: 0.1 for name in combiner_module.DEFAULT_WEIGHTS})
    assert -1.0 <= result.score <= 1.0


def test_confidence_high_when_signals_agree(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    result = combiner.combine({name: 0.8 for name in combiner_module.DEFAULT_WEIGHTS})
    assert result.confidence > 0.7


def test_confidence_low_when_signals_split(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    signals = {name: 0.8 if i % 2 == 0 else -0.8 for i, name in enumerate(combiner_module.DEFAULT_WEIGHTS)}
    result = combiner.combine(signals)
    assert result.confidence < 0.4


def test_vol_spike_reduces_confidence(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    signal_objects = make_signal_objects()
    signal_objects["volatility"].vol_spike = True
    result = combiner.combine({name: 0.8 for name in combiner_module.DEFAULT_WEIGHTS}, signal_objects)
    assert result.confidence < 0.9


def test_cvd_divergence_boosts_confidence(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    signal_objects = make_signal_objects()
    # Alternate +0.1 and 0.0 so only 4/8 signals are positive → baseline confidence = 0.5
    names = list(combiner_module.DEFAULT_WEIGHTS)
    mixed = {name: (0.1 if i % 2 == 0 else 0.0) for i, name in enumerate(names)}
    baseline = combiner.combine(mixed, signal_objects)
    signal_objects["cvd"].cvd_divergence = True
    boosted = combiner.combine(mixed, signal_objects)
    assert boosted.confidence > baseline.confidence


def test_is_arb_friendly_true_when_conditions_met(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    signal_objects = make_signal_objects()
    assert combiner.is_arb_friendly({name: 0.0 for name in combiner_module.DEFAULT_WEIGHTS}, signal_objects) is True


def test_is_arb_friendly_false_when_volatile(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    signal_objects = make_signal_objects()
    signal_objects["volatility"].atr_ratio = 2.5
    assert combiner.is_arb_friendly({name: 0.0 for name in combiner_module.DEFAULT_WEIGHTS}, signal_objects) is False


def test_dominant_signal_identified(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    combiner = SignalCombiner(config)
    result = combiner.combine({"price_action": 0.1, "momentum": -0.8, "microstructure": 0.2})
    assert result.dominant_signal == "momentum"


def test_volatile_regime_on_high_atr(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    objs = make_signal_objects()
    objs["volatility"].atr_ratio = 2.5
    assert detector.classify({}, cr, objs) == "VOLATILE"


def test_volatile_overrides_everything(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    objs = make_signal_objects()
    objs["volatility"].atr_ratio = 2.5
    objs["sentiment"].funding_rate = 0.01
    assert detector.classify({}, cr, objs) == "VOLATILE"


def test_breakout_detected_on_sr_break(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    objs = make_signal_objects()
    objs["price_action"].breakout_detected = True
    assert detector.classify({}, cr, objs) == "BREAKOUT"


def test_funding_rich_on_high_rate(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    objs = make_signal_objects()
    objs["sentiment"].funding_rate = 0.001
    assert detector.classify({}, cr, objs) == "FUNDING_RICH"


def test_ranging_on_balanced_signals(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    assert detector.classify({}, cr, make_signal_objects()) == "RANGING"


def test_trending_up_on_strong_positive_score(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.6, confidence=0.5, arb_friendly=False)
    assert detector.classify({}, cr, make_signal_objects()) == "TRENDING_UP"


def test_trending_down_on_strong_negative_score(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=-0.6, confidence=0.5, arb_friendly=False)
    assert detector.classify({}, cr, make_signal_objects()) == "TRENDING_DOWN"


def test_choppy_when_mixed_signals(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.1, confidence=0.2, arb_friendly=False)
    assert detector.classify({}, cr, make_signal_objects()) == "CHOPPY"


def test_regime_logged_to_csv(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    detector.classify({}, cr, make_signal_objects())
    assert Path(workspace / "data" / "regime_log.csv").exists()


def test_regime_returns_valid_string(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    detector = regime_module.RegimeDetector(config)
    cr = ConsensusResult(score=0.0, confidence=0.5, arb_friendly=True)
    regime = detector.classify({}, cr, make_signal_objects())
    assert regime in ["VOLATILE", "WHALE_MOVING", "BREAKOUT", "FUNDING_RICH", "RANGING", "TRENDING_UP", "TRENDING_DOWN", "CHOPPY"]


def test_strategy_router_routes_known_regimes():
    router = StrategyRouter()
    result = router.route("RANGING", 0.9)
    assert result.strategies
    assert result.size_mult == pytest.approx(1.0)


def test_strategy_router_scales_down_low_confidence():
    router = StrategyRouter()
    result = router.route("RANGING", 0.2)
    assert result.size_mult < 1.0


def test_scorer_hard_rejects_low_profit(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(net_profit_pct=0.0001)
    assert scorer.score(opp, "RANGING", {}, ConsensusResult(), make_signal_objects()) == 0.0


def test_scorer_hard_rejects_low_liquidity(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(liquidity_ratio=3.0)
    assert scorer.score(opp, "RANGING", {}, ConsensusResult(), make_signal_objects()) == 0.0


def test_scorer_hard_rejects_high_latency(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(exchange_latency_ms=400)
    assert scorer.score(opp, "RANGING", {}, ConsensusResult(), make_signal_objects()) == 0.0


def test_scorer_hard_rejects_stale_opportunity(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(detected_at=datetime.now(timezone.utc) - timedelta(seconds=10), expiry_seconds=3)
    assert scorer.score(opp, "RANGING", {}, ConsensusResult(), make_signal_objects()) == 0.0


def test_scorer_hard_rejects_vol_spike(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    signal_objects = make_signal_objects()
    signal_objects["volatility"].vol_spike = True
    opp = make_opportunity()
    assert scorer.score(opp, "RANGING", {}, ConsensusResult(confidence=0.8), signal_objects) == 0.0


def test_scorer_hard_rejects_wrong_regime(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(strategy="triangular_arb")
    assert scorer.score(opp, "VOLATILE", {}, ConsensusResult(confidence=0.8), make_signal_objects()) == 0.0


def test_scorer_returns_high_score_on_good_opportunity(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    ranker = PairRanker(config, ["SOL/USDT"])
    for _ in range(10):
        ranker.update("SOL/USDT", "WIN", 0.02)
    opp = make_opportunity(net_profit_pct=0.005, liquidity_ratio=50.0, exchange_latency_ms=25.0)
    score = scorer.score(opp, "RANGING", {"cvd": 0.2}, ConsensusResult(confidence=0.95), make_signal_objects(), ranker)
    assert score > 0.75


def test_scorer_score_in_range(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(net_profit_pct=0.005, liquidity_ratio=50.0, exchange_latency_ms=25.0)
    score = scorer.score(opp, "RANGING", {"cvd": 0.1}, ConsensusResult(confidence=0.9), make_signal_objects())
    assert 0.0 <= score <= 1.0


def test_scorer_logs_to_opportunity_log(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    scorer = OpportunityScorer(config)
    opp = make_opportunity(net_profit_pct=0.005, liquidity_ratio=50.0, exchange_latency_ms=25.0)
    scorer.score(opp, "RANGING", {"cvd": 0.1}, ConsensusResult(confidence=0.9), make_signal_objects())
    assert Path(workspace / "data" / "opportunity_log.csv").exists()


def test_pair_ranker_updates_after_trade(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    ranker = PairRanker(config, ["MATIC/USDT"])
    ranker.update("MATIC/USDT", "WIN", 0.01)
    assert ranker.win_rate("MATIC/USDT") == pytest.approx(1.0)


def test_pair_tier_promotion(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    ranker = PairRanker(config, ["MATIC/USDT"])
    for _ in range(10):
        ranker.update("MATIC/USDT", "WIN", 0.01)
    ranker._last_tier_update = None
    assert ranker.get_tier("MATIC/USDT") == "A"


def test_pair_tier_demotion(config, workspace, monkeypatch):
    patch_scoring_paths(monkeypatch, workspace)
    ranker = PairRanker(config, ["MATIC/USDT"])
    for _ in range(10):
        ranker.update("MATIC/USDT", "LOSS", -0.01)
    ranker._last_tier_update = None
    assert ranker.get_tier("MATIC/USDT") in {"B", "C"}
