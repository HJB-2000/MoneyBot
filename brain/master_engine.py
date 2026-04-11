import logging
import os
import queue
import threading
import time
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from datetime import datetime, timezone

from brain.market_reader import MarketReader
from brain.signals.price_action import PriceActionSignal
from brain.signals.momentum import MomentumSignal
from brain.signals.microstructure import MicrostructureSignal
from brain.signals.volatility import VolatilitySignal
from brain.signals.sentiment import SentimentSignal
from brain.signals.cvd import CVDSignal
from brain.signals.vwap import VWAPSignal
from brain.signals.whale_detector import WhaleDetectorSignal
from brain.regime_detector import RegimeDetector
from brain.strategy_router import StrategyRouter
from capital.tracker import CapitalTracker
from capital.compounder import Compounder
from scoring.signal_combiner import SignalCombiner
from scoring.opportunity_scorer import OpportunityScorer
from scoring.pair_ranker import PairRanker
from risk.manager import RiskManager
from risk.position_sizer import PositionSizer
from execution.smart_executor import SmartExecutor
from execution.order_manager import OrderManager
from signals.group_candles import CandleSignals
from signals.group_orderbook import OrderBookSignals
from signals.group_trades import TradeSignals
from signals.group_futures import FuturesSignals
from signals.group_derived import DerivedSignals
from strategies.triangular_arb import TriangularArbStrategy
from strategies.stat_arb import StatArbStrategy
from strategies.funding_arb import FundingArbStrategy
from strategies.grid_trader import GridTraderStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.volume_spike import VolumeSpikeStrategy
from strategies.correlation_breakout import CorrelationBreakoutStrategy

_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("MasterEngine")

BORDER = "═" * 55


class MasterEngine:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Core components
        self.market_reader = MarketReader(self.config)
        self.tracker = CapitalTracker(self.config)
        self.compounder = Compounder(self.tracker)
        self.pair_ranker = PairRanker(
            self.config, self.config["pairs"]["scan_universe"]
        )

        # Signal layer
        self.signals_map = {
            "price_action":   PriceActionSignal(),
            "momentum":       MomentumSignal(),
            "microstructure": MicrostructureSignal(),
            "volatility":     VolatilitySignal(),
            "sentiment":      SentimentSignal(),
            "cvd":            CVDSignal(),
            "vwap":           VWAPSignal(),
            "whale":          WhaleDetectorSignal(),
        }
        self.combiner = SignalCombiner(self.config)
        self.regime_detector = RegimeDetector(self.config)
        self.router = StrategyRouter()

        # Strategies
        self.strategies = {
            "triangular_arb":       TriangularArbStrategy(),
            "stat_arb":             StatArbStrategy(),
            "funding_arb":          FundingArbStrategy(),
            "grid_trader":          GridTraderStrategy(),
            "mean_reversion":       MeanReversionStrategy(),
            "volume_spike":         VolumeSpikeStrategy(),
            "correlation_breakout": CorrelationBreakoutStrategy(),
        }

        # Risk + execution
        self.sizer = PositionSizer(self.config, self.tracker)
        self.risk = RiskManager(self.config, self.tracker, self.sizer)
        self.order_manager = OrderManager()
        self.scorer = OpportunityScorer(self.config)

        # 30-signal groups for Bayesian scoring
        self._candle_sig   = CandleSignals()
        self._ob_sig       = OrderBookSignals()
        self._trade_sig    = TradeSignals()
        self._futures_sig  = FuturesSignals()
        self._derived_sig  = DerivedSignals()

        # Threading
        self.brain_queue: queue.Queue = queue.Queue(maxsize=1)
        self.opp_queue: queue.Queue = queue.Queue(maxsize=50)
        self.capital_lock = threading.Lock()
        self._running = False
        self._cycle_count = 0

        # Shared brain state
        self._current_signals: dict = {}
        self._current_signal_objects: dict = {}
        self._current_combiner_result = None
        self._current_regime: str = "CHOPPY"
        self._current_route = None
        self._current_signals_30: dict = {}   # the 30 Bayesian signals

    # ------------------------------------------------------------------ #
    #  Startup                                                             #
    # ------------------------------------------------------------------ #

    def start(self):
        logger.info("Starting MoneyBot...")
        self._running = True

        # Step 8: Start data thread, wait for first fill
        t_data = threading.Thread(target=self._data_thread, name="DataThread", daemon=True)
        t_data.start()
        logger.info("Waiting 5s for initial data...")
        time.sleep(5)

        # Steps 9-13: Start all threads
        threads = [
            threading.Thread(target=self._brain_thread, name="BrainThread", daemon=True),
            threading.Thread(target=self._scan_thread, name="ScanThread", daemon=True),
            threading.Thread(target=self._execution_thread, name="ExecThread", daemon=True),
            threading.Thread(target=self._monitor_thread, name="MonitorThread", daemon=True),
            threading.Thread(target=self._learning_thread, name="LearnThread", daemon=True),
        ]
        for t in threads:
            t.start()

        self._print_startup()

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            self._shutdown()

    def _shutdown(self):
        logger.info("Shutting down...")
        self._running = False
        self.tracker._save_balance(self.tracker.get_capital())
        logger.info(f"Final capital: ${self.tracker.get_capital():.2f}")
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------ #
    #  Thread 1 — Data                                                     #
    # ------------------------------------------------------------------ #

    def _data_thread(self):
        """Pre-fetch and cache market data for all active pairs every 5s."""
        pairs = self.config["pairs"]["scan_universe"]
        while self._running:
            try:
                for sym in pairs:
                    self.market_reader.get_ticker(sym)
                    self.market_reader.get_candles(sym, "5m", limit=100)
            except Exception as e:
                logger.warning(f"DataThread error: {e}")
            time.sleep(5)

    # ------------------------------------------------------------------ #
    #  Thread 2 — Brain                                                    #
    # ------------------------------------------------------------------ #

    def _brain_thread(self):
        """Compute signals → regime → route every 30s."""
        interval = self.config["regime"]["scan_interval_seconds"]
        while self._running:
            try:
                self._run_brain()
            except Exception as e:
                logger.error(f"BrainThread error: {e}", exc_info=True)
            time.sleep(interval)

    def _run_brain(self):
        mr = self.market_reader
        # Use SOL/USDT as the "market representative" for signals
        sym = "SOL/USDT"
        candles = mr.get_candles(sym)
        ob = mr.get_orderbook(sym)
        trades = mr.get_trades(sym)
        funding = mr.get_funding_rate(sym)
        oi = mr.get_open_interest(sym)
        oi_1h_ago = mr.get_oi_1h_ago(sym)
        liquidations = mr.get_liquidations(sym)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(self.signals_map["price_action"].calculate, candles): "price_action",
                pool.submit(self.signals_map["momentum"].calculate, candles): "momentum",
                pool.submit(self.signals_map["microstructure"].calculate, ob, trades): "microstructure",
                pool.submit(self.signals_map["volatility"].calculate, candles): "volatility",
                pool.submit(self.signals_map["sentiment"].calculate,
                            funding or 0.0, [], oi): "sentiment",
                pool.submit(self.signals_map["cvd"].calculate, trades, candles): "cvd",
                pool.submit(self.signals_map["vwap"].calculate, candles, trades): "vwap",
                pool.submit(self.signals_map["whale"].calculate, trades, ob): "whale",
            }
            scores = {}
            for future in as_completed(futures, timeout=10):
                name = futures[future]
                try:
                    scores[name] = future.result()
                except Exception:
                    scores[name] = 0.0

        combiner_result = self.combiner.combine(scores, self.signals_map)
        regime = self.regime_detector.classify(scores, combiner_result, self.signals_map)
        route = self.router.route(regime, combiner_result.confidence)

        self._current_signals = scores
        self._current_signal_objects = dict(self.signals_map)
        self._current_signal_objects["_combiner_result"] = combiner_result
        self._current_signal_objects["_route_result"] = route
        self._current_signal_objects["_signals_30"] = self._current_signals_30
        self._current_combiner_result = combiner_result
        self._current_regime = regime
        self._current_route = route

        # Compute 30 Bayesian signals
        try:
            c_scores = self._candle_sig.calculate(candles)
            ob_scores = self._ob_sig.calculate(ob or {"bids": [], "asks": []},
                                               trade_size_usd=50.0)
            t_scores = self._trade_sig.calculate(trades or [], candles)
            f_scores = self._futures_sig.calculate(funding, oi, liquidations,
                                                   candles, oi_1h_ago=oi_1h_ago)
            d_scores = self._derived_sig.calculate(c_scores, t_scores)
            self._current_signals_30 = {**c_scores, **ob_scores,
                                        **t_scores, **f_scores, **d_scores}
            # Write to file so dashboard can read live signal values
            try:
                import json as _json
                _sig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "live_signals_30.json")
                with open(_sig_path, "w") as _f:
                    _json.dump(self._current_signals_30, _f)
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"30-signal compute error: {e}")

        # Push to brain queue (drop old if full)
        payload = (regime, route, scores, combiner_result,
                   self._current_signal_objects, self._current_signals_30)
        try:
            self.brain_queue.put_nowait(payload)
        except queue.Full:
            try:
                self.brain_queue.get_nowait()
            except queue.Empty:
                pass
            self.brain_queue.put_nowait(payload)

    # ------------------------------------------------------------------ #
    #  Thread 3 — Scan                                                     #
    # ------------------------------------------------------------------ #

    def _scan_thread(self):
        while self._running:
            try:
                item = self.brain_queue.get(timeout=60)
                regime, route, signals, combiner_result, signal_objects, signals_30 = item
                self._run_scan(regime, route, signals, combiner_result, signal_objects, signals_30)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"ScanThread error: {e}", exc_info=True)

    def _run_scan(self, regime, route, signals, combiner_result, signal_objects, signals_30):
        if not route.strategies:
            return

        capital = self.tracker.get_capital()
        all_opportunities = []

        with ThreadPoolExecutor(max_workers=len(route.strategies)) as pool:
            futures = {}
            for strat_name in route.strategies:
                strat = self.strategies.get(strat_name)
                if not strat:
                    continue
                f = pool.submit(
                    strat.scan, regime, route.size_mult, signals,
                    signal_objects, self.market_reader, capital, self.config
                )
                futures[f] = strat_name

            for future in as_completed(futures, timeout=15):
                try:
                    opps = future.result()
                    if opps:
                        all_opportunities.extend(opps)
                except Exception as e:
                    logger.warning(f"Strategy scan error: {e}")

        # Put each opportunity on the queue
        count = 0
        for opp in all_opportunities:
            try:
                self.opp_queue.put_nowait((opp, signals, combiner_result, signal_objects, regime, signals_30))
                count += 1
            except queue.Full:
                break

        if count > 0:
            logger.info(f"Scan found {count} opportunities (regime={regime})")

    # ------------------------------------------------------------------ #
    #  Thread 4 — Execution                                                #
    # ------------------------------------------------------------------ #

    def _execution_thread(self):
        exec_threshold = self.config["scoring"]["execution_threshold"]
        executor = SmartExecutor(
            self.market_reader, self.risk, self.scorer,
            self.tracker, self.order_manager, self.config,
        )

        while self._running:
            try:
                item = self.opp_queue.get(timeout=5)
                opp, signals, combiner_result, signal_objects, regime, signals_30 = item
            except queue.Empty:
                continue
            except Exception:
                continue

            try:
                self._cycle_count += 1
                # Score using Bayesian 30 signals
                score = self.scorer.score(
                    opp, regime, signals, combiner_result, signal_objects,
                    self.pair_ranker, signals_30=signals_30
                )
                if score < exec_threshold:
                    continue

                # Execute
                result = executor.execute(
                    opp, signals, combiner_result, signal_objects, regime
                )
                if result.executed:
                    logger.info(
                        f"EXECUTED {opp.strategy} {opp.pair} "
                        f"score={score:.3f} pnl=${result.net_pnl_usd:.4f}"
                    )
                    self._print_cycle(opp, score, regime, combiner_result)
            except Exception as e:
                logger.error(f"ExecThread error: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    #  Thread 5 — Monitor                                                  #
    # ------------------------------------------------------------------ #

    def _monitor_thread(self):
        while self._running:
            try:
                exits = self.order_manager.check_exits(self.market_reader)
                for pos, reason, exit_price in exits:
                    self._settle_position(pos, reason, exit_price)
            except Exception as e:
                logger.warning(f"MonitorThread error: {e}")
            time.sleep(10)

    def _settle_position(self, pos: dict, reason: str, exit_price: float):
        direction = pos.get("direction", "long")
        entry = pos.get("entry_price", exit_price)
        size = pos.get("size_usd", 0)

        if direction == "long":
            pnl_pct = (exit_price - entry) / entry if entry else 0
        elif direction == "short":
            pnl_pct = (entry - exit_price) / entry if entry else 0
        else:
            pnl_pct = 0.0

        fee_pct = 0.002
        net_pnl = size * (pnl_pct - fee_pct)
        result = "WIN" if net_pnl > 0 else "LOSS"

        self.order_manager.close(pos["id"])
        self.risk.record_close(pos["id"], result)
        self.pair_ranker.update(pos["pair"], result, pnl_pct)

        meta = {
            "strategy": pos["strategy"],
            "pair": pos["pair"],
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "size_usd": size,
            "net_profit_pct": round(pnl_pct - fee_pct, 6),
            "regime": pos.get("regime", ""),
            "result": result,
        }
        with self.capital_lock:
            self.tracker.update(net_pnl, meta)
        logger.info(f"CLOSED {pos['strategy']} {pos['pair']} reason={reason} pnl=${net_pnl:.4f}")

    # ------------------------------------------------------------------ #
    #  Thread 6 — Learning                                                 #
    # ------------------------------------------------------------------ #

    def _learning_thread(self):
        """Run learning engine daily at midnight UTC."""
        from learning.trade_analyzer import TradeAnalyzer
        analyzer = TradeAnalyzer(self.config)
        last_run_date = None

        while self._running:
            now = datetime.now(timezone.utc)
            if now.hour == self.config["learning"]["run_hour"] and now.date() != last_run_date:
                try:
                    logger.info("Running nightly learning engine...")
                    analyzer.daily_analysis()
                    self.combiner.reload_weights()
                    last_run_date = now.date()
                    logger.info("Learning engine complete.")
                except Exception as e:
                    logger.error(f"LearningThread error: {e}", exc_info=True)
            time.sleep(60)

    # ------------------------------------------------------------------ #
    #  Console output                                                      #
    # ------------------------------------------------------------------ #

    def _print_startup(self):
        capital = self.tracker.get_capital()
        print(f"\n{BORDER}")
        print(f"  MoneyBot STARTED — Paper Mode")
        print(f"  Capital: ${capital:.2f}")
        print(f"  Pairs: {len(self.config['pairs']['scan_universe'])}")
        print(f"  Strategies: {len(self.strategies)}")
        print(f"{BORDER}\n")

    def _print_cycle(self, opp, score, regime, cr):
        capital = self.tracker.get_capital()
        daily = self.tracker.get_daily_pnl()
        dd = self.tracker.get_drawdown()
        cb_str = "CB1 " if self.risk.cb1_active else ""
        cb_str += "CB3 " if self.risk.cb3_active else ""
        cb_str = cb_str.strip() or "CLEAR"
        s = self._current_signals
        print(f"\n{BORDER}")
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Cycle #{self._cycle_count} | {regime} (conf:{cr.confidence:.2f})")
        print(f"Capital: ${capital:.2f} ({'+'if daily>=0 else ''}{daily:.2f} today) | DD:{dd*100:.1f}%")
        print(f"Open trades: {self.risk.open_trade_count}/{self.config['risk']['max_open_trades']} | CB: {cb_str}")
        print(f"Signals: PA={s.get('price_action',0):+.2f} MOM={s.get('momentum',0):+.2f} "
              f"MIC={s.get('microstructure',0):+.2f} VOL={s.get('volatility',0):+.2f}")
        print(f"         CVD={s.get('cvd',0):+.2f} VWAP={s.get('vwap',0):+.2f} "
              f"SEN={s.get('sentiment',0):+.2f} WHL={s.get('whale',0):+.2f}")
        print(f"Best: {opp.strategy} {opp.pair} | score={score:.3f} | {opp.net_profit_pct*100:+.3f}%")
        print(BORDER)
