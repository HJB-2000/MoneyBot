"""
MoneyBot Dashboard — Flask app.
Reads bot output files every request and serves JSON + HTML.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import time
from pathlib import Path

_START_TIME = time.time()

from flask import Flask, jsonify, render_template


# ── paths (resolved relative to project root, not this file) ─────────────────
ROOT = Path(__file__).resolve().parents[1]
DB_PATH        = ROOT / "moneyBot.db"
TRADE_LOG      = ROOT / "data" / "trade_log.csv"
REGIME_LOG     = ROOT / "data" / "regime_log.csv"
OPP_LOG        = ROOT / "data" / "opportunity_log.csv"
WEIGHTS_FILE   = ROOT / "data" / "signal_weights.json"
PAIR_CSV       = ROOT / "data" / "pair_rankings.csv"
BOT_LOG          = ROOT / "logs" / "bot.log"
LIVE_SIGNALS_30  = ROOT / "data" / "live_signals_30.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv_tail(path: Path, n: int) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def _capital_from_db() -> dict:
    if not DB_PATH.exists():
        return {"balance": 0.0, "history": [], "milestones": []}
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT balance, updated_at FROM capital ORDER BY id ASC"
        ).fetchall()
        milestones = [r[0] for r in conn.execute("SELECT value FROM milestones").fetchall()]
        conn.close()
    except Exception:
        return {"balance": 0.0, "history": [], "milestones": []}

    if not rows:
        return {"balance": 0.0, "history": [], "milestones": milestones}

    history = [{"balance": r[0], "ts": r[1]} for r in rows[-100:]]
    return {
        "balance": rows[-1][0],
        "history": history,
        "milestones": milestones,
    }


def _daily_pnl_and_drawdown(history: list[dict]) -> tuple[float, float]:
    if len(history) < 2:
        return 0.0, 0.0
    # Daily P&L: difference from first entry today
    today_prefix = history[-1]["ts"][:10] if history else ""
    today_rows = [r for r in history if r["ts"].startswith(today_prefix)]
    daily_pnl = (today_rows[-1]["balance"] - today_rows[0]["balance"]) if len(today_rows) >= 2 else 0.0
    # Drawdown from peak
    peak = max(r["balance"] for r in history)
    current = history[-1]["balance"]
    drawdown = max(0.0, (peak - current) / peak) if peak > 0 else 0.0
    return round(daily_pnl, 4), round(drawdown * 100, 2)


def _current_regime() -> dict:
    rows = _read_csv_tail(REGIME_LOG, 1)
    if not rows:
        return {"regime": "UNKNOWN", "confidence": 0.0, "score": 0.0, "ts": "—"}
    r = rows[-1]
    return {
        "regime":     r.get("regime", "UNKNOWN"),
        "confidence": float(r.get("confidence", 0)),
        "score":      float(r.get("score", 0)),
        "atr_ratio":  float(r.get("atr_ratio", 1)),
        "funding_rate": float(r.get("funding_rate", 0)),
        "ts":         r.get("timestamp", "—"),
    }


def _signal_weights() -> dict:
    if WEIGHTS_FILE.exists() and WEIGHTS_FILE.stat().st_size > 0:
        try:
            return json.loads(WEIGHTS_FILE.read_text())
        except Exception:
            pass
    return {}


def _recent_trades(n: int = 20) -> list[dict]:
    rows = _read_csv_tail(TRADE_LOG, n)
    out = []
    for r in reversed(rows):
        out.append({
            "ts":         r.get("timestamp", "")[:19],
            "strategy":   r.get("strategy", ""),
            "pair":       r.get("pair", ""),
            "result":     r.get("result", ""),
            "net_pnl":    _safe_float(r.get("net_pnl_usd")),
            "net_pct":    _safe_float(r.get("net_profit_pct")),
            "regime":     r.get("regime", ""),
            "confidence": _safe_float(r.get("confidence")),
            "score":      _safe_float(r.get("score")),
        })
    return out


def _recent_opps(n: int = 10) -> list[dict]:
    rows = _read_csv_tail(OPP_LOG, n)
    out = []
    for r in reversed(rows):
        out.append({
            "ts":         r.get("timestamp", "")[:19],
            "strategy":   r.get("strategy", ""),
            "pair":       r.get("pair", ""),
            "regime":     r.get("regime", ""),
            "profit_pct": _safe_float(r.get("net_profit_pct")),
            "score":      _safe_float(r.get("score")),
            "executed":   r.get("executed", "0") == "1",
        })
    return out


def _bot_log_tail(n: int = 80) -> list[str]:
    if not BOT_LOG.exists() or BOT_LOG.stat().st_size == 0:
        return ["(no log yet)"]
    lines = BOT_LOG.read_text(errors="replace").splitlines()
    # Filter out noisy dashboard HTTP request lines — keep only meaningful bot activity
    filtered = [l for l in lines if "GET /api/status" not in l and "GET / HTTP" not in l]
    return filtered[-n:]


def _confidence_history(n: int = 100) -> list[dict]:
    """Last N regime rows for confidence-over-time chart."""
    rows = _read_csv_tail(REGIME_LOG, n)
    out = []
    for r in rows:
        out.append({
            "ts":         r.get("timestamp", "")[:19],
            "confidence": _safe_float(r.get("confidence")),
            "score":      _safe_float(r.get("score")),
            "regime":     r.get("regime", "UNKNOWN"),
        })
    return out


def _signal_history() -> dict:
    """Last regime row as signal snapshot."""
    rows = _read_csv_tail(REGIME_LOG, 1)
    if not rows:
        return {}
    r = rows[-1]
    return {
        "regime":       r.get("regime", "UNKNOWN"),
        "confidence":   _safe_float(r.get("confidence")),
        "score":        _safe_float(r.get("score")),
        "atr_ratio":    _safe_float(r.get("atr_ratio", 1)),
        "funding_rate": _safe_float(r.get("funding_rate", 0)),
        "ts":           r.get("timestamp", "—"),
    }


def _pair_rankings() -> list[dict]:
    rows = _read_csv_tail(PAIR_CSV, 50)
    out = []
    for r in rows:
        out.append({
            "pair":    r.get("pair", ""),
            "tier":    r.get("tier", "B"),
            "wins":    int(r.get("wins", 0) or 0),
            "losses":  int(r.get("losses", 0) or 0),
            "trades":  int(r.get("total_trades", 0) or 0),
            "win_rate": _safe_float(r.get("win_rate")),
        })
    return sorted(out, key=lambda x: x["win_rate"], reverse=True)


def _safe_float(val) -> float:
    try:
        return round(float(val), 6)
    except (TypeError, ValueError):
        return 0.0


# ── factory ───────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/uptime")
    def uptime():
        secs = int(time.time() - _START_TIME)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return jsonify({"seconds": secs, "display": f"{h:02d}:{m:02d}:{s:02d}"})

    @app.route("/api/signals30")
    def signals30():
        if LIVE_SIGNALS_30.exists() and LIVE_SIGNALS_30.stat().st_size > 0:
            try:
                return jsonify(json.loads(LIVE_SIGNALS_30.read_text()))
            except Exception:
                pass
        return jsonify({})

    @app.route("/api/logs")
    def logs():
        return jsonify({"lines": _bot_log_tail(80)})

    @app.route("/api/confidence")
    def confidence():
        return jsonify({"history": _confidence_history(100)})

    @app.route("/api/signals")
    def signals():
        return jsonify(_signal_history())

    @app.route("/api/status")
    def status():
        cap = _capital_from_db()
        daily_pnl, drawdown = _daily_pnl_and_drawdown(cap["history"])
        trades = _recent_trades(20)

        # Win/loss stats from recent trades
        wins   = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        total  = wins + losses
        win_rate = round(wins / total, 4) if total > 0 else 0.0
        total_pnl = round(sum(t["net_pnl"] for t in trades), 4)

        return jsonify({
            "capital": {
                "balance":    round(cap["balance"], 2),
                "daily_pnl":  daily_pnl,
                "drawdown":   drawdown,
                "milestones": cap["milestones"],
                "history":    cap["history"],
            },
            "stats": {
                "wins":     wins,
                "losses":   losses,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
            },
            "regime":      _current_regime(),
            "weights":     _signal_weights(),
            "trades":      trades,
            "opportunities": _recent_opps(10),
            "pairs":       _pair_rankings(),
        })

    return app
