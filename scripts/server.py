"""
MoneyBot — Keep-alive server.
Keeps the Codespace awake by responding to pings.
Serves a status dashboard at / readable on a phone.
"""
from __future__ import annotations

import os
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── global state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
server_state: dict = {
    "started_at":          None,
    "pings_received":      0,
    "last_ping_at":        None,
    "training_status":     "idle",
    "training_started_at": None,
    "training_step":       None,
    "training_error":      None,
    "tests_status":        None,
    "tests_detail":        None,
}

app = FastAPI()


# ── helpers ───────────────────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uptime() -> float:
    if server_state["started_at"] is None:
        return 0.0
    started = datetime.fromisoformat(server_state["started_at"])
    return (datetime.now(timezone.utc) - started).total_seconds()


def _disk_free_gb() -> float:
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        return round(free / (1024 ** 3), 1)
    except Exception:
        return 0.0


def _tail_log(path: str, n: int = 30) -> str:
    p = Path(path)
    if not p.exists():
        return "(no log yet)"
    lines = p.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    with _lock:
        server_state["pings_received"] += 1
        server_state["last_ping_at"] = _now()
        return {
            "status":   "alive",
            "pings":    server_state["pings_received"],
            "training": server_state["training_status"],
        }


@app.get("/status")
def status():
    with _lock:
        data = dict(server_state)
    data["uptime_seconds"]  = round(_uptime(), 1)
    data["python_version"]  = sys.version.split()[0]
    data["disk_free_gb"]    = _disk_free_gb()
    return JSONResponse(data)


@app.post("/update")
async def update(request: Request):
    body = await request.json()
    with _lock:
        for k, v in body.items():
            if k in server_state:
                server_state[k] = v
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with _lock:
        state = dict(server_state)

    status_val = state["training_status"]
    color_map  = {
        "idle":     "#64748b",
        "running":  "#22c55e",
        "complete": "#3b82f6",
        "failed":   "#ef4444",
    }
    status_color = color_map.get(status_val, "#64748b")

    uptime_s = _uptime()
    uptime_h = int(uptime_s // 3600)
    uptime_m = int((uptime_s % 3600) // 60)

    log_text = _tail_log("logs/training.log", 30)
    error_html = ""
    if state["training_error"]:
        error_html = f'<div style="background:#7f1d1d;color:#fca5a5;padding:12px;border-radius:8px;margin:12px 0;font-family:monospace;white-space:pre-wrap">{state["training_error"]}</div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="30">
<title>MoneyBot Status</title>
<style>
  body{{background:#0d0f14;color:#e2e8f0;font-family:system-ui,sans-serif;font-size:15px;padding:16px;max-width:600px;margin:0 auto}}
  h1{{color:#06b6d4;font-size:20px;margin-bottom:4px}}
  .badge{{display:inline-block;padding:4px 14px;border-radius:20px;font-weight:700;font-size:13px;background:{status_color}22;color:{status_color};border:1px solid {status_color}44}}
  .row{{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #1f2535}}
  .label{{color:#64748b}}
  pre{{background:#161a22;border:1px solid #1f2535;border-radius:8px;padding:12px;font-size:11px;overflow-x:auto;white-space:pre-wrap;word-break:break-all;max-height:300px;overflow-y:auto}}
  .refresh{{color:#64748b;font-size:11px;text-align:right;margin-top:8px}}
</style>
</head>
<body>
<h1>⚡ MoneyBot</h1>
<p style="margin-bottom:16px"><span class="badge">{status_val.upper()}</span></p>
{error_html}
<div class="row"><span class="label">Step</span><span>{state["training_step"] or "—"}</span></div>
<div class="row"><span class="label">Tests</span><span>{state["tests_status"] or "—"} {state["tests_detail"] or ""}</span></div>
<div class="row"><span class="label">Uptime</span><span>{uptime_h}h {uptime_m}m</span></div>
<div class="row"><span class="label">Pings received</span><span>{state["pings_received"]}</span></div>
<div class="row"><span class="label">Last ping</span><span>{(state["last_ping_at"] or "—")[:19]}</span></div>
<div class="row"><span class="label">Disk free</span><span>{_disk_free_gb()} GB</span></div>
<div class="row"><span class="label">Python</span><span>{sys.version.split()[0]}</span></div>
<h2 style="margin-top:20px;font-size:14px;color:#64748b">Training log (last 30 lines)</h2>
<pre>{log_text}</pre>
<p class="refresh">Auto-refreshes every 30 seconds</p>
</body>
</html>"""
    return html


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with _lock:
        server_state["started_at"] = _now()
    port = int(os.environ.get("PORT", 8080))
    print(f"[MoneyBot server] Dashboard → http://localhost:{port}/")
    print(f"[MoneyBot server] Health    → http://localhost:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
