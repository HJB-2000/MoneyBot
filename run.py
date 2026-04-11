#!/usr/bin/env python3
"""
MoneyBot entry point.
Usage: python run.py
"""
import os
import sys
import threading

# Ensure we run from the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain.master_engine import MasterEngine


def start_dashboard():
    from dashboard.app import create_app
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    print(f"[MoneyBot] Dashboard running at http://0.0.0.0:{port}")

    engine = MasterEngine("config/config.yaml")
    engine.start()
