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
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()
    print("[MoneyBot] Dashboard running at http://localhost:5000")

    engine = MasterEngine("config/config.yaml")
    engine.start()
