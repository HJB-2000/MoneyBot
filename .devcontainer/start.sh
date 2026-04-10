#!/bin/bash
# MoneyBot — Training Launcher
# Usage: bash .devcontainer/start.sh
# Run this ONCE after opening the Codespace.
# It runs tests, starts keep-alive server, starts training.
# You can close your laptop after the "All processes running" message.

set -e

# ════════════════════════════════════════════
# PHASE 0 — ENVIRONMENT CHECK
# ════════════════════════════════════════════

echo "╔══════════════════════════════════════════╗"
echo "║   MoneyBot Training — Codespace Start   ║"
echo "╚══════════════════════════════════════════╝"
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')]"
echo "Machine: $(hostname)"
echo "Python:  $(python3 --version 2>&1)"
echo "Disk:    $(df -h . | tail -1)"
echo "RAM:     $(free -h | grep Mem)"
echo ""

# Check Python >= 3.10
PYTHON_OK=$(python3 -c "import sys; print(1 if sys.version_info >= (3,10) else 0)" 2>/dev/null || echo "0")
if [ "$PYTHON_OK" != "1" ]; then
  echo "❌ Python 3.10+ required. Current: $(python3 --version 2>&1)"
  exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
pip install fastapi uvicorn --quiet
echo "✅ Dependencies ready"
echo ""

# Check Gmail env vars
if [ -z "$GMAIL_SENDER" ]; then
  echo "⚠  Email notifications are DISABLED."
  echo "   To enable: set these env vars before running start.sh:"
  echo "   export GMAIL_SENDER='your@gmail.com'"
  echo "   export GMAIL_PASSWORD='xxxx xxxx xxxx xxxx'"
  echo "   export GMAIL_RECEIVER='your@gmail.com'"
  echo "   Or add them to GitHub Codespace Secrets (recommended)."
  echo "   Continuing without email notifications..."
else
  echo "✅ Gmail notifications enabled"
fi
echo ""

# ════════════════════════════════════════════
# PHASE 1 — RUN TESTS
# ════════════════════════════════════════════

mkdir -p logs

echo "Running 164 unit tests before starting training..."
set +e
python3 -m pytest unit_test/ -v --tb=short 2>&1 | tee logs/test_results.txt
TEST_EXIT=$?
set -e

if [ $TEST_EXIT -ne 0 ]; then
  echo ""
  echo "Last 30 lines of test output:"
  tail -30 logs/test_results.txt
  python3 -c "
from scripts.notify import Notifier
import pathlib
n = Notifier()
log = pathlib.Path('logs/test_results.txt').read_text()
n.training_failed('unit_tests', 'Tests failed before training started', log[-3000:])
" 2>/dev/null || true
  echo "❌ Tests failed. Training aborted. Fix tests first."
  exit 1
fi

TEST_SUMMARY=$(grep -E "passed|failed" logs/test_results.txt | tail -1 || echo "tests complete")
echo "✅ All tests passed: $TEST_SUMMARY"

# Update server state (server may not be up yet — ignore errors)
curl -s -X POST http://localhost:8080/update \
  -H "Content-Type: application/json" \
  -d "{\"tests_status\": \"passed\", \"tests_detail\": \"$TEST_SUMMARY\"}" 2>/dev/null || true

echo ""

# ════════════════════════════════════════════
# PHASE 2 — START KEEP-ALIVE SERVER
# ════════════════════════════════════════════

pkill -f "scripts/server.py" 2>/dev/null || true
sleep 1

nohup python3 scripts/server.py > logs/server.log 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > logs/server.pid

echo "Waiting for server to start..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
  echo "❌ Server failed to start. Check logs/server.log"
  cat logs/server.log
  exit 1
fi

echo "✅ Keep-alive server running (PID $SERVER_PID)"
echo ""
echo "⚠  IMPORTANT: Make port 8080 PUBLIC now:"
echo "   Ports tab → right-click 8080 → Port Visibility → Public"
echo "   Then set up a free ping at cron-job.org:"
echo "   URL: [your-codespace-url]:8080/health  (every 20 minutes)"
echo ""
echo "   Dashboard: http://localhost:8080/"
echo "   You can close this terminal tab — training runs in background."
echo ""

sleep 5

# ════════════════════════════════════════════
# PHASE 3 — RUN TRAINING IN BACKGROUND
# ════════════════════════════════════════════

curl -s -X POST http://localhost:8080/update \
  -H "Content-Type: application/json" \
  -d '{"training_status": "running", "training_step": "starting"}' > /dev/null 2>&1 || true

python3 -c "
from scripts.notify import Notifier
n = Notifier()
n.training_started([
  'Step 1: Download 2 years Binance data (~35 min)',
  'Step 2: Generate 30 signals on history (~12 min)',
  'Step 3: Generate win/loss labels (~3 min)',
  'Step 4: Train likelihood ratios (~2 min)',
  'Step 5: Build signal correlation matrix (~1 min)',
  'Step 6: Validate and save outputs (~1 min)',
])
" 2>/dev/null || true

echo "🚀 Training started. You can close your laptop now."
echo "   Check status at: http://localhost:8080/"
echo "   You will receive a Gmail notification when done."
echo ""

nohup bash -c '
  START_TIME=$(date +%s)

  python3 scripts/train_bayesian.py >> logs/training.log 2>&1
  EXIT_CODE=$?

  END_TIME=$(date +%s)
  DURATION=$(( (END_TIME - START_TIME) / 60 ))

  if [ $EXIT_CODE -eq 0 ]; then
    curl -s -X POST http://localhost:8080/update \
      -H "Content-Type: application/json" \
      -d "{\"training_status\": \"complete\", \"training_step\": \"done\"}" > /dev/null 2>&1 || true

    python3 -c "
import json, pathlib
from scripts.notify import Notifier

n = Notifier()
files = {}
for f in [\"data/likelihood_ratios.json\",
          \"data/signal_correlations.json\",
          \"data/base_rates.json\"]:
    p = pathlib.Path(f)
    files[f] = f\"{p.stat().st_size / 1024:.1f} KB\" if p.exists() else \"MISSING\"

results = {
  \"duration_minutes\": '"$DURATION"',
  \"files\": files,
  \"exit_code\": 0,
}
n.training_complete(results)
" 2>/dev/null || true

    echo ""
    echo "✅ Training complete in ${DURATION} minutes."
    echo "   Run: python3 run.py"

  else
    curl -s -X POST http://localhost:8080/update \
      -H "Content-Type: application/json" \
      -d "{\"training_status\": \"failed\", \"training_error\": \"Exit code $EXIT_CODE\"}" > /dev/null 2>&1 || true

    python3 -c "
import pathlib
from scripts.notify import Notifier

n = Notifier()
log = pathlib.Path(\"logs/training.log\").read_text() if pathlib.Path(\"logs/training.log\").exists() else \"No log found\"
n.training_failed(\"train_bayesian.py\", \"Script exited with code $EXIT_CODE\", log[-4000:])
" 2>/dev/null || true

    echo ""
    echo "❌ Training failed after ${DURATION} minutes."
    echo "   Check: cat logs/training.log"
  fi
' >> logs/training_wrapper.log 2>&1 &

TRAINING_PID=$!
echo $TRAINING_PID > logs/training.pid
echo "   Training PID: $TRAINING_PID"
echo "   Full log: tail -f logs/training.log"
echo ""
echo "═══════════════════════════════════════════"
echo "  All processes running. Terminal is free."
echo "  Close it, close your laptop, go rest."
echo "═══════════════════════════════════════════"
