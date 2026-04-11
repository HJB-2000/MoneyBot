FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create required directories and empty data files
RUN mkdir -p data logs reports/daily && \
    touch data/trade_log.csv data/opportunity_log.csv \
          data/regime_log.csv data/learning_log.csv \
          data/pair_rankings.csv data/decision_log.csv \
          data/training_log.csv && \
    echo "timestamp,strategy,pair,direction,entry_price,exit_price,size_usd,gross_profit_pct,fees_pct,slippage_pct,net_profit_pct,regime,confidence,score,hold_seconds,result" > data/trade_log.csv && \
    echo "timestamp,strategy,pair,regime,net_profit_pct,liquidity_ratio,confidence,latency_ms,score,executed" > data/opportunity_log.csv && \
    echo "timestamp,regime,confidence,score,atr_ratio,funding_rate,whale_buying,whale_selling,cvd_divergence,vol_spike,consolidating" > data/regime_log.csv

# Expose dashboard port
EXPOSE 5000

CMD ["python3", "-u", "run.py"]
