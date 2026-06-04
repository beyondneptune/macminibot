#!/usr/bin/env zsh
# Chain script: wait for current top-5 tick pull to finish, then kick off top-10
# extension (NVDA, GOOGL, TSLA, AMZN, META) for Friday's session.
#
# Designed to be run with nohup so it survives the Claude Code session closing:
#
#     cd /Users/weeoto/macminibot
#     nohup zsh scripts/chain_top10_after_top5.sh > data/chain.log 2>&1 &
#     disown
#
# Polls every 30s for the top-5 python process; when it exits, fires the top-10
# extension with a fresh clientId. Logs everything to data/chain.log.

set -u
cd /Users/weeoto/macminibot

echo "[chain $(date -u +%FT%TZ)] waiting for top-5 pull (scripts/07_nightly_tick_pull.py with TARGETS=AAPL,MSFT,ES,GLD,EURUSD) to exit..."

while pgrep -f "07_nightly_tick_pull.py" > /dev/null; do
  sleep 30
done

echo "[chain $(date -u +%FT%TZ)] top-5 exited. Sleeping 60s for IBKR connection to clear..."
sleep 60

echo "[chain $(date -u +%FT%TZ)] starting top-5 extension (NVDA,GOOGL,TSLA,AMZN,META, Friday session)"

TARGETS=NVDA,GOOGL,TSLA,AMZN,META \
SESSION_DATE=2026-05-15 \
HOURS_BACK=8 \
CLIENT_ID=24 \
PYTHONUNBUFFERED=1 \
  /Users/weeoto/macminibot/.venv/bin/python -u scripts/07_nightly_tick_pull.py \
  > data/07_top5_extension.log 2>&1

echo "[chain $(date -u +%FT%TZ)] top-5 extension exited with code $?"

# After both batches finish we have a full top-10 dollar-bar dataset for Friday.
echo "[chain $(date -u +%FT%TZ)] chain complete. Top-10 dollar bars now in equity.dollar_bars + future.dollar_bars + etf.dollar_bars + fx.dollar_bars."
