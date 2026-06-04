#!/usr/bin/env zsh
# Wait for [GOOGL] DONE in the extension log, then kill the sequential chain
# (and its current python child, which would otherwise start META next) and
# spawn META, NVDA, TSLA as 3 parallel pulls on separate clientIds.

set -u
cd /Users/weeoto/macminibot

echo "[parallel-trigger $(date -u +%FT%TZ)] polling for [GOOGL] DONE..."

while ! grep -q "\[GOOGL\] DONE" data/07_top5_extension.log 2>/dev/null; do
  sleep 5
done

echo "[parallel-trigger $(date -u +%FT%TZ)] GOOGL DONE detected, stopping chain + child"

CHAIN_PID=$(pgrep -f "chain_top10_after_top5.sh" | head -1)
if [ -n "${CHAIN_PID:-}" ]; then
  PYTHON_PID=$(pgrep -P "$CHAIN_PID" | head -1)
  [ -n "${PYTHON_PID:-}" ] && kill "$PYTHON_PID" 2>/dev/null && echo "  killed python PID $PYTHON_PID"
  kill "$CHAIN_PID" 2>/dev/null && echo "  killed chain PID $CHAIN_PID"
fi

# Brief settle for IBKR connection cleanup
sleep 5

echo "[parallel-trigger $(date -u +%FT%TZ)] spawning META(cid=25), NVDA(cid=26), TSLA(cid=27) in parallel"

nohup env TARGETS=META SESSION_DATE=2026-05-15 HOURS_BACK=8 CLIENT_ID=25 PYTHONUNBUFFERED=1 \
  /Users/weeoto/macminibot/.venv/bin/python -u scripts/07_nightly_tick_pull.py \
  > data/07_META_parallel.log 2>&1 &
disown
echo "  META PID $!"

nohup env TARGETS=NVDA SESSION_DATE=2026-05-15 HOURS_BACK=8 CLIENT_ID=26 PYTHONUNBUFFERED=1 \
  /Users/weeoto/macminibot/.venv/bin/python -u scripts/07_nightly_tick_pull.py \
  > data/07_NVDA_parallel.log 2>&1 &
disown
echo "  NVDA PID $!"

nohup env TARGETS=TSLA SESSION_DATE=2026-05-15 HOURS_BACK=8 CLIENT_ID=27 PYTHONUNBUFFERED=1 \
  /Users/weeoto/macminibot/.venv/bin/python -u scripts/07_nightly_tick_pull.py \
  > data/07_TSLA_parallel.log 2>&1 &
disown
echo "  TSLA PID $!"

echo "[parallel-trigger $(date -u +%FT%TZ)] all 3 parallel pulls spawned. trigger exiting."
