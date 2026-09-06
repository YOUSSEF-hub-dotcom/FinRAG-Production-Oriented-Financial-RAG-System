#!/usr/bin/env bash
# Forensic streaming-latency runner (diagnosis only). Starts infra + uvicorn,
# runs the external SSE timing client 2x, and dumps uvicorn log timing markers.
set -u
cd /home/youssef/Financial_RAG

# --- 0. Infra ---
bash scripts/_start_infra.sh

# --- 1. Kill stale uvicorn for a clean log ---
pkill -f "app.api.main" 2>/dev/null
sleep 3
pgrep -af "app.api.main" && echo "STILL RUNNING" || echo "old stopped"

# --- 2. Launch fresh uvicorn (captures clean logs) ---
setsid bash scripts/_start_uvicorn.sh > /home/youssef/Financial_RAG/uvicorn_streamdiag.log 2>&1 < /dev/null &
disown

# --- 3. Wait for health ---
code=000
for i in $(seq 1 60); do
  code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then echo "HEALTHY after ~$((i*3))s"; break; fi
  sleep 3
done
echo "health code=$code"
if [ "$code" != "200" ]; then echo "--- log tail ---"; tail -n 40 uvicorn_streamdiag.log; exit 1; fi

sleep 2

# --- 4. Run diagnostic client twice ---
for run in 1 2; do
  echo "################## DIAGNOSTIC RUN $run ##################"
  ./Financial_env/bin/python scripts/_diag_stream_timing.py $run
  sleep 2
done

echo "################## UVICORN LOG TIMING MARKERS ##################"
echo "--- 'Stream complete' lines (len of full_response_parts) ---"
grep -n "Stream complete" uvicorn_streamdiag.log
echo "--- request/retrieval/rerank/generation markers (grep) ---"
grep -inE "request|retriev|rerank|augment|context|generat|first_token|token|warm|Watchdog|memory" uvicorn_streamdiag.log | grep -ivE "Stream complete" | head -n 120
echo "--- line count of log ---"
wc -l uvicorn_streamdiag.log
