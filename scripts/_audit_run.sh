#!/usr/bin/env bash
set -u
cd /home/youssef/Financial_RAG
fuser -k 8000/tcp 2>/dev/null
sleep 2
./Financial_env/bin/python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 < /dev/null &
for i in $(seq 1 70); do
  h=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
  if [ "$h" = "200" ]; then echo "SERVER_UP after ${i}s"; break; fi
  sleep 1
done
h_final=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
if [ "$h_final" != "200" ]; then echo "SERVER_NOT_READY (health=$h_final) - aborting audit"; exit 1; fi
export PYTHONPATH=src:.
./Financial_env/bin/python -u run_audit.py > audit_run3.log 2>&1
echo "AUDIT_EXIT=$?"
grep -E "PASS:|FAIL:|PARTIAL:|Total" audit_run3.log | tail -12
fuser -k 8000/tcp 2>/dev/null
echo SERVER_STOPPED
