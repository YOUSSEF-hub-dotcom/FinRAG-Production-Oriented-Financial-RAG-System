#!/usr/bin/env bash
set -u
cd /home/youssef/Financial_RAG
setsid bash scripts/_start_uvicorn.sh > uvicorn.log 2>&1 < /dev/null &
for i in $(seq 1 40); do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  if [ "$code" = "200" ]; then echo "server up after $i tries"; break; fi
  sleep 3
done
curl -s -o /dev/null -w "final_health=%{http_code}\n" http://127.0.0.1:8000/health
rm -f audit_run2.log
./Financial_env/bin/python -u run_audit.py > audit_run2.log 2>&1
echo AUDIT_DONE_RC=$?
