#!/usr/bin/env bash
set -u
cd /home/youssef/Financial_RAG

if ! fuser 27017/tcp >/dev/null 2>&1; then
  echo "starting mongod..."
  rm -f /home/youssef/Financial_RAG/mongodb_data/mongod.lock
  mongod --dbpath /home/youssef/Financial_RAG/mongodb_data --fork --logpath /home/youssef/Financial_RAG/mongod.log >/dev/null 2>&1
  sleep 4
fi
if ! fuser 6379/tcp >/dev/null 2>&1; then
  echo "starting redis..."
  redis-server --daemonize yes >/dev/null 2>&1
  sleep 2
fi

fuser -k 8000/tcp 2>/dev/null
sleep 2
setsid bash /home/youssef/Financial_RAG/scripts/_start_uvicorn.sh > uvicorn.log 2>&1 < /dev/null &
for i in $(seq 1 60); do
  h=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health)
  if [ "$h" = "200" ]; then echo "SERVER_UP after ${i}s"; break; fi
  sleep 1
done
export PYTHONPATH=src:.
./Financial_env/bin/python scripts/_audit_subset.py > subset_run.log 2>&1
echo "SUBSET_EXIT=$?"
tail -6 subset_run.log
fuser -k 8000/tcp 2>/dev/null
echo SERVER_STOPPED
