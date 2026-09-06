#!/usr/bin/env bash
code=000
for i in $(seq 1 40); do
  code=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "HEALTHY after $((i*3))s"
    break
  fi
  sleep 3
done
echo "final code=$code"
echo "--- log tail ---"
tail -n 30 /home/youssef/Financial_RAG/uvicorn.log
