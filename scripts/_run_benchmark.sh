#!/usr/bin/env bash
set -u
cd /home/youssef/Financial_RAG
# Kill stale python processes that may hold the Qdrant lock or model.
pkill -f "app.api.main" 2>/dev/null
pkill -f "benchmark_reranker" 2>/dev/null
pkill -f "probe_boot" 2>/dev/null
sleep 3
# Start infra only if not present.
if ! pgrep -x mongod >/dev/null; then
  mkdir -p mongodb_data
  nohup mongod --dbpath /home/youssef/Financial_RAG/mongodb_data --bind_ip 127.0.0.1 > mongodb_data/mongod.log 2>&1 &
  echo "mongod pid $!"
fi
if ! pgrep -x redis-server >/dev/null; then
  nohup redis-server --bind 127.0.0.1 --port 6379 > redis.log 2>&1 &
  echo "redis pid $!"
fi
sleep 3
rm -f scripts/_bench_results.txt
echo "==================== NVIDIA-SMI before ===================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo
echo "==================== BENCHMARK ===================="
time ./Financial_env/bin/python scripts/_benchmark_reranker.py 2> scripts/_bench.err
echo "EXIT=$?"
echo
echo "==================== RESULTS ===================="
cat scripts/_bench_results.txt 2>/dev/null