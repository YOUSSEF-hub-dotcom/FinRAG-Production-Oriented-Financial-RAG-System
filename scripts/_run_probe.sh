#!/usr/bin/env bash
set -u
cd /home/youssef/Financial_RAG
if ! pgrep -x mongod >/dev/null; then nohup mongod --dbpath /home/youssef/Financial_RAG/mongodb_data --bind_ip 127.0.0.1 > mongodb_data/mongod.log 2>&1 & fi
if ! pgrep -x redis-server >/dev/null; then nohup redis-server --bind 127.0.0.1 --port 6379 > redis.log 2>&1 & fi
sleep 3
rm -f scripts/_bench_progress.txt
time ./Financial_env/bin/python scripts/_probe_boot.py
echo "======== PROGRESS FILE ========"
cat scripts/_bench_progress.txt
