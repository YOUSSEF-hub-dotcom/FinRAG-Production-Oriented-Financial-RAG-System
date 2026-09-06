#!/usr/bin/env bash
set -e
if ! pgrep -x mongod >/dev/null; then
  mkdir -p /home/youssef/Financial_RAG/mongodb_data
  nohup mongod --dbpath /home/youssef/Financial_RAG/mongodb_data --bind_ip 127.0.0.1 > /home/youssef/Financial_RAG/mongodb_data/mongod.log 2>&1 &
  echo "mongod launched pid $!"
else
  echo "mongod already running"
fi
if ! pgrep -x redis-server >/dev/null; then
  nohup redis-server --bind 127.0.0.1 --port 6379 > /home/youssef/Financial_RAG/redis.log 2>&1 &
  echo "redis launched pid $!"
else
  echo "redis already running"
fi
sleep 3
