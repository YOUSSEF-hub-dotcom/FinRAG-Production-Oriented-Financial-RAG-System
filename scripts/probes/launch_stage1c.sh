#!/usr/bin/env bash
set -e
cd ~/Financial_RAG
rm -f /tmp/stage1c_run70b.pid
JUDGE_MAX_CONTEXT_CHARS=8000 nohup Financial_env/bin/python run_stage_1c_resilient.py > /tmp/stage1c_run70b.log 2>&1 &
echo $! > /tmp/stage1c_run70b.pid
disown
sleep 2
echo "pid: $(cat /tmp/stage1c_run70b.pid)"
