#!/usr/bin/env bash
setsid bash /home/youssef/Financial_RAG/scripts/_start_uvicorn.sh > /home/youssef/Financial_RAG/uvicorn.log 2>&1 < /dev/null &
disown
sleep 12
echo "procs:"
pgrep -af "app.api.main"
if (ss -ltn 2>/dev/null | grep -q :8000); then echo "PORT 8000 LISTENING"; else echo "port not yet"; fi
echo "log head:"
head -n 25 /home/youssef/Financial_RAG/uvicorn.log
