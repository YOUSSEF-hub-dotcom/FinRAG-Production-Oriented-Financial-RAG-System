#!/usr/bin/env bash
pkill -f "app.api.main" 2>/dev/null
sleep 3
pgrep -af "app.api.main" && echo "STILL RUNNING" || echo "old stopped"
setsid bash /home/youssef/Financial_RAG/scripts/_start_uvicorn.sh > /home/youssef/Financial_RAG/uvicorn.log 2>&1 < /dev/null &
disown
sleep 3
echo "relaunched"
