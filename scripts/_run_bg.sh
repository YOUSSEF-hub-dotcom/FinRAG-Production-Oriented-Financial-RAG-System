#!/bin/bash
# Launcher: run a benchmark script UNBUFFERED, log to a file, in background.
cd /home/youssef/Financial_RAG
SCRIPT="$1"
LOGFILE="$2"
rm -f "$LOGFILE"
./Financial_env/bin/python -u "$SCRIPT" > "$LOGFILE" 2>&1 &
echo "started pid $!"