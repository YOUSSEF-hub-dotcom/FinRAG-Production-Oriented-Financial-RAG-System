#!/usr/bin/env bash
cd /home/youssef/Financial_RAG
./Financial_env/bin/python scripts/_verify_gpu_reranker.py > /tmp/gpu_verif.out 2> /tmp/gpu_verif.err
echo "EXIT=$?"
echo "==================== STDOUT ===================="
cat /tmp/gpu_verif.out
echo "==================== STDERR (filtered) ===================="
grep -vE "Warning|warn|FutureWarning|UserWarning|Downloading|Fetching|HuggingFace|it/s|s/it|Running inference|proxied" /tmp/gpu_verif.err | tail -30
