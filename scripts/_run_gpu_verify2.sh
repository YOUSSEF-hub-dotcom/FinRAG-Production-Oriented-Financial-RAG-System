#!/usr/bin/env bash
cd /home/youssef/Financial_RAG
echo "==================== NVIDIA-SMI (BEFORE) ===================="
nvidia-smi
echo
echo "==================== NVIDIA-SMI CSV ===================="
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
echo
echo "==================== VERIFY SCRIPT ===================="
./Financial_env/bin/python scripts/_verify_gpu_reranker.py > /tmp/gpu_verif2.out 2> /tmp/gpu_verif2.err
echo "EXIT=$?"
cat /tmp/gpu_verif2.out
