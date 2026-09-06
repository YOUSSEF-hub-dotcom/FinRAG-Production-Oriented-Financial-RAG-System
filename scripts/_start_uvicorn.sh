#!/usr/bin/env bash
cd /home/youssef/Financial_RAG
export PYTHONPATH=/home/youssef/Financial_RAG
exec ./Financial_env/bin/python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
