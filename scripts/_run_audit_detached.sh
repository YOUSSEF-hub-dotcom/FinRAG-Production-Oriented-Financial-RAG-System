#!/usr/bin/env bash
cd /home/youssef/Financial_RAG
export PYTHONPATH=src:.
exec ./Financial_env/bin/python -u run_audit.py
