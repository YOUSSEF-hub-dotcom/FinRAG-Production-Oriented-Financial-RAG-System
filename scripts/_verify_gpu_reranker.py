#!/usr/bin/env python3
"""
Forensic GPU verification for the Financial_RAG reranker (READ/MEASURE only).

Uses the EXACT production reranker class and model (no separate toy model),
loads it through the same import path, and proves three levels:
  LEVEL 1: torch CUDA is available
  LEVEL 2: the actual reranker model parameters live on CUDA
  LEVEL 3: actual inference executes on that CUDA model (memory + nvidia-smi)

No production code is modified. No device/config/batch changes.
"""
import json
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/home/youssef/Financial_RAG")
sys.path.insert(0, "/home/youssef/Financial_RAG/src")
# Mirror production path setup (pipeline.py:36): src/4_retrieval on sys.path.
sys.path.insert(0, "/home/youssef/Financial_RAG/src/4_retrieval")
sys.path.insert(0, "/home/youssef/Financial_RAG/src/5_generation")

print("=" * 70)
print("LEVEL 1: PYTHON / PYTORCH / CUDA")
print("=" * 70)
import platform
import torch
print("python            :", platform.python_version())
print("pytorch           :", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available :", torch.cuda.is_available())
print("cuda.device_count :", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda.get_device_name(0):", torch.cuda.get_device_name(0))
    print("cuda capability(0)     :", torch.cuda.get_device_capability(0))
    print("cuda current_device    :", torch.cuda.current_device())
else:
    print("NO CUDA")

print()
print("=" * 70)
print("LEVEL 2 + 3: ACTUAL PRODUCTION RERANKER")
print("=" * 70)
# Import the SAME module/class the production pipeline uses.
from reranker import CrossEncoderReranker, _AUTO_DEVICE  # noqa: E402
from config.settings import CACHE_RERANKER_MODEL  # noqa: E402

print("production device override (_AUTO_DEVICE):", repr(_AUTO_DEVICE))
print("production model setting (CACHE_RERANKER_MODEL):", CACHE_RERANKER_MODEL)

reranker = CrossEncoderReranker(top_n=8)   # same constructor the pipeline uses
print("reranker._device:", repr(reranker._device))

# Force lazy init so the model is loaded exactly as production loads it.
reranker._lazy_init()
model = reranker._model   # sentence_transformers.CrossEncoder instance

print()
print("LEVEL 2: MODEL DEVICE / DTYPE")
print("  model type            :", type(model).__name__)
print("  model device attr     :", getattr(model, "device", "n/a"))
# Underlying transformer + classifier
for attr_name in ("model", "classifier"):
    sub = getattr(model, attr_name, None)
    if sub is not None:
        try:
            devs = {str(p.device) for p in sub.parameters()}
            print("  .%s params devices    : %s" % (attr_name, devs))
        except Exception as e:
            print("  .%s (params inspect fail: %s)" % (attr_name, e))

all_devs = {str(p.device) for p in model.parameters()}
buff_devs = {str(b.device) for b in model.buffers()}
first_param = next(model.parameters())
first_dtype = first_param.dtype
print("  ALL params devices    :", all_devs)
print("  ALL buffer  devices   :", buff_devs)
print("  first param device    :", first_param.device)
print("  first param dtype     :", first_dtype)
print("  is model on cuda      :", all(d.startswith("cuda") for d in all_devs))

# ---------- LEVEL 3: inference ----------
torch.cuda.empty_cache()
time.sleep(0.2)
mem_alloc_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
mem_resv_before = torch.cuda.memory_reserved() if torch.cuda.is_available() else -1

# nvidia-smi sampler in a background thread during inference
smi_samples = []
stop = threading.Event()

def smi_sampler():
    q = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout.strip()
            if out:
                smi_samples.append((time.time(), out))
        except Exception:
            pass
        time.sleep(0.15)

th = threading.Thread(target=smi_sampler, daemon=True)
th.start()

# Build a realistic batch (mimics a chunked multi-ticker rerank batch).
query = ("What were the operating margins and total net revenue for Apple, "
         "Microsoft, and NVIDIA in fiscal year 2025?")
chunks = [{"text": ("Apple reported total net revenue of $416,161 million and operating "
                    "income of $133,050 million in fiscal year 2025, an operating margin "
                    "of approximately 31.96 percent. Revenue grew year over year from the "
                    "prior fiscal period.")} for _ in range(24)]

print()
print("LEVEL 3: INFERENCE (actual predict on %d chunk pairs)" % len(chunks))
t_start = time.time()
torch.cuda.synchronize()
t0 = time.time()
scores = reranker.predict_scores(query, chunks)
torch.cuda.synchronize()
latency_ms = (time.time() - t0) * 1000
stop.set()
th.join(timeout=6)
total_ms = (time.time() - t_start) * 1000

mem_alloc_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else -1
mem_resv_after = torch.cuda.memory_reserved() if torch.cuda.is_available() else -1

print("  len(scores)          :", len(scores))
print("  scores[:3]           :", scores[:3])
print("  inference latency ms :", round(latency_ms, 1))
print("  mem_allocated before :", mem_alloc_before)
print("  mem_allocated after  :", mem_alloc_after, " (+%d)" % (mem_alloc_after - mem_alloc_before))
print("  mem_reserved before  :", mem_resv_before)
print("  mem_reserved after   :", mem_resv_after, " (+%d)" % (mem_resv_after - mem_resv_before))

print()
print("LEVEL 3: NVIDIA-SMI SAMPLES DURING INFERENCE (name, util%, memUsedMB, memTotalMB)")
if smi_samples:
    seen = {}
    for _t, line in smi_samples:
        key = line
        if key not in seen:
            seen[key] = 0
        seen[key] += 1
    for key, cnt in seen.items():
        print("  [%dx] %s" % (cnt, key))
    # max utilization observed
    import re
    utils = []
    for _t, line in smi_samples:
        parts = line.split(",")
        try:
            utils.append(int(parts[1].strip()))
        except Exception:
            pass
    if utils:
        print("  max GPU utilization %% observed: %d" % max(utils))
else:
    print("  none captured")

print()
print("CONCURRENCY NOTE: reranker._device =", reranker._device,
      "| model all-params-cuda =", all(d.startswith("cuda") for d in all_devs))
