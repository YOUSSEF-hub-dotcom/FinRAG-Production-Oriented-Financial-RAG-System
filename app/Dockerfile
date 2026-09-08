# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Financial RAG — Backend (FastAPI + model inference)
#
# Multi-stage production image. Build context = repository root
# (see docker-compose.yml: build.context = ., build.dockerfile = app/Dockerfile).
#
# GPU strategy: requirements.txt pins torch==2.13.0+cu130 resolved from
# https://download.pytorch.org/whl/cu130. The +cu130 wheels bundle the CUDA
# runtime libraries (nvidia-* pip packages), so NO CUDA toolkit is needed
# inside the image. The NVIDIA driver is supplied by the host via the
# container runtime (compose `gpus` block on the backend service); no
# `nvidia/cuda` base image is required.
# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# System packages needed to build/compile any sdist wheels pulled by pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (largest dependency) in its own layer so later
# requirements changes reuse this cached layer.
RUN pip install torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code only (see .dockerignore for excluded data/secrets).
COPY app ./app
COPY config ./config
COPY src ./src
COPY scripts ./scripts

# ---------------------------------------------------------------------------
# Runtime stage — slim, non-root, GPU-capable.
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/src \
    HF_HOME=/opt/hf \
    HF_HUB_CACHE=/opt/hf/models \
    HUGGINGFACE_HUB_CACHE=/opt/hf/models \
    TRANSFORMERS_CACHE=/opt/hf/models \
    NLTK_DATA=/opt/nltk_data \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf/models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

# Copy installed Python packages and app code from the builder stage.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

RUN mkdir -p /app/logs /opt/hf /opt/nltk_data \
    && chown -R app:app /app /opt/hf /opt/nltk_data

USER app
WORKDIR /app

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]