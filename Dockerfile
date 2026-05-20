# syntax=docker/dockerfile:1.6
ARG PYTHON_BASE=python:3.11-slim
ARG WORKER_BASE=python:3.11-slim
# `cpu`, `cu121`, `cu124`, … — PyTorch index variant.
# Default is CPU-only, which avoids pulling in ~3-4 GB of NVIDIA CUDA libs.
# The GPU compose override (docker-compose.gpu.yml) sets this to cu121.
ARG TORCH_INDEX=cpu

# ---- builder: shared wheel build ----
FROM ${PYTHON_BASE} AS builder
ARG TORCH_INDEX
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git curl libpq-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
# Resolve torch from the PyTorch index (CPU-only by default) and fall back
# to PyPI for everything else. Without this, pip pulls CUDA-bundled torch
# wheels even when the GPU runtime is absent.
RUN pip install --upgrade pip wheel \
 && pip wheel --wheel-dir /wheels \
      --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" \
      --extra-index-url "https://pypi.org/simple" \
      .

# ---- runtime base for API (slim) ----
FROM ${PYTHON_BASE} AS api-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FORECASTER_CONFIG_DIR=/app/config
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels forecaster \
 && rm -rf /wheels
COPY config /app/config
COPY scripts /app/scripts
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/healthz || exit 1
CMD ["forecaster-api"]

# ---- runtime base for worker / scheduler (same image, different cmd) ----
FROM ${WORKER_BASE} AS worker-runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FORECASTER_CONFIG_DIR=/app/config
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels forecaster \
 && rm -rf /wheels
COPY config /app/config
COPY scripts /app/scripts
CMD ["forecaster-worker"]
