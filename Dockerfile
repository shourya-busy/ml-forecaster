# syntax=docker/dockerfile:1.6
ARG PYTHON_BASE=python:3.11-slim
ARG WORKER_BASE=python:3.11-slim

# ---- builder: shared wheel build ----
FROM ${PYTHON_BASE} AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ git curl libpq-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip wheel \
 && pip wheel --wheel-dir /wheels .

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
