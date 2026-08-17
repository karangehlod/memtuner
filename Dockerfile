# GPU passthrough note:
#   NVIDIA GPU:  docker run --gpus all ...
#   CPU only:    docker run ...          (no flag needed)
#
# The container itself does not require CUDA; sentence-transformers will
# detect and use GPU automatically when the NVIDIA runtime is available.

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml ./
# Create a minimal package stub so pip can resolve the project metadata
RUN mkdir -p benchmark && touch benchmark/__init__.py

# Install core deps + matplotlib (viz); skip database and explorer optionals
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        ".[bm25]" \
        "matplotlib>=3.7"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BENCHMARK_HW_DEBUG=0

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source
COPY . .

ENTRYPOINT ["python", "study_runner.py"]
