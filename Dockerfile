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

# Install all core deps including rank-bm25, pyarrow, matplotlib
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[viz]"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BENCHMARK_HW_DEBUG=0

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source (scripts/ must be present — study_runner.py is loaded at runtime)
COPY . .

# Expose default explorer port
EXPOSE 8501

# Default: run the full doctor check so users get hardware-aware instructions
# Override with: docker run memtuner memtuner study --mode quick
ENTRYPOINT ["memtuner"]
CMD ["doctor"]
