# ======================================================
# 🍕 Pizza Sales Forecasting API — Dockerfile
# ======================================================
# Multi-stage build for lean production image
# Final image ~500MB instead of ~2GB
# ======================================================

# ── Stage 1: Builder ──
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ──
FROM python:3.10-slim AS runtime

LABEL maintainer="MLOps Pipeline"
LABEL description="Pizza Sales Forecasting API with XGBoost"
LABEL version="1.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/serve_model.py src/serve_model.py

# Copy trained model & data (baked into image)
COPY models/ models/
COPY data/pizza_sales_train.csv data/pizza_sales_train.csv
COPY data/pizza_sales_monitoring.csv data/pizza_sales_monitoring.csv

# Create empty __init__.py for module resolution
RUN touch src/__init__.py

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the API
CMD ["uvicorn", "src.serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
