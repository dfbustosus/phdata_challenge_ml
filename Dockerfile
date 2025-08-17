FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_PORT=8000 \
    MODEL_DIR=/app/model \
    DEMOGRAPHICS_CSV=/app/data/zipcode_demographics.csv \
    DEFAULT_MODEL_VERSION=v2 \
    LOG_LEVEL=INFO

# System deps for pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata and source code
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/

# Install dependencies
RUN pip install --upgrade pip && pip install .

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/model /app/data && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser model/ /app/model/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Use the correct module path for the new structure
CMD ["uvicorn", "src.housing_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
