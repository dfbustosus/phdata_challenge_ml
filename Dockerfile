FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UVICORN_PORT=8000

# System deps for pandas/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project metadata and source, then install
COPY pyproject.toml README.md LICENSE /app/
COPY src/ /app/src/
RUN pip install --upgrade pip && pip install .

# Create non-root user
RUN useradd -m appuser
USER appuser

# Copy source
COPY --chown=appuser:appuser data/ /app/data/
# Create model directory (artifacts should be mounted at runtime)
RUN mkdir -p /app/model

EXPOSE 8000

CMD ["uvicorn", "housing_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
