# ===========================================================================
# LFL Backend API — Dockerfile
# ===========================================================================
# Minimal Python image for the FastAPI backend.
#
# Build:
#   docker build -t lfl-api .
#
# Run standalone (without docker-compose):
#   docker run -p 8000:8000 --env-file .env lfl-api
# ===========================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System-level deps (numpy/scipy build + healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY backend/ backend/
COPY domain_packs/ domain_packs/
COPY scripts/ scripts/
COPY test_data/ test_data/

# Runtime data directory (memory DB, pattern index, graph outbox). No dataset
# ships with the image; mount or generate one.
RUN mkdir -p data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
