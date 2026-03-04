# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Keeps Python output unbuffered so logs stream straight to Docker.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer independently of
# source-code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

# ── Default command ───────────────────────────────────────────────────────────
# Starts the FastAPI app via uvicorn.
# --reload watches /app for source changes (mirrors the volume mount in compose).
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
