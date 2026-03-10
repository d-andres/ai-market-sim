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

EXPOSE 7860

# ── Default command ───────────────────────────────────────────────────────────
# Starts the FastAPI + NiceGUI app via uvicorn.
# Hugging Face Spaces provides PORT at runtime; default to 7860 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
