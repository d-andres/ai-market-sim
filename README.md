---
title: ai-market-sim
emoji: "🛒"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ai-market-sim

Tile-based fantasy market simulation with a FastAPI backend and NiceGUI frontend.

## Hugging Face Spaces (Docker)

This repository is configured to run as a Docker Space.

Requirements already in this repo:

- A `Dockerfile` at project root
- App server binding to `0.0.0.0`
- Runtime port support via `PORT` environment variable

### How to deploy

1. Create a new Space on Hugging Face.
2. Select `Docker` as the Space SDK.
3. Connect this GitHub repository (or push this code to the Space repo).
4. Ensure your Space has enough hardware for Python + NiceGUI (CPU basic is fine for this stage).
5. Trigger a build by pushing to your default branch.

Once the build succeeds, your app will be served from the Space URL.

## Local run

### With Docker

```bash
docker build -t ai-market-sim .
docker run --rm -p 7860:7860 ai-market-sim
```

Open `http://localhost:7860`.

### Without Docker

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Endpoints

- `GET /health` - liveness check
- `GET /world` - current map state in JSON
- `GET /` - NiceGUI dashboard
