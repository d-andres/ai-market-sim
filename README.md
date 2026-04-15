---
title: ai-market-sim
emoji: "⚔️"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# ai-market-sim

Tile-based fantasy market simulation where autonomous AI agents (guards, shopkeepers, players) live, trade, fight, and negotiate inside a persistent physical world. Every action — movement, combat, item use, conversation — is governed by spatial coordinates, field-of-vision, and real-time inventory management. AI brains are powered by LLMs via a local Ollama server.

## Features

- **10 action types**: move (8-directional), wait, propose\_trade, pick\_up, place, converse, attack, equip, unequip, use\_item
- **Non-blocking LLM planning**: background `ThreadPoolExecutor`; tick loop never blocks on model latency
- **Conversation packets**: one LLM call generates opener + follow-up lines consumed over multiple ticks
- **Combat system**: melee attacks, critical hits, equipment slots (head/chest/legs/hand/offhand), death with loot drops
- **Crime & justice**: witness-based theft/assault reporting, guard confrontation escalation (warning → fine → lethal), self-defense window, infamy/fame tracking
- **Dialogue social inference**: LLM-assessed likeness/suspicion deltas from conversation content
- **Adaptive fallback**: AI-authored strategy profiles that refresh periodically, used when primary planning fails
- **20 items**: weapons, armour, consumables, food/drink, and the legendary Lost Crown (10,000g)
- **NiceGUI dashboard**: dark retro terminal aesthetic, ASCII map, colour-coded event log, play/pause controls

## Local run

### With Docker

```bash
docker build -t ai-market-sim:local .
docker run --rm -p 7860:7860 \
	--env-file .env \
	--add-host=host.docker.internal:host-gateway \
	ai-market-sim:local
```

Open `http://localhost:7860`.

### Without Docker

```bash
pip install -r requirements.txt
python main.py
```

### Environment variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `qwen3:4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `ENABLE_AI` | `true` | Toggle AI brains on/off |
| `TICK_RATE` | `2.0` | Seconds per simulation tick |
| `MANUAL_TICK` | `false` | If `true`, ticks only advance via UI button |
| `BLOCK_ON_INITIAL_PLANS` | `true` | Wait for all initial LLM plans before going live |

## Hugging Face Spaces (Docker)

This repository is configured to run as a Docker Space.

1. Create a new Space on Hugging Face and select `Docker` as the SDK.
2. Connect this GitHub repository (or push this code to the Space repo).
3. Set environment variables in Settings → Variables and Secrets.
4. Trigger a build by pushing to your default branch.

## Endpoints

- `GET /health` — liveness check
- `GET /world` — full world state snapshot (JSON)
- `GET /` — NiceGUI dashboard

## Tests

```bash
python -m pytest tests/       # or run each file directly:
python tests/test_combat.py
python tests/test_crime_system.py
python tests/test_dialogue_social.py
python tests/test_items.py
python tests/test_movement_physics.py
python tests/test_response_parser.py
python tests/test_goal_fallback.py
python tests/test_initial_warmup_response.py
python tests/test_ollama_client.py
```

122+ tests — no LLM calls required (AI disabled in test harnesses). 4 live LLM tests skipped by default.
