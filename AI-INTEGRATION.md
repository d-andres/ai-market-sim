# AI Agent Integration Guide

This guide explains how to integrate LLM-powered AI agents into the ai-market-sim project. Actors (Guards, Shopkeepers, and Players) can be controlled by AI "brains" that use Large Language Models to make autonomous decisions.

## Overview

Each actor in the simulation has an `AgentBrain` that:
1. **Observes** the world using registered tools
2. **Reasons** about the situation using an LLM
3. **Acts** by calling tools (move, wait, interact)

The system uses **Smolagents** as the agentic framework with **Ollama** as the configured model backend.

## AI Tools Available to Agents

Agents can perform these actions each turn:

- **`observe_surroundings(vision_range=10)`** - See nearby tiles, actors, and shops within range
- **`move(direction)`** - Move in 8 directions: north, south, east, west, northeast, northwest, southeast, southwest
- **`wait()`** - Stay still and observe without moving

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `smolagents` - Agentic framework
- `ollama` - Python client for Ollama (if using Ollama)
- Other required packages (fastapi, nicegui, pydantic, etc.)

---

## Ollama Integration

**Pros:**
- Free and runs entirely offline
- No API keys required
- Fast iteration during development
- Privacy - your data never leaves your machine

**Cons:**
- Requires local compute resources
- Model quality depends on hardware

### Installation

#### Windows
```powershell
# Using winget (recommended)
winget install Ollama.Ollama

# Or download installer from:
# https://ollama.com/download
```

#### macOS
```bash
# Using Homebrew
brew install ollama

# Or download from:
# https://ollama.com/download
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Start Ollama Server
```bash
# The service usually starts automatically after installation
# Verify it's running by visiting:
# http://localhost:11434

# Or start manually:
ollama serve
```

### Pull a Model
```bash
# Option 1: llama3.2 (recommended, balanced performance)
ollama pull llama3.2

# Option 2: mistral (larger, more capable)
ollama pull mistral

# Option 3: phi3 (smaller, faster, less capable)
ollama pull phi3

# See all models: https://ollama.com/library
```

### Quick Workflow: Cloudflared Tunnel to Hugging Face Spaces

Use this when Ollama runs on your local machine, but your app runs on Hugging Face Spaces.

1. Start Ollama locally:
```bash
ollama serve
```

2. Start a quick Cloudflared tunnel to Ollama:
```bash
cloudflared tunnel --url http://localhost:11434
```

3. Copy the generated URL (example: `https://abc123.trycloudflare.com`).

4. In your Hugging Face Space, update Settings -> Variables and Secrets:
- Variable or Secret: `OLLAMA_BASE_URL=https://abc123.trycloudflare.com`
- Variable: `LLM_MODEL=ollama/llama3.2`
- Variable: `ENABLE_AI=true`
- Variable: `TICK_RATE=2.0`

5. Restart the Hugging Face Space so it picks up the new values.

Notes:
- Quick tunnel URLs change whenever you restart Cloudflared.
- When that happens, update only `OLLAMA_BASE_URL` in Hugging Face Space and restart.
- Keep both `ollama serve` and `cloudflared tunnel --url ...` running on your local machine.

### Configure in main.py
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=float(os.getenv("TICK_RATE", "2.0")),
    enable_ai=_env_bool("ENABLE_AI", True),
    ollama_model=os.getenv("LLM_MODEL", "ollama/llama3.2"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
```

**Model name format:** Always prefix with `ollama/` (e.g., `ollama/llama3.2`, `ollama/mistral`)

---

## Configuration

### Enable/Disable AI

Toggle AI agents on or off in [main.py](main.py):
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=float(os.getenv("TICK_RATE", "2.0")),
    enable_ai=_env_bool("ENABLE_AI", True),
    ollama_model=os.getenv("LLM_MODEL", "ollama/llama3.2"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
)
```

For Hugging Face Spaces, set these in Settings -> Variables and Secrets:
- `OLLAMA_BASE_URL`
- `LLM_MODEL`
- `ENABLE_AI`
- `TICK_RATE`

### Adjust Tick Speed

Control how often agents take turns:
```python
tick_rate=5.0  # Slower: 1 tick every 5 seconds (easier to observe)
tick_rate=1.0  # Faster: 1 tick every second
```

### Change Default Vision Range

Edit [src/agents/brain.py](src/agents/brain.py) line 35:
```python
"default": 10,  # Change to any value (tiles)
```

### Limit Agent Reasoning Steps

Edit [src/agents/brain.py](src/agents/brain.py) line 189:
```python
self.agent = CodeAgent(
    tools=self.tools,
    model=self.model,
    system_prompt=system_prompt,
    max_steps=3,  # Increase for more complex reasoning chains
)
```

---

## Running the Simulation

### Start the Server
```bash
python main.py
```

Visit **http://localhost:8080** to access the dashboard.

### What You'll See

The dashboard displays:
- **World Map** - ASCII grid showing actor positions
- **Simulation Heartbeat** - Tick counter and elapsed time
- **Recent Events** - Last 10 agent actions with descriptions
- **State Inspector** - Full JSON state snapshot
- **Physics Debug** - FOV and pathfinding data per actor

### Observing Agent Decisions

Watch the **Recent Events** panel to see agents making autonomous decisions:
```
Tick 3: Thorne (guard) - "Moved east to (6,5)"
Tick 3: Elara (shopkeeper) - "Waiting and observing"
Tick 3: Adventurer (player) - "Moved north to (10,9)"
```

---

## Troubleshooting

### "Import smolagents could not be resolved"
```bash
pip install smolagents
```

### Ollama: "Connection refused"
- Make sure Ollama is running: `ollama serve`
- Verify at: http://localhost:11434
- Check `OLLAMA_BASE_URL` in Hugging Face Spaces matches your current Cloudflared URL

### Ollama: "Model not found"
```bash
# Pull the model first
ollama pull llama3.2

# List installed models
ollama list
```

### Agents taking too long to respond
- **Try a smaller/faster model:** `ollama/phi3`
- **Reduce max_steps** in brain.py (e.g., from 3 to 1)
- **Increase tick_rate** for slower ticks (more time between decisions)

### Agents making illogical moves
- **Check system prompts** in `src/agents/prompts.py`
- **Try a more capable model:** `ollama/mistral`
- **Increase max_steps** for more reasoning (e.g., 3 → 5)
- **Verify agent can see targets:** Check vision_range is sufficient
