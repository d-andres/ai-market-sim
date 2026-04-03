# AI Agent Integration Guide

This guide explains how LLM-powered AI agents work in ai-market-sim. Actors (Guards, Shopkeepers, and Players) are controlled by `AgentBrain` instances that use Large Language Models to plan, trade, converse, and fight autonomously.

## Architecture

Each actor's `AgentBrain` (`src/agents/brain.py`) wraps a **LiteLLMModel** from `smolagents[litellm]`. This supports any backend that LiteLLM can route to — Ollama, OpenAI, Anthropic, etc. There is no `CodeAgent`; all LLM calls are direct `ChatMessage` requests with structured response parsing.

### Planning Flow

1. **Observe** — `ObserveSurroundingsTool` builds a text description of the actor's position, gold, inventory, visible actors/items, relationship history.
2. **Plan** — The LLM receives the observation + role-specific system prompt and returns a JSON plan of up to `PLAN_HORIZON = 30` action steps.
3. **Execute** — The engine pops one action per tick from the queue and dispatches it.
4. **Replan** — When the queue empties, an interrupt occurs, or `REPLAN_CADENCE_TICKS = 23` ticks elapse, a new plan is requested.

Planning runs on a background `ThreadPoolExecutor`; the tick loop never blocks on LLM latency. Actors with empty queues run local **reflex actions** (wander, hold, greet) instead of freezing.

### Adaptive Fallback

When primary planning fails, `AgentBrain` uses an AI-authored **fallback policy** (identity summary, goals, tactics, risk posture) to generate a smaller 10-step recovery plan. The policy refreshes every `FALLBACK_POLICY_TTL_TICKS = 40` ticks.

## Action Types

Agents can plan any of these actions each step:

| Action | Params | Description |
|---|---|---|
| `move` | `direction` (8-directional) | Walk to adjacent tile; blocked by walls/actors |
| `wait` | — | Idle observation |
| `propose_trade` | `target_actor_id`, gold/item offers | Trigger LLM trade evaluation on target |
| `pick_up` | `item_id` | Pick up unowned floor item (owned = theft) |
| `place` | `item_id`, `x`, `y` | Drop item on tile; shop tile sets owner\_id |
| `converse` | `target_actor_id`, `opening_line` | Non-blocking conversation via packet system |
| `attack` | `target_actor_id` | Melee strike (Chebyshev ≤ 1); triggers crime if unprovoked |
| `equip` | `item_id` | Move inventory item to equipment slot |
| `unequip` | `slot` | Return equipped item to inventory |
| `use_item` | `item_id` | Consume item (heal, buff, poison, etc.) |

## Role Prompts

Role-specific system prompts live in `src/agents/prompts.py`:

- **`GUARD_PROMPT`** — Patrol marketplace, investigate suspicious behaviour, maintain order.
- **`SHOPKEEPER_PROMPT`** — Manage inventory, negotiate trades, maximise profit, protect stock.
- **`PLAYER_PROMPT`** — Acquire high-value items through negotiation, deception, theft, or violence.

Additional prompts: `TRADE_EVALUATION_PROMPT`, `CONVERSATION_PACKET_PROMPT`, `DIALOGUE_SOCIAL_IMPACT_PROMPT`, `ADAPTIVE_FALLBACK_POLICY_PROMPT`, `ADAPTIVE_FALLBACK_PLAN_PROMPT`.

## Prerequisites

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama

```bash
# Windows
winget install Ollama.Ollama

# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama serve
ollama pull qwen3:4b    # or any model: llama3.2, mistral, phi3
```

### 3. Configure Environment

Copy `.env.example` to `.env`:
```env
LLM_MODEL=ollama/qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434
ENABLE_AI=true
TICK_RATE=2.0
```

Model name format: always prefix with `ollama/` (e.g., `ollama/qwen3:4b`, `ollama/llama3.2`).

## Cloudflared Tunnel (Hugging Face Spaces)

When Ollama runs locally but the app runs on HF Spaces:

```bash
ollama serve
cloudflared tunnel --url http://localhost:11434
```

Copy the generated URL and set `OLLAMA_BASE_URL` in HF Space Settings → Variables and Secrets. Quick tunnel URLs change on restart.

## Running

```bash
python main.py
```

Visit **http://localhost:7860**. Click **GENERATE MAP** to initialize the world. Once initial LLM plans complete, the simulation goes live.

### Dashboard

- **ASCII map** — tile grid with actor positions and item markers
- **Event log** — colour-coded by type (red = combat, orange = crime, cyan = trade, etc.)
- **Play/Pause** — toggle simulation ticking
- **LLM call counter** — total and pending planning requests

## Key Constants

| Constant | File | Value | Meaning |
|---|---|---|---|
| `PLAN_HORIZON` | `brain.py` | `30` | Steps per LLM plan (~60s at 2s/tick) |
| `REPLAN_CADENCE_TICKS` | `engine.py` | `23` | ~46s between routine replans |
| `PLAN_FAILURE_COOLDOWN_TICKS` | `engine.py` | `15` | ~30s retry wait after failure |
| `MAX_CONCURRENT_PLAN_JOBS` | `engine.py` | `10` | Max simultaneous LLM requests |
| `CONVERSATION_RANGE` | `prompts.py` | `8` | Max Chebyshev tiles for converse |
| `WITNESS_VISION_RANGE` | `engine.py` | `10` | Tiles within which actors witness crimes |
| `SELF_DEFENSE_WINDOW_TICKS` | `engine.py` | `12` | Retaliation exempt from crime within this window |

## Troubleshooting

### Ollama: "Connection refused"
- Ensure Ollama is running: `ollama serve`
- Verify at: http://localhost:11434
- Check `OLLAMA_BASE_URL` matches your tunnel URL on HF Spaces

### Ollama: "Model not found"
```bash
ollama pull qwen3:4b
ollama list
```

### Agents making illogical moves
- Check system prompts in `src/agents/prompts.py`
- Try a more capable model (`ollama/mistral`, `ollama/llama3.2`)
- Verify agent can see targets (default vision range: 10 tiles)
