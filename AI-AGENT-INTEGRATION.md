# AI Agent Integration Guide

This guide explains how to integrate LLM-powered AI agents into the ai-market-sim project. Actors (Guards, Shopkeepers, and Players) can be controlled by AI "brains" that use Large Language Models to make autonomous decisions.

## Overview

Each actor in the simulation has an `AgentBrain` that:
1. **Observes** the world using registered tools
2. **Reasons** about the situation using an LLM
3. **Acts** by calling tools (move, wait, interact)

The system uses **Smolagents** as the agentic framework, which supports multiple LLM providers through **LiteLLM**.

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

### 2. Choose Your LLM Provider

You can use any LLM provider supported by LiteLLM. Below are guides for the most common options.

---

## Option A: Local LLM with Ollama (Recommended for Development)

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

### Configure in main.py
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=2.0,
    enable_ai=True,
    ollama_model="ollama/llama3.2",  # Format: "ollama/<model_name>"
    ollama_base_url="http://localhost:11434",
)
```

**Model name format:** Always prefix with `ollama/` (e.g., `ollama/llama3.2`, `ollama/mistral`)

---

## Option B: OpenAI API

**Pros:**
- High-quality models (GPT-4, GPT-3.5)
- No local compute required
- Very capable reasoning

**Cons:**
- Costs money per API call
- Requires internet connection
- Data sent to OpenAI servers

### Setup

1. **Get an API key** from [platform.openai.com](https://platform.openai.com/api-keys)

2. **Set environment variable:**
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# macOS/Linux
export OPENAI_API_KEY="sk-your-key-here"
```

3. **Configure in main.py:**
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=2.0,
    enable_ai=True,
    ollama_model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo"
    ollama_base_url=None,  # Not used for OpenAI
)
```

**Note:** Despite the parameter name `ollama_model`, LiteLLM auto-detects the provider based on the model name.

---

## Option C: Anthropic Claude

**Pros:**
- Excellent reasoning capabilities
- Strong at following instructions
- Good code generation

**Cons:**
- Costs money per API call
- Requires internet connection

### Setup

1. **Get an API key** from [console.anthropic.com](https://console.anthropic.com/)

2. **Set environment variable:**
```bash
# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"

# macOS/Linux
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

3. **Configure in main.py:**
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=2.0,
    enable_ai=True,
    ollama_model="claude-3-5-sonnet-20241022",  # or "claude-3-opus-20240229"
    ollama_base_url=None,
)
```

---

## Option D: Other LLM Providers

LiteLLM supports 100+ providers. See [LiteLLM docs](https://docs.litellm.ai/docs/providers) for full list.

### Examples

**Azure OpenAI:**
```python
ollama_model="azure/my-deployment-name"
# Requires: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION env vars
```

**Hugging Face:**
```python
ollama_model="huggingface/meta-llama/Llama-2-7b-chat-hf"
# Requires: HUGGINGFACE_API_KEY env var
```

**Google Gemini:**
```python
ollama_model="gemini/gemini-pro"
# Requires: GEMINI_API_KEY env var
```

**Groq (fast inference):**
```python
ollama_model="groq/llama3-8b-8192"
# Requires: GROQ_API_KEY env var
```

---

## Configuration

### Enable/Disable AI

Toggle AI agents on or off in [main.py](main.py):
```python
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=2.0,
    enable_ai=False,  # Set to False to disable AI (actors won't move)
)
```

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

## Customizing Agent Behavior

### Role-Specific Prompts

Agents have different personalities and goals based on their role. Edit [src/agents/prompts.py](src/agents/prompts.py):

- **GUARD_PROMPT** - Patrols, watches for threats, investigates suspicious activity
- **SHOPKEEPER_PROMPT** - Stays near shop, watches inventory, ready to trade
- **PLAYER_PROMPT** - Explores, observes, plans strategic interactions

Example customization:
```python
GUARD_PROMPT = """You are an extremely aggressive guard.
Chase down anyone who comes near the shops.
Your goal is to intimidate and control the marketplace.
..."""
```

### Adding New Tools

To give agents new capabilities, create a new Tool class in [src/agents/brain.py](src/agents/brain.py):

```python
class InteractTool(Tool):
    name = "interact"
    description = "Interact with a nearby actor or shop"
    inputs = {
        "target_name": {
            "type": "string",
            "description": "Name of the actor or shop to interact with",
        }
    }
    output_type = "string"
    
    def forward(self, target_name: str) -> str:
        # Your interaction logic here
        return f"Interacted with {target_name}"
```

Then register it in the `AgentBrain.__init__()` method:
```python
self.tools = [
    ObserveSurroundingsTool(...),
    MoveTool(...),
    WaitTool(...),
    InteractTool(...),  # Add your new tool
]
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
- Check the base URL in main.py matches your Ollama port

### Ollama: "Model not found"
```bash
# Pull the model first
ollama pull llama3.2

# List installed models
ollama list
```

### OpenAI/Anthropic: "Authentication failed"
- Verify your API key is set: `echo $OPENAI_API_KEY`
- Check you have credits remaining in your account
- Ensure the API key has the correct permissions

### Agents taking too long to respond
- **Try a smaller/faster model:** `ollama/phi3`, `gpt-3.5-turbo`
- **Reduce max_steps** in brain.py (e.g., from 3 to 1)
- **Increase tick_rate** for slower ticks (more time between decisions)

### Agents making illogical moves
- **Check system prompts** in `src/agents/prompts.py`
- **Try a more capable model:** `ollama/mistral`, `gpt-4o`, `claude-3-5-sonnet`
- **Increase max_steps** for more reasoning (e.g., 3 → 5)
- **Verify agent can see targets:** Check vision_range is sufficient

### High API costs
- **Use Ollama** for free local inference during development
- **Switch to smaller models:** `gpt-3.5-turbo` instead of `gpt-4`
- **Increase tick_rate** so agents act less frequently
- **Disable AI for non-critical actors** (edit engine initialization code)

---

## Performance Tips

### For Ollama (Local)
- Use GPU acceleration if available (automatically used)
- Smaller models = faster responses: `phi3` > `llama3.2` > `mistral`
- Close other heavy applications for more resources
- Consider smaller context windows if using very large models

### For API Providers
- Use caching when possible (some providers support prompt caching)
- Batch actions when feasible
- Monitor costs via provider dashboard
- Set usage limits to avoid surprise bills

---

## Next Steps

With AI agents integrated, you can:
- **Add more tools:** Interact with shops, attack, pick up items, trade
- **Enhance prompts:** Give agents memories, goals, personalities
- **Add player control:** Allow human input to override or guide AI
- **Implement economy:** Enable buying/selling via agent tools
- **Add persistence:** Save agent memories across simulation sessions
- **Deploy to Hugging Face Spaces:** Share your simulation online

See [GOAL.md](GOAL.md) for the full project roadmap.
