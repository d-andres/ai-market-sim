"""main.py – FastAPI application entry point for ai-market-sim.

This module wires the simulation state produced by the ``data`` package into
HTTP and WebSocket endpoints that the frontend will consume.  It is kept thin
on purpose: all domain logic lives in ``data/`` and (later) ``src/agents/``.
"""

from fastapi import FastAPI
from nicegui import ui

from data import DEFAULT_MARKET, render_ascii
from src.ui import register_pages
from src.simulation.physics import get_visible_tiles_and_actors, breadth_first_search
from src.simulation.engine import initialize_engine, get_engine

app = FastAPI(title="ai-market-sim", version="0.1.0")

# Initialize the simulation engine with AI enabled.
# Set enable_ai=False to disable AI agents for testing.
# See AI-AGENT-INTEGRATION.md for LLM provider setup instructions.
engine = initialize_engine(
    DEFAULT_MARKET,
    tick_rate=2.0,
    enable_ai=True,  # Toggle AI on/off
    ollama_model="ollama/llama3.2",  # LLM model (format depends on provider)
    ollama_base_url="http://localhost:11434",  # API base URL (for Ollama)
)


def get_world_snapshot() -> dict:
    """Return the current world state as a serializable snapshot.
    
    Includes physics data: actor FOV, pathfinding results.
    Advances the simulation by one tick each call.
    """
    # Tick the engine (advance simulation by one step).
    engine.tick()
    
    # Compute physics data for each actor.
    actor_data = []
    for actor in DEFAULT_MARKET.actors:
        viewport = get_visible_tiles_and_actors(DEFAULT_MARKET, actor, vision_range=10)
        
        # Find path to a target for demo (e.g., first visible actor).
        path = None
        if viewport.visible_actors:
            target = viewport.visible_actors[0]
            path = breadth_first_search(
                DEFAULT_MARKET,
                actor.x,
                actor.y,
                target.x,
                target.y,
            )
        
        actor_data.append({
            "id": actor.id,
            "name": actor.name,
            "role": actor.role.value,
            "x": actor.x,
            "y": actor.y,
            "gold": actor.gold,
            "hp": actor.hp,
            "visible_tiles": list(viewport.visible_tiles),
            "visible_actors": [
                {"id": a.id, "name": a.name, "x": a.x, "y": a.y}
                for a in viewport.visible_actors
            ],
            "path": path,
        })
    
    return {
        "tick": engine.tick_count,
        "elapsed_time": engine.get_elapsed_time(),
        "elapsed_time_formatted": engine.get_elapsed_time_formatted(),
        "width": DEFAULT_MARKET.width,
        "height": DEFAULT_MARKET.height,
        "tiles": [
            {
                "x": tile.x,
                "y": tile.y,
                "tile_type": tile.tile_type.value,
                "walkable": tile.walkable,
                "interactable": tile.interactable,
                "symbol": tile.symbol,
            }
            for tile in DEFAULT_MARKET.tiles
        ],
        "ascii": render_ascii(DEFAULT_MARKET, show_actors=True),
        "actors": actor_data,
        "recent_events": engine.get_event_log(limit=10),
    }


register_pages(get_world_snapshot)
ui.run_with(app, storage_secret="ai-market-sim-dev-secret")


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check — confirms the server is running."""
    return {"status": "ok"}


@app.get("/world")
async def get_world() -> dict:
    """Return the current world grid as structured data.

    The response includes the raw tile list (for programmatic use by the
    frontend renderer) and an ASCII snapshot for quick debugging.
    Also includes actor physics data: FOV, visible actors, pathfinding results.
    """
    return get_world_snapshot()
