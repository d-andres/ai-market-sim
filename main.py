"""main.py – FastAPI application entry point for ai-market-sim.

This module wires the simulation state produced by the ``data`` package into
HTTP and WebSocket endpoints that the frontend will consume.  It is kept thin
on purpose: all domain logic lives in ``data/`` and (later) ``src/agents/``.
"""

import os
import threading
import time
from threading import Lock

from fastapi import FastAPI
from nicegui import ui

from data import load_or_build_default_map, render_ascii
from src.ui import register_pages
from src.models.schema import Actor, ActorRole
from src.simulation.physics import get_visible_tiles_and_actors, breadth_first_search
from src.simulation.engine import initialize_engine

app = FastAPI(title="ai-market-sim", version="0.1.0")


def _load_local_env_file(env_file: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file.

    Existing environment variables are not overwritten.
    """
    if not os.path.exists(env_file):
        return

    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_local_env_file()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

_runtime_lock = Lock()
_runtime_state = {
    "world_map": None,
    "engine": None,
    "ready": False,
    "generating": False,
    "error": "",
}


manual_tick = _env_bool("MANUAL_TICK", False)


def _default_actors() -> list[Actor]:
    """Create default actor population used when a map has no actors."""
    return [
        Actor(
            id="guard_1",
            name="Guard Thorne",
            role=ActorRole.GUARD,
            x=5,
            y=5,
            gold=50,
            hp=100,
        ),
        Actor(
            id="shopkeeper_1",
            name="Merchant Elara",
            role=ActorRole.SHOPKEEPER,
            x=10,
            y=3,
            gold=200,
            hp=80,
        ),
        Actor(
            id="player",
            name="Adventurer",
            role=ActorRole.PLAYER,
            x=10,
            y=10,
            gold=0,
            hp=100,
        ),
    ]


def is_world_ready() -> bool:
    with _runtime_lock:
        return bool(_runtime_state["ready"])


def get_generation_status() -> dict:
    with _runtime_lock:
        return {
            "ready": bool(_runtime_state["ready"]),
            "generating": bool(_runtime_state["generating"]),
            "error": str(_runtime_state["error"]),
        }


def generate_world() -> None:
    """Generate map, initialize engine, and run initial planning before going live."""
    with _runtime_lock:
        if _runtime_state["ready"] or _runtime_state["generating"]:
            return
        _runtime_state["generating"] = True
        _runtime_state["error"] = ""

    try:
        world_map = load_or_build_default_map()
        if not world_map.actors:
            world_map.actors = _default_actors()

        engine = initialize_engine(
            world_map,
            tick_rate=float(os.getenv("TICK_RATE", "2.0")),
            enable_ai=_env_bool("ENABLE_AI", True),
            ollama_model=os.getenv("LLM_MODEL", "ollama/llama3.2"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        # Warm up: one initial plan per actor so simulation starts with intent.
        if engine.enable_ai:
            for actor in world_map.actors:
                brain = engine.agent_brains.get(actor.id)
                if not brain:
                    continue
                plan, thought_summary = brain.create_plan()
                actor.action_queue = plan
                actor.needs_replan = False
                actor.interrupt_reason = ""
                engine._log_event(
                    actor_id=actor.id,
                    event_type="plan",
                    description=thought_summary,
                    data={"steps": len(plan), "initial": True},
                )

        with _runtime_lock:
            _runtime_state["world_map"] = world_map
            _runtime_state["engine"] = engine
            _runtime_state["ready"] = True
            _runtime_state["generating"] = False
            _runtime_state["error"] = ""
    except Exception as exc:  # noqa: BLE001
        with _runtime_lock:
            _runtime_state["ready"] = False
            _runtime_state["generating"] = False
            _runtime_state["error"] = str(exc)


def advance_tick() -> None:
    """Advance the simulation by one tick (used by the UI button and auto-loop)."""
    with _runtime_lock:
        engine = _runtime_state["engine"]
    if engine is None:
        return
    engine.tick()


def _tick_loop() -> None:
    """Background thread: advances the simulation at the configured tick rate."""
    while True:
        with _runtime_lock:
            engine = _runtime_state["engine"]
            ready = bool(_runtime_state["ready"])
        try:
            if ready and engine is not None:
                engine.tick()
        except Exception:  # noqa: BLE001 — keep the loop alive on errors
            pass
        time.sleep(engine.tick_rate if engine is not None else 0.5)


if not manual_tick:
    _ticker = threading.Thread(target=_tick_loop, daemon=True)
    _ticker.start()


def get_world_snapshot() -> dict:
    """Return the current world state as a serializable snapshot.

    Reads state only — ticking happens on the background thread.
    """
    with _runtime_lock:
        world_map = _runtime_state["world_map"]
        engine = _runtime_state["engine"]
        ready = bool(_runtime_state["ready"])
        generating = bool(_runtime_state["generating"])
        error = str(_runtime_state["error"])

    if not ready or world_map is None or engine is None:
        return {
            "ready": False,
            "generating": generating,
            "error": error,
            "tick": 0,
            "elapsed_time": 0.0,
            "elapsed_time_formatted": "00:00",
            "width": 0,
            "height": 0,
            "tiles": [],
            "ascii": "",
            "actors": [],
            "recent_events": [],
            "llm_calls": 0,
            "llm_pending_actors": [],
        }

    # Compute physics data for each actor.
    actor_data = []
    for actor in world_map.actors:
        viewport = get_visible_tiles_and_actors(world_map, actor, vision_range=10)
        
        # Find path to a target for demo (e.g., first visible actor).
        path = None
        if viewport.visible_actors:
            target = viewport.visible_actors[0]
            path = breadth_first_search(
                world_map,
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
        "ready": True,
        "generating": False,
        "error": "",
        "elapsed_time": engine.get_elapsed_time(),
        "elapsed_time_formatted": engine.get_elapsed_time_formatted(),
        "width": world_map.width,
        "height": world_map.height,
        "tiles": [
            {
                "x": tile.x,
                "y": tile.y,
                "tile_type": tile.tile_type.value,
                "walkable": tile.walkable,
                "interactable": tile.interactable,
                "symbol": tile.symbol,
            }
            for tile in world_map.tiles
        ],
        "ascii": render_ascii(world_map, show_actors=True),
        "actors": actor_data,
        "recent_events": engine.get_event_log(limit=200),
        "llm_calls": engine.llm_call_count,
        "llm_pending_actors": list(engine.llm_pending_actors),
    }


register_pages(
    get_world_snapshot,
    advance_tick=advance_tick,
    manual_tick=manual_tick,
    is_world_ready=is_world_ready,
    start_generation=generate_world,
    get_generation_status=get_generation_status,
)
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
