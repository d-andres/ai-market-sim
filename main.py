"""main.py – FastAPI application entry point for ai-market-sim.

This module wires the simulation state produced by the ``data`` package into
HTTP and WebSocket endpoints that the frontend will consume.  It is kept thin
on purpose: all domain logic lives in ``data/`` and (later) ``src/agents/``.
"""

import os
import threading
import time
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI
from nicegui import ui

from data import load_or_build_default_map, render_ascii, load_item_catalog
from src.ui import register_pages
from src.models.schema import Actor, ActorRole, PlannedAction, WorldItem
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
    "initial_plan_expected": 0,
    "initial_plan_completed": 0,
    "block_on_initial_plans": True,
    "paused": True,
}


def _populate_default_items(world_map) -> None:
    """Place default items from items.json on shop shelves owned by the shopkeeper."""
    shopkeeper = next(
        (a for a in world_map.actors if a.role == ActorRole.SHOPKEEPER), None
    )
    owner_id = shopkeeper.id if shopkeeper else None

    shop_tiles = [(t.x, t.y) for t in world_map.tiles if t.tile_type.value == "shop"]
    if not shop_tiles:
        return

    catalog = load_item_catalog()

    # Cycle through shop tiles so we can place most/all catalog items even when
    # the map has fewer shelf tiles than item types.
    for i, item in enumerate(catalog):
        tx, ty = shop_tiles[i % len(shop_tiles)]
        world_map.world_items.append(
            WorldItem(item=item.model_copy(deep=True), x=tx, y=ty, owner_id=owner_id)
        )


def _default_actors() -> list[Actor]:
    """Create default actor population used when a map has no actors."""
    catalog = {item.id: item for item in load_item_catalog()}

    def _catalog_item(item_id: str):
        item = catalog.get(item_id)
        return item.model_copy(deep=True) if item else None

    guard_equipped = {
        "head": _catalog_item("iron_helmet"),
        "chest": _catalog_item("iron_chest"),
        "legs": _catalog_item("iron_plates"),
        "hand": _catalog_item("short_sword"),
        "offhand": _catalog_item("shield"),
    }

    return [
        Actor(
            id="guard_1",
            name="Guard Thorne",
            role=ActorRole.GUARD,
            x=5,
            y=5,
            gold=50,
            personality_traits=["vigilant", "dutiful", "suspicious"],
            hp=100,
            max_hp=100,
            base_attack=8,
            base_defense=3,
            base_crit=0.05,
            max_carry=8,
            equipped=guard_equipped,
        ),
        Actor(
            id="shopkeeper_1",
            name="Merchant Elara",
            role=ActorRole.SHOPKEEPER,
            x=10,
            y=3,
            gold=200,
            personality_traits=["shrewd", "protective", "persuasive"],
            hp=80,
            max_hp=80,
            base_attack=4,
            base_defense=2,
            base_crit=0.05,
            max_carry=6,
        ),
        Actor(
            id="player",
            name="Adventurer",
            role=ActorRole.PLAYER,
            x=10,
            y=10,
            gold=50,
            personality_traits=["ambitious", "adaptive", "risk_tolerant"],
            hp=100,
            max_hp=100,
            base_attack=6,
            base_defense=2,
            base_crit=0.05,
            max_carry=10,
        ),
    ]


def is_world_ready() -> bool:
    with _runtime_lock:
        return bool(_runtime_state["ready"])


def get_generation_status() -> dict:
    with _runtime_lock:
        expected = int(_runtime_state["initial_plan_expected"])
        completed = int(_runtime_state["initial_plan_completed"])
        block_on_initial_plans = bool(_runtime_state["block_on_initial_plans"])
        return {
            "ready": bool(_runtime_state["ready"]),
            "generating": bool(_runtime_state["generating"]),
            "error": str(_runtime_state["error"]),
            "initial_plan_expected": expected,
            "initial_plan_completed": completed,
            "awaiting_responses": max(0, expected - completed),
            "block_on_initial_plans": block_on_initial_plans,
        }


def generate_world() -> None:
    """Generate map, initialize engine, and run initial planning before going live."""
    # Default behavior is blocking warm-up so actors start with ready plans.
    block_on_initial_plans = _env_bool("BLOCK_ON_INITIAL_PLANS", True)

    with _runtime_lock:
        if _runtime_state["ready"] or _runtime_state["generating"]:
            return
        _runtime_state["generating"] = True
        _runtime_state["error"] = ""
        _runtime_state["initial_plan_expected"] = 0
        _runtime_state["initial_plan_completed"] = 0
        _runtime_state["block_on_initial_plans"] = block_on_initial_plans

    try:
        world_map = load_or_build_default_map()
        if not world_map.actors:
            world_map.actors = _default_actors()
        _populate_default_items(world_map)

        engine = initialize_engine(
            world_map,
            tick_rate=float(os.getenv("TICK_RATE", "2.0")),
            enable_ai=_env_bool("ENABLE_AI", True),
            ollama_model=os.getenv("LLM_MODEL", "ollama/llama3.2"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

        # Warm up mode:
        # - default (blocking): wait for all initial plans before world becomes ready
        # - non-blocking opt-out: set BLOCK_ON_INITIAL_PLANS=false

        # Warm up: one initial plan per actor so simulation starts with intent.
        if engine.enable_ai:
            initial_targets = [a for a in world_map.actors if engine.agent_brains.get(a.id)]
            with _runtime_lock:
                if block_on_initial_plans:
                    _runtime_state["initial_plan_expected"] = len(initial_targets)
                    _runtime_state["initial_plan_completed"] = 0
                else:
                    _runtime_state["initial_plan_expected"] = 0
                    _runtime_state["initial_plan_completed"] = 0

            if block_on_initial_plans and initial_targets:
                max_workers = max(1, len(initial_targets))
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_to_actor = {
                        pool.submit(engine.agent_brains[a.id].create_plan, "", False): a
                        for a in initial_targets
                    }
                    for future in as_completed(future_to_actor):
                        actor = future_to_actor[future]
                        try:
                            plan, thought_summary = future.result()
                        except Exception as exc:  # noqa: BLE001
                            plan = [
                                PlannedAction(
                                    action_type="wait",
                                    params={},
                                    reason=f"Initial plan generation failed: {exc}",
                                )
                            ]
                            thought_summary = f"{actor.name} failed to generate an initial plan and will retry in simulation."

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
                            _runtime_state["initial_plan_completed"] += 1
            elif initial_targets:
                for actor in initial_targets:
                    # Let the simulation start immediately; the regular tick loop
                    # will request/apply plans concurrently without blocking generation.
                    actor.action_queue = []
                    actor.needs_replan = True
                    actor.interrupt_reason = ""
                engine._log_event(
                    actor_id="system",
                    event_type="init",
                    description="Initial plan warm-up deferred to runtime ticks (non-blocking mode).",
                    data={"actors": len(initial_targets)},
                )

        with _runtime_lock:
            _runtime_state["world_map"] = world_map
            _runtime_state["engine"] = engine
            _runtime_state["ready"] = True
            _runtime_state["generating"] = False
            _runtime_state["error"] = ""
            _runtime_state["initial_plan_expected"] = 0
            _runtime_state["initial_plan_completed"] = 0
            _runtime_state["paused"] = False
    except Exception as exc:  # noqa: BLE001
        with _runtime_lock:
            _runtime_state["ready"] = False
            _runtime_state["generating"] = False
            _runtime_state["error"] = str(exc)
            _runtime_state["block_on_initial_plans"] = False


def start_generation() -> None:
    """Start world generation in the background to keep UI responsive."""
    with _runtime_lock:
        if _runtime_state["ready"] or _runtime_state["generating"]:
            return
    threading.Thread(target=generate_world, daemon=True).start()


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
            paused = bool(_runtime_state["paused"])
        try:
            if ready and engine is not None and not paused:
                engine.tick()
        except Exception:  # noqa: BLE001 — keep the loop alive on errors
            pass
        time.sleep(engine.tick_rate if engine is not None else 0.5)


_ticker = threading.Thread(target=_tick_loop, daemon=True)
_ticker.start()


def is_paused() -> bool:
    """Return True when the simulation is paused."""
    with _runtime_lock:
        return bool(_runtime_state["paused"])


def set_paused(paused: bool) -> None:
    """Pause or resume the simulation tick loop."""
    with _runtime_lock:
        _runtime_state["paused"] = paused


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
        visible_tile_set = set(viewport.visible_tiles)
        visible_items = [
            {
                "item_id": gi.item.id,
                "item_name": gi.item.name,
                "quantity": gi.item.quantity,
                "base_price": gi.item.base_price,
                "x": gi.x,
                "y": gi.y,
                "owner_id": gi.owner_id,
                "owner_role": next((a.role.value for a in world_map.actors if a.id == gi.owner_id), None),
            }
            for gi in world_map.world_items
            if (gi.x, gi.y) in visible_tile_set
        ]
        
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
                {"id": a.id, "name": a.name, "role": a.role.value, "x": a.x, "y": a.y}
                for a in viewport.visible_actors
            ],
            "visible_items": visible_items,
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
        "world_items": [
            {
                "item_id": gi.item.id,
                "item_name": gi.item.name,
                "item_description": gi.item.description,
                "quantity": gi.item.quantity,
                "base_price": gi.item.base_price,
                "x": gi.x,
                "y": gi.y,
                "owner_id": gi.owner_id,
                "is_shelf": world_map.tile_at(gi.x, gi.y).tile_type.value == "shop",
            }
            for gi in world_map.world_items
        ],
        "recent_events": engine.get_event_log(limit=200),
        "llm_calls": engine.llm_call_count,
        "llm_pending_actors": list(engine.llm_pending_actors),
    }


register_pages(
    get_world_snapshot,
    is_paused=is_paused,
    set_paused=set_paused,
    is_world_ready=is_world_ready,
    start_generation=start_generation,
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
