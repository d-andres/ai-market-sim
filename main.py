"""main.py – FastAPI application entry point for ai-market-sim.

This module wires the simulation state produced by the ``data`` package into
HTTP and WebSocket endpoints that the frontend will consume.  It is kept thin
on purpose: all domain logic lives in ``data/`` and (later) ``src/agents/``.
"""

from fastapi import FastAPI
from nicegui import ui

from data import DEFAULT_MARKET, render_ascii
from src.ui import register_pages

app = FastAPI(title="ai-market-sim", version="0.1.0")


def get_world_snapshot() -> dict:
    """Return the current world state as a serializable snapshot."""
    return {
        "width": DEFAULT_MARKET.width,
        "height": DEFAULT_MARKET.height,
        "tiles": [
            {
                "x": tile.x,
                "y": tile.y,
                "tile_type": tile.tile_type,
                "walkable": tile.walkable,
                "interactable": tile.interactable,
                "symbol": tile.symbol,
            }
            for tile in DEFAULT_MARKET.tiles
        ],
        "ascii": render_ascii(DEFAULT_MARKET),
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
    """
    return get_world_snapshot()
