"""Data access layer for map state and loaders."""

import json
from pathlib import Path

from src.models.schema import Actor, ActorRole, Item, Map, Tile, TileType

from .map import (
    DEFAULT_MAP,
    DEFAULT_MAP_PATH,
    DEFAULT_MARKET,
    build_default_map,
    build_default_market,
    get_shop_locations,
    get_wall_coordinates,
    load_map_from_json,
    load_or_build_default_map,
    render_ascii,
)

_ITEMS_JSON_PATH = Path(__file__).parent / "items.json"


def load_item_catalog() -> list[Item]:
    """Load all items from items.json as Item instances."""
    raw = json.loads(_ITEMS_JSON_PATH.read_text(encoding="utf-8"))
    return [Item(**entry) for entry in raw["items"]]


__all__ = [
    "TileType",
    "Tile",
    "Map",
    "Item",
    "ActorRole",
    "Actor",
    "DEFAULT_MAP_PATH",
    "build_default_map",
    "build_default_market",
    "get_wall_coordinates",
    "get_shop_locations",
    "load_map_from_json",
    "load_or_build_default_map",
    "render_ascii",
    "DEFAULT_MAP",
    "DEFAULT_MARKET",
    "load_item_catalog",
]
