"""Data access layer for map state and loaders."""

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
]
