"""Map data loading and default market layout."""

from __future__ import annotations

import json
from pathlib import Path

from src.models.schema import Map, Tile, TileType


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_MAP_PATH = DATA_DIR / "map.json"


def _build_tile_lookup(width: int, height: int) -> dict[tuple[int, int], TileType]:
	tile_lookup = {(x, y): TileType.FLOOR for y in range(height) for x in range(width)}

	# Border walls.
	for x in range(width):
		tile_lookup[(x, 0)] = TileType.WALL
		tile_lookup[(x, height - 1)] = TileType.WALL
	for y in range(height):
		tile_lookup[(0, y)] = TileType.WALL
		tile_lookup[(width - 1, y)] = TileType.WALL

	# Interior shelf/shop rows.
	for x in range(2, width - 2):
		tile_lookup[(x, 2)] = TileType.SHOP
		tile_lookup[(x, 4)] = TileType.SHOP

	# Keep a center aisle in each shop row.
	mid_x = width // 2
	tile_lookup[(mid_x, 2)] = TileType.FLOOR
	tile_lookup[(mid_x, 4)] = TileType.FLOOR

	# Entrance in south wall.
	tile_lookup[(mid_x, height - 1)] = TileType.ENTRANCE

	return tile_lookup


def build_default_map(width: int = 20, height: int = 12) -> Map:
	if width < 10 or height < 8:
		raise ValueError("Map must be at least 10 columns wide and 8 rows tall.")

	tile_lookup = _build_tile_lookup(width=width, height=height)
	tiles: list[Tile] = []
	for y in range(height):
		for x in range(width):
			tile_type = tile_lookup[(x, y)]
			shop_id = f"shop_{x}_{y}" if tile_type == TileType.SHOP else None
			tiles.append(Tile(x=x, y=y, tile_type=tile_type, shop_id=shop_id))
	return Map(width=width, height=height, tiles=tiles)


def _tiles_from_rows(rows: list[str]) -> tuple[int, int, list[Tile]]:
	if not rows:
		raise ValueError("Map rows cannot be empty.")

	width = len(rows[0])
	if any(len(row) != width for row in rows):
		raise ValueError("All map rows must have the same length.")

	symbol_to_type = {
		"#": TileType.WALL,
		".": TileType.FLOOR,
		"S": TileType.SHOP,
		"E": TileType.ENTRANCE,
	}

	tiles: list[Tile] = []
	for y, row in enumerate(rows):
		for x, symbol in enumerate(row):
			if symbol not in symbol_to_type:
				raise ValueError(f"Unsupported map symbol '{symbol}' at ({x}, {y}).")
			tile_type = symbol_to_type[symbol]
			shop_id = f"shop_{x}_{y}" if tile_type == TileType.SHOP else None
			tiles.append(Tile(x=x, y=y, tile_type=tile_type, shop_id=shop_id))

	return width, len(rows), tiles


def load_map_from_json(path: Path | None = None) -> Map:
	map_path = path or DEFAULT_MAP_PATH
	with map_path.open("r", encoding="utf-8") as file:
		payload = json.load(file)

	rows = payload.get("rows")
	if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
		raise ValueError("map.json must define 'rows' as a list of strings.")

	width, height, tiles = _tiles_from_rows(rows)
	return Map(width=width, height=height, tiles=tiles)


def load_or_build_default_map(path: Path | None = None) -> Map:
	map_path = path or DEFAULT_MAP_PATH
	if map_path.exists():
		return load_map_from_json(path=map_path)
	return build_default_map()


def get_wall_coordinates(grid: Map) -> list[tuple[int, int]]:
	return [(tile.x, tile.y) for tile in grid.tiles if tile.tile_type == TileType.WALL]


def get_shop_locations(grid: Map) -> list[tuple[int, int]]:
	return [(tile.x, tile.y) for tile in grid.tiles if tile.tile_type == TileType.SHOP]


def render_ascii(grid: Map, show_actors: bool = True) -> str:
	"""Render map as ASCII string, optionally with actor positions.
	
	Args:
	    grid: The Map to render.
	    show_actors: If True, overlay actor symbols (G=Guard, P=Player, K=Shopkeeper, @=Other).
	    
	Returns:
	    ASCII string representation of the map.
	"""
	# Build base map from tiles.
	char_grid = [[grid.tile_at(x, y).symbol for x in range(grid.width)] for y in range(grid.height)]
	
	# Overlay actors if requested.
	if show_actors:
		from src.models.schema import ActorRole
		
		role_symbols = {
			ActorRole.GUARD: "G",
			ActorRole.SHOPKEEPER: "K",
			ActorRole.PLAYER: "@",
		}
		
		for actor in grid.actors:
			if 0 <= actor.x < grid.width and 0 <= actor.y < grid.height:
				symbol = role_symbols.get(actor.role, "?")
				char_grid[actor.y][actor.x] = symbol
	
	return "\n".join("".join(row) for row in char_grid)


DEFAULT_MAP: Map = load_or_build_default_map()

# Add test actors for physics demonstration.
from src.models.schema import Actor, ActorRole

if not DEFAULT_MAP.actors:  # Only add if no actors exist.
	DEFAULT_MAP.actors = [
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

# Backward-compatibility names used by current src/main.py.
build_default_market = build_default_map
DEFAULT_MARKET = DEFAULT_MAP


__all__ = [
	"DEFAULT_MAP_PATH",
	"build_default_map",
	"build_default_market",
	"load_map_from_json",
	"load_or_build_default_map",
	"get_wall_coordinates",
	"get_shop_locations",
	"render_ascii",
	"DEFAULT_MAP",
	"DEFAULT_MARKET",
]
