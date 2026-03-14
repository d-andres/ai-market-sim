"""Core simulation schema models.

This module defines the authoritative data contracts used across the backend.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TileType(str, Enum):
	FLOOR = "floor"
	WALL = "wall"
	SHOP = "shop"
	ENTRANCE = "entrance"


class ActorRole(str, Enum):
	PLAYER = "player"
	GUARD = "guard"
	SHOPKEEPER = "shopkeeper"


class Item(BaseModel):
	id: str = Field(min_length=1)
	name: str = Field(min_length=1)
	description: str = ""
	base_price: int = Field(ge=0)
	quantity: int = Field(default=1, ge=0)
	metadata: dict[str, str] = Field(default_factory=dict)


class Tile(BaseModel):
	model_config = ConfigDict(frozen=True)

	x: int = Field(ge=0)
	y: int = Field(ge=0)
	tile_type: TileType
	shop_id: str | None = None

	@property
	def walkable(self) -> bool:
		return self.tile_type in {TileType.FLOOR, TileType.ENTRANCE}

	@property
	def interactable(self) -> bool:
		return self.tile_type == TileType.SHOP

	@property
	def symbol(self) -> str:
		symbols = {
			TileType.FLOOR: ".",
			TileType.WALL: "#",
			TileType.SHOP: "S",
			TileType.ENTRANCE: "E",
		}
		return symbols[self.tile_type]


class PlannedAction(BaseModel):
	"""A single action step in an actor's current plan."""

	action_type: str  # "move", "wait", "propose_trade"
	params: dict = Field(default_factory=dict)
	reason: str = ""


class Actor(BaseModel):
	id: str = Field(min_length=1)
	name: str = Field(min_length=1)
	role: ActorRole
	x: int = Field(ge=0)
	y: int = Field(ge=0)
	gold: int = Field(default=0, ge=0)
	hp: int = Field(default=100, ge=0)
	inventory: list[Item] = Field(default_factory=list)
	# Reactive planning state
	action_queue: list[PlannedAction] = Field(default_factory=list)
	needs_replan: bool = Field(default=True)
	interrupt_reason: str = ""


class Map(BaseModel):
	width: int = Field(ge=1)
	height: int = Field(ge=1)
	tiles: list[Tile]
	actors: list[Actor] = Field(default_factory=list)

	@model_validator(mode="after")
	def validate_layout(self) -> "Map":
		expected_tile_count = self.width * self.height
		if len(self.tiles) != expected_tile_count:
			raise ValueError(
				f"Expected {expected_tile_count} tiles for a {self.width}x{self.height} map, "
				f"got {len(self.tiles)}."
			)

		tile_positions = {(tile.x, tile.y) for tile in self.tiles}
		if len(tile_positions) != len(self.tiles):
			raise ValueError("Duplicate tile coordinates detected.")

		for x, y in tile_positions:
			if not (0 <= x < self.width and 0 <= y < self.height):
				raise ValueError(f"Tile ({x}, {y}) is out of bounds.")

		for actor in self.actors:
			if not self.in_bounds(actor.x, actor.y):
				raise ValueError(
					f"Actor '{actor.id}' is out of bounds at ({actor.x}, {actor.y})."
				)

		return self

	def in_bounds(self, x: int, y: int) -> bool:
		return 0 <= x < self.width and 0 <= y < self.height

	def tile_at(self, x: int, y: int) -> Tile:
		if not self.in_bounds(x, y):
			raise IndexError(f"Coordinate ({x}, {y}) is out of bounds.")
		return self.tiles[y * self.width + x]

	def is_walkable(self, x: int, y: int) -> bool:
		return self.in_bounds(x, y) and self.tile_at(x, y).walkable


class TradeProposal(BaseModel):
	"""A proposed exchange of items and/or gold between two actors."""

	proposer_id: str
	target_id: str
	offered_items: list[Item] = Field(default_factory=list)
	offered_gold: int = Field(default=0, ge=0)
	requested_items: list[Item] = Field(default_factory=list)
	requested_gold: int = Field(default=0, ge=0)


__all__ = [
	"TileType",
	"ActorRole",
	"Item",
	"PlannedAction",
	"Tile",
	"Actor",
	"Map",
	"TradeProposal",
]
