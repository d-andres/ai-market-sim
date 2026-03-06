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
