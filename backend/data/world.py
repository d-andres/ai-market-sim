"""world.py – Physical world representation for ai-market-sim.

Three concerns are addressed here:

1. **Tile Schema**   – Pydantic models that describe every cell in the grid.
2. **Grid Logic**    – A rectangular container with spatial queries used by
                       movement, pathfinding, and agent-interaction systems.
3. **ASCII Renderer** – A lightweight string renderer so the world state can
                        be inspected in a terminal or logged during simulation.

Coordinate convention
---------------------
  x – column index, 0 = left-most, increases rightward.
  y – row    index, 0 = top-most, increases downward.

The flat ``tiles`` list is stored in **row-major** order::

    index = y * width + x
"""

from __future__ import annotations

from enum import Enum
from typing import Iterator, Optional

from pydantic import BaseModel, ConfigDict, computed_field, model_validator


# ---------------------------------------------------------------------------
# TileType enum
# ---------------------------------------------------------------------------

class TileType(str, Enum):
    """All valid tile classifications in the simulation world.

    Inheriting from ``str`` means a ``TileType`` value serialises cleanly to
    JSON and can be stored / compared as a plain string wherever needed.
    """

    FLOOR    = "floor"     # walkable, empty space
    WALL     = "wall"      # impassable structural barrier
    SHELF    = "shelf"     # impassable, but agents may interact from adjacent tile
    ENTRANCE = "entrance"  # walkable transition point (spawn / exit)


# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------
# Keeping these outside the model avoids repeated dict construction and keeps
# the Tile class declarative.

_WALKABLE: dict[TileType, bool] = {
    TileType.FLOOR:    True,
    TileType.WALL:     False,
    TileType.SHELF:    False,
    TileType.ENTRANCE: True,
}

_INTERACTABLE: dict[TileType, bool] = {
    TileType.FLOOR:    False,
    TileType.WALL:     False,
    TileType.SHELF:    True,
    TileType.ENTRANCE: False,
}

_SYMBOL: dict[TileType, str] = {
    TileType.FLOOR:    ".",
    TileType.WALL:     "#",
    TileType.SHELF:    "S",
    TileType.ENTRANCE: "E",
}


# ---------------------------------------------------------------------------
# Tile – the atomic unit of the world
# ---------------------------------------------------------------------------

class Tile(BaseModel):
    """A single immutable cell within the world grid.

    Attributes
    ----------
    tile_type:
        The primary classification of this cell.  Everything else is derived.
    x:
        Column index (0 = left).
    y:
        Row index (0 = top).
    walkable:
        ``True`` when an agent may occupy this cell.  Derived from
        ``tile_type`` via :data:`_WALKABLE`.
    interactable:
        ``True`` when an agent may *interact* with this cell (e.g. browse a
        shelf) while standing on an adjacent tile.  Derived from
        ``tile_type`` via :data:`_INTERACTABLE`.
    symbol:
        The single ASCII character used to represent this tile in a rendered
        map.  Derived from ``tile_type`` via :data:`_SYMBOL`.
    """

    model_config = ConfigDict(frozen=True)

    tile_type: TileType
    x: int
    y: int

    @computed_field  # type: ignore[misc]
    @property
    def walkable(self) -> bool:
        """Whether an agent may stand on this tile."""
        return _WALKABLE[self.tile_type]

    @computed_field  # type: ignore[misc]
    @property
    def interactable(self) -> bool:
        """Whether an agent may interact with this tile from an adjacent cell."""
        return _INTERACTABLE[self.tile_type]

    @computed_field  # type: ignore[misc]
    @property
    def symbol(self) -> str:
        """Single-character ASCII glyph for this tile."""
        return _SYMBOL[self.tile_type]

    def __repr__(self) -> str:
        return f"Tile({self.tile_type.value!r}, x={self.x}, y={self.y})"


# ---------------------------------------------------------------------------
# Direction constants
# ---------------------------------------------------------------------------

# Cardinal neighbours only – used for movement and interaction checks.
_CARDINAL: list[tuple[int, int]] = [
    ( 0, -1),   # north
    ( 1,  0),   # east
    ( 0,  1),   # south
    (-1,  0),   # west
]

# Eight-directional neighbours – useful for future line-of-sight work.
_ALL_DIRS: list[tuple[int, int]] = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),           (1,  0),
    (-1,  1), (0,  1), (1,  1),
]


# ---------------------------------------------------------------------------
# Grid – the world container
# ---------------------------------------------------------------------------

class Grid(BaseModel):
    """A rectangular 2-D grid of :class:`Tile` objects.

    Attributes
    ----------
    width:
        Number of columns.
    height:
        Number of rows.
    tiles:
        Flat, row-major list of all tiles.  Length must equal
        ``width * height``.

    Notes
    -----
    The grid is conceptually **immutable**.  Methods that change a tile
    (e.g. :meth:`set_tile`) return a *new* ``Grid`` instance, leaving the
    original untouched.  This makes it straightforward to snapshot world
    state for replays or agent planning.
    """

    width:  int
    height: int
    tiles:  list[Tile]

    @model_validator(mode="after")
    def _check_tile_count(self) -> "Grid":
        expected = self.width * self.height
        if len(self.tiles) != expected:
            raise ValueError(
                f"tiles list length {len(self.tiles)} does not match "
                f"grid dimensions {self.width}×{self.height} = {expected}."
            )
        return self

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def in_bounds(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is a valid grid coordinate."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_tile(self, x: int, y: int) -> Tile:
        """Return the :class:`Tile` at ``(x, y)``.

        Raises
        ------
        IndexError
            If the coordinate lies outside the grid boundaries.
        """
        if not self.in_bounds(x, y):
            raise IndexError(
                f"Coordinate ({x}, {y}) is out of bounds for a "
                f"{self.width}×{self.height} grid."
            )
        return self.tiles[y * self.width + x]

    def set_tile(self, x: int, y: int, tile_type: TileType) -> "Grid":
        """Return a new :class:`Grid` with the tile at ``(x, y)`` replaced.

        Parameters
        ----------
        x, y:
            Target coordinate.
        tile_type:
            The replacement :class:`TileType`.

        Returns
        -------
        Grid
            A new Grid instance with the updated tile; the original is
            unchanged.

        Raises
        ------
        IndexError
            If the coordinate lies outside the grid boundaries.
        """
        if not self.in_bounds(x, y):
            raise IndexError(
                f"Coordinate ({x}, {y}) is out of bounds."
            )
        new_tiles = list(self.tiles)
        new_tiles[y * self.width + x] = Tile(tile_type=tile_type, x=x, y=y)
        return Grid(width=self.width, height=self.height, tiles=new_tiles)

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def is_walkable(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is in-bounds *and* walkable."""
        return self.in_bounds(x, y) and self.get_tile(x, y).walkable

    def is_interactable(self, x: int, y: int) -> bool:
        """Return ``True`` if ``(x, y)`` is in-bounds *and* interactable."""
        return self.in_bounds(x, y) and self.get_tile(x, y).interactable

    def cardinal_neighbors(self, x: int, y: int) -> list[Tile]:
        """Return the (up to four) tiles cardinally adjacent to ``(x, y)``.

        Out-of-bounds directions are silently omitted.
        """
        result: list[Tile] = []
        for dx, dy in _CARDINAL:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                result.append(self.get_tile(nx, ny))
        return result

    def walkable_neighbors(self, x: int, y: int) -> list[Tile]:
        """Return walkable cardinal neighbours of ``(x, y)``.

        This is the primary input for grid-based pathfinding (e.g. A*).
        """
        return [t for t in self.cardinal_neighbors(x, y) if t.walkable]

    def adjacent_interactables(self, x: int, y: int) -> list[Tile]:
        """Return interactable tiles cardinally adjacent to ``(x, y)``.

        Agents call this to discover which shelves (or other interactable
        objects) they can reach without moving.
        """
        return [t for t in self.cardinal_neighbors(x, y) if t.interactable]

    def tiles_of_type(self, tile_type: TileType) -> list[Tile]:
        """Return all tiles in the grid that match ``tile_type``."""
        return [t for t in self.tiles if t.tile_type == tile_type]

    def iter_rows(self) -> Iterator[list[Tile]]:
        """Yield each row as a list of Tiles, from top (y=0) to bottom."""
        for y in range(self.height):
            yield [self.tiles[y * self.width + x] for x in range(self.width)]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Grid(width={self.width}, height={self.height})"


# ---------------------------------------------------------------------------
# ASCII Renderer
# ---------------------------------------------------------------------------

def render_ascii(
    grid: Grid,
    agent_positions: Optional[dict[tuple[int, int], str]] = None,
    show_coords: bool = False,
) -> str:
    """Render a :class:`Grid` as a multi-line ASCII string.

    Parameters
    ----------
    grid:
        The world grid to render.
    agent_positions:
        Optional ``{(x, y): glyph}`` mapping.  When supplied, the given
        single-character glyph is drawn at that position instead of the
        tile's own symbol.  Useful for overlaying agents (``@``), items
        (``!``), or debug markers.
    show_coords:
        When ``True``, a column-index header and row-index prefix are
        added to aid spatial debugging.

    Returns
    -------
    str
        A single string; rows are separated by newlines.

    Example
    -------
    ::

        grid = build_default_market()
        print(render_ascii(grid, agent_positions={(10, 10): "@"}, show_coords=True))
    """
    agent_positions = agent_positions or {}
    lines: list[str] = []

    if show_coords:
        # Two-row column header showing tens then units digit.
        tens = "".join(
            str(x // 10) if x % 10 == 0 else " " for x in range(grid.width)
        )
        ones = "".join(str(x % 10) for x in range(grid.width))
        prefix = "   "              # aligns with the "NN " row prefix below
        lines.append(f"{prefix}{tens}")
        lines.append(f"{prefix}{ones}")

    for y, row in enumerate(grid.iter_rows()):
        row_str = ""
        for tile in row:
            pos = (tile.x, tile.y)
            row_str += agent_positions.get(pos, tile.symbol)
        if show_coords:
            lines.append(f"{y:2} {row_str}")
        else:
            lines.append(row_str)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal layout helpers
# ---------------------------------------------------------------------------

def _fill(
    tiles: list[Tile],
    width: int,
    x0: int, y0: int,
    x1: int, y1: int,
    tile_type: TileType,
) -> None:
    """In-place fill a rectangular region of a flat ``tiles`` list."""
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            tiles[y * width + x] = Tile(tile_type=tile_type, x=x, y=y)


# ---------------------------------------------------------------------------
# Default market layout factory
# ---------------------------------------------------------------------------

def build_default_market(width: int = 20, height: int = 12) -> Grid:
    """Build a default single-room marketplace layout.

    The room is enclosed by walls on all four sides.  Two horizontal shelf
    rows sit near the top of the interior with an aisle gap in the middle so
    agents can pass east–west.  A single entrance tile is centred in the
    south wall, giving agents a canonical spawn / exit point.

    Visual overview (20 × 12, ``#`` = wall, ``S`` = shelf,
    ``.`` = floor, ``E`` = entrance)::

        ####################
        #..................#
        #..SSSSSSSS.SSSSSS.#  ← shelf row y=2  (gap at mid_x)
        #..................#
        #..SSSSSSSS.SSSSSS.#  ← shelf row y=4  (gap at mid_x)
        #..................#
        #..................#
        #..................#
        #..................#
        #..................#
        #..................#
        #########E##########  ← entrance at mid_x

    Parameters
    ----------
    width:
        Total columns.  Must be ≥ 10.
    height:
        Total rows.  Must be ≥ 8.

    Raises
    ------
    ValueError
        If ``width`` or ``height`` is below the minimum.
    """
    if width < 10 or height < 8:
        raise ValueError("Market must be at least 10 columns wide and 8 rows tall.")

    # 1. Seed every cell as FLOOR.
    tiles: list[Tile] = [
        Tile(tile_type=TileType.FLOOR, x=x, y=y)
        for y in range(height)
        for x in range(width)
    ]

    # 2. Outer walls ── border all four edges.
    _fill(tiles, width, 0,         0,          width - 1, 0,          TileType.WALL)  # top
    _fill(tiles, width, 0,         height - 1, width - 1, height - 1, TileType.WALL)  # bottom
    _fill(tiles, width, 0,         0,          0,         height - 1, TileType.WALL)  # left
    _fill(tiles, width, width - 1, 0,          width - 1, height - 1, TileType.WALL)  # right

    # 3. Shelf rows inside the top half of the room.
    interior_x0 = 2
    interior_x1 = width - 2
    _fill(tiles, width, interior_x0, 2, interior_x1, 2, TileType.SHELF)  # upper shelf row
    _fill(tiles, width, interior_x0, 4, interior_x1, 4, TileType.SHELF)  # lower shelf row

    # 4. Aisle gaps so agents can pass east–west through each shelf row.
    mid_x = width // 2
    for shelf_y in (2, 4):
        tiles[shelf_y * width + mid_x] = Tile(
            tile_type=TileType.FLOOR, x=mid_x, y=shelf_y
        )

    # 5. Entrance ── centred gap in the south wall.
    entrance_x = mid_x
    entrance_y = height - 1
    tiles[entrance_y * width + entrance_x] = Tile(
        tile_type=TileType.ENTRANCE, x=entrance_x, y=entrance_y
    )

    return Grid(width=width, height=height, tiles=tiles)


# ---------------------------------------------------------------------------
# Module-level default instance
# ---------------------------------------------------------------------------

#: A ready-made 20×12 market grid – import and use directly for quick tests.
DEFAULT_MARKET: Grid = build_default_market()


# ---------------------------------------------------------------------------
# Smoke-test  (python -m backend.data.world  or  python world.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    grid = build_default_market()

    print("=" * 44)
    print("  Default market layout  (20 × 12)")
    print("=" * 44)
    print(render_ascii(grid, show_coords=True))

    # ── Basic queries ──────────────────────────────────────────────────
    entrance_tiles = grid.tiles_of_type(TileType.ENTRANCE)
    ex, ey = entrance_tiles[0].x, entrance_tiles[0].y
    print(f"\nEntrance tile  : ({ex}, {ey})")

    walkable_from_entrance = grid.walkable_neighbors(ex, ey)
    print(f"Walkable north : {[(t.x, t.y) for t in walkable_from_entrance]}")

    # ── Agent overlay demo ─────────────────────────────────────────────
    # Place a simulated agent one step north of the entrance.
    agent_x, agent_y = ex, ey - 1
    nearby_shelves = grid.adjacent_interactables(agent_x, agent_y)

    print(f"\nAgent standing at ({agent_x}, {agent_y})")
    if nearby_shelves:
        print(f"Adjacent shelves : {[(t.x, t.y) for t in nearby_shelves]}")
    else:
        print("No interactable tiles adjacent.")

    print("\nRendered with agent overlay (@):")
    print(render_ascii(grid, agent_positions={(agent_x, agent_y): "@"}, show_coords=True))
