"""Physics and mechanics for the ai-market-sim simulation.

This module implements the "Laws of the Universe":
- Walkability and collision detection
- Field of vision (line-of-sight)
- Movement validation
- Distance calculations

All spatial logic begins here.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.schema import Actor, Map


# ── Direction vectors for 8-directional movement ──────────────────────────────
DIRECTIONS_8 = {
	"north": (0, -1),
	"northeast": (1, -1),
	"east": (1, 0),
	"southeast": (1, 1),
	"south": (0, 1),
	"southwest": (-1, 1),
	"west": (-1, 0),
	"northwest": (-1, -1),
}


# ── Core Walkability & Collision ───────────────────────────────────────────────


def is_walkable(world_map: Map, x: int, y: int) -> bool:
	"""Check if a tile is walkable (not a wall, and in bounds).
	
	Args:
	    world_map: The Map object defining the world.
	    x: X coordinate.
	    y: Y coordinate.
	    
	Returns:
	    True if the tile at (x, y) is walkable (FLOOR or ENTRANCE).
	"""
	return world_map.is_walkable(x, y)


def get_blocking_actor(world_map: Map, x: int, y: int) -> Actor | None:
	"""Get the actor occupying a tile, if any.
	
	Args:
	    world_map: The Map object with actors.
	    x: X coordinate.
	    y: Y coordinate.
	    
	Returns:
	    The Actor at (x, y), or None if no actor is there.
	"""
	for actor in world_map.actors:
		if actor.x == x and actor.y == y:
			return actor
	return None


def can_move_to(world_map: Map, x: int, y: int, exclude_actor_id: str | None = None) -> bool:
	"""Check if a move to (x, y) is valid: walkable and not occupied.
	
	Args:
	    world_map: The Map object.
	    x: Target X coordinate.
	    y: Target Y coordinate.
	    exclude_actor_id: If provided, ignore this actor when checking occupancy.
	    
	Returns:
	    True if movement is allowed.
	"""
	if not is_walkable(world_map, x, y):
		return False
	
	blocking_actor = get_blocking_actor(world_map, x, y)
	if blocking_actor and blocking_actor.id != exclude_actor_id:
		return False
	
	return True


# ── Distance Calculations ──────────────────────────────────────────────────────


def distance_euclidean(x1: int, y1: int, x2: int, y2: int) -> float:
	"""Calculate Euclidean distance between two points.
	
	Args:
	    x1, y1: First point.
	    x2, y2: Second point.
	    
	Returns:
	    Euclidean distance.
	"""
	return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def distance_manhattan(x1: int, y1: int, x2: int, y2: int) -> int:
	"""Calculate Manhattan distance between two points.
	
	Args:
	    x1, y1: First point.
	    x2, y2: Second point.
	    
	Returns:
	    Manhattan distance (L1 norm).
	"""
	return abs(x1 - x2) + abs(y1 - y2)


def distance_chebyshev(x1: int, y1: int, x2: int, y2: int) -> int:
	"""Calculate Chebyshev distance (max of absolute differences, 8-directional).
	
	Args:
	    x1, y1: First point.
	    x2, y2: Second point.
	    
	Returns:
	    Chebyshev distance (L-infinity norm).
	"""
	return max(abs(x1 - x2), abs(y1 - y2))


# ── Field of Vision: Shadowcasting ────────────────────────────────────────────


@dataclass
class Viewport:
	"""Field of view results for an actor."""
	
	viewer_x: int
	viewer_y: int
	vision_range: int
	visible_tiles: set[tuple[int, int]]
	visible_actors: list[Actor]


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
	"""Bresenham line algorithm: all points on the line from (x0, y0) to (x1, y1).
	
	Used for line-of-sight checks.
	"""
	points = []
	dx = abs(x1 - x0)
	dy = abs(y1 - y0)
	sx = 1 if x1 > x0 else -1
	sy = 1 if y1 > y0 else -1
	
	if dx > dy:
		err = dx / 2.0
		y = y0
		for x in range(x0, x1 + sx, sx):
			points.append((x, y))
			err -= dy
			if err < 0:
				y += sy
				err += dx
	else:
		err = dy / 2.0
		x = x0
		for y in range(y0, y1 + sy, sy):
			points.append((x, y))
			err -= dx
			if err < 0:
				x += sx
				err += dy
	
	return points


def can_see(
	world_map: Map,
	viewer_x: int,
	viewer_y: int,
	target_x: int,
	target_y: int,
) -> bool:
	"""Check if target is visible from viewer position (line-of-sight, no walls blocking).
	
	Args:
	    world_map: The Map object.
	    viewer_x, viewer_y: Viewing position.
	    target_x, target_y: Target position to check visibility.
	    
	Returns:
	    True if the target is visible (line of sight not blocked by walls).
	"""
	# Get all points on the line from viewer to target.
	line = _bresenham_line(viewer_x, viewer_y, target_x, target_y)
	
	# Check all intermediate points (excluding the viewer, including target).
	for i, (x, y) in enumerate(line[1:], start=1):
		if not is_walkable(world_map, x, y):
			# Wall blocks the line; target is not visible unless it's the target.
			if (x, y) != (target_x, target_y):
				return False
	
	return True


def field_of_view(
	world_map: Map,
	actor_x: int,
	actor_y: int,
	vision_range: int = 10,
) -> set[tuple[int, int]]:
	"""Compute field of view (FOV) for an actor using shadowcasting.
	
	Simple implementation: all tiles within Manhattan distance and line-of-sight.
	
	Args:
	    world_map: The Map object.
	    actor_x, actor_y: Actor's position.
	    vision_range: Vision range in tiles (Manhattan distance).
	    
	Returns:
	    Set of (x, y) coordinates visible from the actor's position.
	"""
	visible_tiles: set[tuple[int, int]] = {(actor_x, actor_y)}
	
	# Check all tiles within vision range.
	for x in range(actor_x - vision_range, actor_x + vision_range + 1):
		for y in range(actor_y - vision_range, actor_y + vision_range + 1):
			# Skip the actor's own position.
			if (x, y) == (actor_x, actor_y):
				continue
			
			# Must be within map bounds.
			if not world_map.in_bounds(x, y):
				continue
			
			# Must be within vision range (Manhattan distance).
			if distance_manhattan(actor_x, actor_y, x, y) > vision_range:
				continue
			
			# Must have line of sight (no walls blocking).
			if can_see(world_map, actor_x, actor_y, x, y):
				visible_tiles.add((x, y))
	
	return visible_tiles


def get_visible_actors(
	world_map: Map,
	actor: Actor,
	vision_range: int = 10,
) -> list[Actor]:
	"""Get all actors visible from the given actor's position.
	
	Args:
	    world_map: The Map object with all actors.
	    actor: The viewing actor.
	    vision_range: Vision range in tiles.
	    
	Returns:
	    List of Actor objects visible from the viewing actor.
	"""
	visible_tiles = field_of_view(world_map, actor.x, actor.y, vision_range)
	visible_actors = []
	
	for other_actor in world_map.actors:
		if other_actor.id == actor.id:  # Don't include self.
			continue
		if (other_actor.x, other_actor.y) in visible_tiles:
			visible_actors.append(other_actor)
	
	return visible_actors


def get_visible_tiles_and_actors(
	world_map: Map,
	actor: Actor,
	vision_range: int = 10,
) -> Viewport:
	"""Full FOV query: visible tiles and actors from an actor's perspective.
	
	Args:
	    world_map: The Map object.
	    actor: The viewing actor.
	    vision_range: Vision range in tiles.
	    
	Returns:
	    Viewport object with all visible tiles and actors.
	"""
	visible_tiles = field_of_view(world_map, actor.x, actor.y, vision_range)
	visible_actors = get_visible_actors(world_map, actor, vision_range)
	
	return Viewport(
		viewer_x=actor.x,
		viewer_y=actor.y,
		vision_range=vision_range,
		visible_tiles=visible_tiles,
		visible_actors=visible_actors,
	)


# ── Pathfinding: Simple BFS ────────────────────────────────────────────────────


def breadth_first_search(
	world_map: Map,
	start_x: int,
	start_y: int,
	goal_x: int,
	goal_y: int,
	exclude_actor_id: str | None = None,
) -> list[tuple[int, int]] | None:
	"""Find shortest path from start to goal using BFS.
	
	Respects walkability and actor occupancy.
	
	Args:
	    world_map: The Map object.
	    start_x, start_y: Starting position.
	    goal_x, goal_y: Goal position.
	    exclude_actor_id: If provided, ignore this actor when checking for blocking.
	    
	Returns:
	    List of (x, y) coordinates from start to goal (inclusive), or None if no path.
	"""
	# Only reject if the goal is out-of-bounds or a wall — allow occupied tiles
	# so we can compute a path *toward* another actor without stepping on them.
	if not world_map.in_bounds(goal_x, goal_y) or not world_map.tile_at(goal_x, goal_y).walkable:
		return None
	
	from collections import deque
	
	queue: deque[tuple[int, int, list[tuple[int, int]]]] = deque(
		[(start_x, start_y, [(start_x, start_y)])]
	)
	visited: set[tuple[int, int]] = {(start_x, start_y)}
	
	while queue:
		x, y, path = queue.popleft()
		
		if (x, y) == (goal_x, goal_y):
			return path
		
		# Explore 8 adjacent tiles (N, NE, E, SE, S, SW, W, NW).
		for dx, dy in DIRECTIONS_8.values():
			nx, ny = x + dx, y + dy
			if (nx, ny) in visited:
				continue
			if not can_move_to(world_map, nx, ny, exclude_actor_id):
				continue
			
			visited.add((nx, ny))
			queue.append((nx, ny, path + [(nx, ny)]))
	
	return None


__all__ = [
	"DIRECTIONS_8",
	"is_walkable",
	"get_blocking_actor",
	"can_move_to",
	"distance_euclidean",
	"distance_manhattan",
	"distance_chebyshev",
	"can_see",
	"field_of_view",
	"get_visible_actors",
	"get_visible_tiles_and_actors",
	"breadth_first_search",
	"Viewport",
]
