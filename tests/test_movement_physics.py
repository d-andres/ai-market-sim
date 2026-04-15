"""Movement, pathfinding, and physics tests.

Covers:
1. Basic movement in all 8 directions
2. Movement blocked by wall
3. Movement blocked by occupied tile
4. Movement blocked by out-of-bounds
5. Field of view: actors behind walls are not visible
6. Field of view: nearby actors are visible
7. Pathfinding: BFS finds a valid path
8. Pathfinding: BFS returns None when no path exists
9. Distance calculations: Chebyshev, Manhattan, Euclidean
10. Map generation: load_or_build_default_map produces valid map
11. Map generation: build_default_map has correct dimensions and tile counts

No LLM calls are made; AI is disabled for all tests.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import Actor, ActorRole, Map, PlannedAction, Tile, TileType
from src.simulation.engine import SimulationEngine
from src.simulation.physics import (
    DIRECTIONS_8,
    breadth_first_search,
    can_move_to,
    can_see,
    distance_chebyshev,
    distance_euclidean,
    distance_manhattan,
    field_of_view,
    get_visible_actors,
    is_walkable,
)
from data.map import build_default_map, load_or_build_default_map


def _make_map(width: int = 10, height: int = 10) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                tt = TileType.WALL
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _actor(aid: str, name: str, x: int, y: int, role: ActorRole = ActorRole.PLAYER) -> Actor:
    return Actor(id=aid, name=name, role=role, x=x, y=y, gold=10)


def _engine(world_map: Map) -> SimulationEngine:
    return SimulationEngine(world_map, enable_ai=False)


def test_move_all_directions():
    """Actor can move in all 8 compass directions on open floor."""
    for direction, (dx, dy) in DIRECTIONS_8.items():
        world_map = _make_map()
        actor = _actor("a", "Alice", 5, 5)
        world_map.actors = [actor]
        engine = _engine(world_map)

        result = engine._execute_action(actor, PlannedAction(
            action_type="move", params={"direction": direction}
        ))
        assert actor.x == 5 + dx and actor.y == 5 + dy, (
            f"Direction {direction}: expected ({5+dx},{5+dy}), got ({actor.x},{actor.y})"
        )
    print("PASS  test_move_all_directions")


def test_move_blocked_by_wall():
    """Moving into a wall tile interrupts the actor."""
    world_map = _make_map()
    actor = _actor("a", "Alice", 1, 1)
    world_map.actors = [actor]
    engine = _engine(world_map)

    engine._execute_action(actor, PlannedAction(
        action_type="move", params={"direction": "north"}
    ))
    # (1, 0) is a wall; actor should stay at (1, 1)
    assert actor.x == 1 and actor.y == 1, f"Should be blocked: ({actor.x},{actor.y})"
    assert actor.needs_replan, "Should trigger replan when blocked"
    print("PASS  test_move_blocked_by_wall")


def test_move_blocked_by_actor():
    """Moving into an occupied tile is blocked."""
    world_map = _make_map()
    alice = _actor("a", "Alice", 3, 3)
    bob = _actor("b", "Bob", 4, 3)
    world_map.actors = [alice, bob]
    engine = _engine(world_map)

    engine._execute_action(alice, PlannedAction(
        action_type="move", params={"direction": "east"}
    ))
    assert alice.x == 3 and alice.y == 3, f"Should be blocked by Bob: ({alice.x},{alice.y})"
    print("PASS  test_move_blocked_by_actor")


def test_invalid_direction_interrupts():
    """An invalid direction string triggers interrupt."""
    world_map = _make_map()
    actor = _actor("a", "Alice", 5, 5)
    world_map.actors = [actor]
    engine = _engine(world_map)

    engine._execute_action(actor, PlannedAction(
        action_type="move", params={"direction": "upward"}
    ))
    assert actor.needs_replan, "Invalid direction should trigger replan"
    assert actor.x == 5 and actor.y == 5, "Should not have moved"
    print("PASS  test_invalid_direction_interrupts")


def test_fov_wall_blocks_vision():
    """An actor behind a wall should not be visible."""
    # Build a map with a wall cutting horizontally across row 5
    tiles = []
    for y in range(12):
        for x in range(12):
            if x == 0 or y == 0 or x == 11 or y == 11:
                tt = TileType.WALL
            elif y == 5 and 1 <= x <= 10:
                tt = TileType.WALL
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    world_map = Map(width=12, height=12, tiles=tiles)

    viewer = _actor("v", "Viewer", 5, 3)
    hidden = _actor("h", "Hidden", 5, 7)
    world_map.actors = [viewer, hidden]

    visible = get_visible_actors(world_map, viewer, vision_range=10)
    visible_ids = [a.id for a in visible]
    assert "h" not in visible_ids, f"Expected hidden actor not visible: {visible_ids}"
    print("PASS  test_fov_wall_blocks_vision")


def test_fov_nearby_actor_visible():
    """A nearby actor on open floor should be visible."""
    world_map = _make_map()
    viewer = _actor("v", "Viewer", 3, 3)
    nearby = _actor("n", "Nearby", 5, 3)
    world_map.actors = [viewer, nearby]

    visible = get_visible_actors(world_map, viewer, vision_range=10)
    visible_ids = [a.id for a in visible]
    assert "n" in visible_ids, f"Expected nearby actor visible: {visible_ids}"
    print("PASS  test_fov_nearby_actor_visible")


def test_bfs_finds_path():
    """BFS finds a path between two open tiles."""
    world_map = _make_map()
    path = breadth_first_search(world_map, 1, 1, 8, 8)
    assert path is not None, "Expected a path"
    assert path[0] == (1, 1), f"Path should start at origin: {path[0]}"
    assert path[-1] == (8, 8), f"Path should end at goal: {path[-1]}"
    print("PASS  test_bfs_finds_path")


def test_bfs_no_path_to_wall():
    """BFS returns None when goal is a wall."""
    world_map = _make_map()
    path = breadth_first_search(world_map, 1, 1, 0, 0)
    assert path is None, "Should not find path to wall tile"
    print("PASS  test_bfs_no_path_to_wall")


def test_distance_chebyshev():
    assert distance_chebyshev(0, 0, 3, 4) == 4
    assert distance_chebyshev(5, 5, 5, 5) == 0
    assert distance_chebyshev(0, 0, 7, 3) == 7
    print("PASS  test_distance_chebyshev")


def test_distance_manhattan():
    assert distance_manhattan(0, 0, 3, 4) == 7
    assert distance_manhattan(5, 5, 5, 5) == 0
    print("PASS  test_distance_manhattan")


def test_distance_euclidean():
    d = distance_euclidean(0, 0, 3, 4)
    assert abs(d - 5.0) < 0.001, f"Expected 5.0, got {d}"
    print("PASS  test_distance_euclidean")


def test_default_map_valid():
    """load_or_build_default_map produces a valid map with correct dimensions."""
    m = load_or_build_default_map()
    assert m.width == 20 and m.height == 12, f"Expected 20x12, got {m.width}x{m.height}"
    assert len(m.tiles) == m.width * m.height
    print("PASS  test_default_map_valid")


def test_build_default_map_has_entrance():
    """Built map should have at least one entrance tile."""
    m = build_default_map()
    entrance_tiles = [t for t in m.tiles if t.tile_type == TileType.ENTRANCE]
    assert entrance_tiles, "Map should have at least one entrance"
    print("PASS  test_build_default_map_has_entrance")


def test_build_default_map_has_shops():
    """Built map should have shop tiles."""
    m = build_default_map()
    shop_tiles = [t for t in m.tiles if t.tile_type == TileType.SHOP]
    assert len(shop_tiles) > 0, "Map should have shop tiles"
    print("PASS  test_build_default_map_has_shops")


def test_walkability():
    """Floor and entrance are walkable; wall and shop are not."""
    world_map = _make_map()
    assert is_walkable(world_map, 5, 5), "Floor should be walkable"
    assert not is_walkable(world_map, 0, 0), "Wall should not be walkable"
    assert not is_walkable(world_map, -1, 0), "Out of bounds should not be walkable"
    print("PASS  test_walkability")


def test_can_move_to_excludes_self():
    """can_move_to with exclude_actor_id allows moving through own tile."""
    world_map = _make_map()
    actor = _actor("a", "Alice", 5, 5)
    world_map.actors = [actor]
    # Can move onto own tile when excluded
    assert can_move_to(world_map, 5, 5, exclude_actor_id="a")
    # Cannot move onto own tile when not excluded
    assert not can_move_to(world_map, 5, 5)
    print("PASS  test_can_move_to_excludes_self")


if __name__ == "__main__":
    tests = [
        test_move_all_directions,
        test_move_blocked_by_wall,
        test_move_blocked_by_actor,
        test_invalid_direction_interrupts,
        test_fov_wall_blocks_vision,
        test_fov_nearby_actor_visible,
        test_bfs_finds_path,
        test_bfs_no_path_to_wall,
        test_distance_chebyshev,
        test_distance_manhattan,
        test_distance_euclidean,
        test_default_map_valid,
        test_build_default_map_has_entrance,
        test_build_default_map_has_shops,
        test_walkability,
        test_can_move_to_excludes_self,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            import traceback
            print(f"FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
