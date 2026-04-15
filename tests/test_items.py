"""Item system integration tests.

Covers:
1. Shelf visibility  — actors can see items placed on shop tiles in their FoV
2. Shelf trade       — trading with a shopkeeper transfers a shelf item to the buyer
3. Inventory privacy — carried items are NOT visible to other actors via the ground list
4. Ground place/pickup — items placed on the floor appear to observers and can be picked up

No LLM calls are made; AI is disabled for all tests.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import (
    Actor,
    ActorRole,
    WorldItem,
    Item,
    Map,
    PlannedAction,
    Tile,
    TileType,
    TradeProposal,
)
from src.simulation.engine import SimulationEngine
from src.agents.brain import ObserveSurroundingsTool, RelationshipMemory
from data.map import render_ascii


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_map(width: int = 10, height: int = 10) -> Map:
    """Build a minimal flat map with two shop tiles at (1,1) and (2,1)."""
    tiles = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                tt = TileType.WALL
            elif y == 1 and x in (1, 2):
                tt = TileType.SHOP
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _shopkeeper(x: int = 3, y: int = 3) -> Actor:
    return Actor(id="shopkeeper_1", name="Merchant Elara", role=ActorRole.SHOPKEEPER, x=x, y=y, gold=100)


def _player(x: int = 4, y: int = 4) -> Actor:
    return Actor(id="player_1", name="Adventurer", role=ActorRole.PLAYER, x=x, y=y, gold=50)


def _guard(x: int = 7, y: int = 7) -> Actor:
    return Actor(id="guard_1", name="Guard Thorne", role=ActorRole.GUARD, x=x, y=y, gold=20)


def _potion() -> Item:
    return Item(id="potion_health", name="Health Potion", description="Restores HP.", base_price=10, quantity=1)


def _sword() -> Item:
    return Item(id="sword_short", name="Short Sword", description="A blade.", base_price=50, quantity=1)


def _engine(world_map: Map) -> SimulationEngine:
    return SimulationEngine(world_map, tick_rate=1.0, enable_ai=False)


# ---------------------------------------------------------------------------
# Test 1: shelf visibility in observation tool
# ---------------------------------------------------------------------------

def test_shelf_item_visible_in_observation():
    """An actor should see a shopkeeper's shelf item when within discovery range (dist <= 2)."""
    world_map = _make_map()
    sk = _shopkeeper(x=3, y=3)
    player = _player(x=2, y=2)  # within DISCOVERY_RANGE (dist 1 to shop tile at 1,1)
    world_map.actors = [sk, player]

    # Place a potion on shelf tile (1,1) owned by shopkeeper
    potion = _potion()
    world_map.world_items = [WorldItem(item=potion, x=1, y=1, owner_id=sk.id)]

    memory = RelationshipMemory()
    obs_tool = ObserveSurroundingsTool(actor=player, world_map=world_map, memory=memory)
    output = obs_tool.forward(vision_range=10)

    assert "Health Potion" in output, f"Expected shelf item in observation:\n{output}"
    assert "potion_health" in output, f"Expected item id in observation:\n{output}"
    assert "[shopkeeper]" in output.lower(), f"Expected visible actor role in observation:\n{output}"
    assert "owned by" in output.lower(), f"Expected item ownership annotation in observation:\n{output}"
    assert "carrying/selling" in output, f"Expected carrying/selling label:\n{output}"
    print("PASS  test_shelf_item_visible_in_observation")
    return output


def test_shelf_item_hidden_when_far():
    """Shop shelf items on the ground should show as hidden when actor is beyond discovery range."""
    world_map = _make_map()
    # Shopkeeper far away so their carrying/selling section is not visible either
    sk = _shopkeeper(x=1, y=1)
    player = _player(x=5, y=5)  # dist 4 to shelf at (1,1) — beyond DISCOVERY_RANGE
    world_map.actors = [sk, player]

    potion = _potion()
    world_map.world_items = [WorldItem(item=potion, x=1, y=1, owner_id=sk.id)]

    memory = RelationshipMemory()
    obs_tool = ObserveSurroundingsTool(actor=player, world_map=world_map, memory=memory)
    output = obs_tool.forward(vision_range=10)

    assert "move closer to inspect" in output.lower(), f"Expected hidden shelf message:\n{output}"
    # Item name should only appear in the carrying/selling section (if shopkeeper visible within range)
    # but NOT in the ground items section
    ground_section = output.split("Items in view")[1] if "Items in view" in output else ""
    assert "Health Potion" not in ground_section, f"Item details should be hidden in ground items at distance:\n{output}"
    print("PASS  test_shelf_item_hidden_when_far")


# ---------------------------------------------------------------------------
# Test 2: shelf trade — player buys from shopkeeper's shelf
# ---------------------------------------------------------------------------

def test_shelf_trade_transfers_item():
    """propose_trade targeting a shelf item removes it from the shelf and gives it to buyer."""
    world_map = _make_map()
    sk = _shopkeeper(x=3, y=3)
    player = _player(x=4, y=4)  # within Chebyshev 2 of shopkeeper
    world_map.actors = [sk, player]

    potion = _potion()
    world_map.world_items = [WorldItem(item=potion, x=1, y=1, owner_id=sk.id)]

    engine = _engine(world_map)

    # Manually build and execute a trade proposal: player offers 10g for the potion
    proposal = TradeProposal(
        proposer_id=player.id,
        target_id=sk.id,
        offered_gold=10,
        offered_items=[],
        requested_items=[potion],
        requested_gold=0,
    )

    # Directly call _resolve_trade (bypasses LLM evaluation)
    err = engine._resolve_trade(proposal, player, sk)

    assert err is None, f"Trade failed: {err}"
    # Item should be in player inventory
    player_item_ids = [i.id for i in player.inventory]
    assert "potion_health" in player_item_ids, f"Player inventory: {player_item_ids}"
    # Item should be GONE from ground
    remaining = [gi.item.id for gi in world_map.world_items]
    assert "potion_health" not in remaining, f"Item still on shelf: {remaining}"
    # Gold transferred
    assert player.gold == 40, f"Player gold: {player.gold}"
    assert sk.gold == 110, f"Shopkeeper gold: {sk.gold}"
    print("PASS  test_shelf_trade_transfers_item")


# ---------------------------------------------------------------------------
# Test 3: inventory privacy — carried items not in ground list
# ---------------------------------------------------------------------------

def test_inventory_not_in_world_items():
    """Items in an actor's carried inventory must NOT appear in world_items."""
    world_map = _make_map()
    player = _player(x=4, y=4)
    sword = _sword()
    player.inventory = [sword]  # sword is carried, not placed
    world_map.actors = [player]

    # No world items at all
    assert len(world_map.world_items) == 0

    # Observer should not see the sword as a world item
    guard = _guard(x=5, y=5)
    world_map.actors = [player, guard]
    memory = RelationshipMemory()
    obs_tool = ObserveSurroundingsTool(actor=guard, world_map=world_map, memory=memory)
    output = obs_tool.forward(vision_range=10)

    # Sword should appear as a *carried* item (in the actor line)
    assert "Short Sword" in output, f"Guard should see player is carrying sword:\n{output}"
    # But it must NOT appear under the ground-items section
    assert "Items on the ground" not in output, f"Sword should not appear as floor item:\n{output}"
    print("PASS  test_inventory_not_in_world_items")


# ---------------------------------------------------------------------------
# Test 4a: place item — appears on ground, visible to others
# ---------------------------------------------------------------------------

def test_place_item_visible_on_ground():
    """An item placed on a floor tile becomes a floor item visible to nearby actors."""
    world_map = _make_map()
    player = _player(x=4, y=4)
    guard = _guard(x=5, y=5)
    sword = _sword()
    player.inventory = [sword]
    world_map.actors = [player, guard]

    engine = _engine(world_map)

    # Player places the sword on tile (4,3) — adjacent floor tile
    place_action = PlannedAction(
        action_type="place",
        params={"item_id": "sword_short", "x": 4, "y": 3},
    )
    result = engine._execute_action(player, place_action)
    assert "placed" in result.lower(), f"Unexpected place result: {result}"

    # Item must be on the ground now
    ground_ids = [gi.item.id for gi in world_map.world_items]
    assert "sword_short" in ground_ids, f"Expected sword on ground, got: {ground_ids}"
    # owner_id should be None for a floor tile
    gi = next(g for g in world_map.world_items if g.item.id == "sword_short")
    assert gi.owner_id is None, f"Floor item should have no owner, got: {gi.owner_id}"
    # Player inventory should be empty
    assert player.inventory == [], f"Player inventory: {player.inventory}"

    # Guard's observation should list the sword as a floor item
    memory = RelationshipMemory()
    obs_tool = ObserveSurroundingsTool(actor=guard, world_map=world_map, memory=memory)
    output = obs_tool.forward(vision_range=10)
    assert "Items in view" in output, f"Guard should see floor items:\n{output}"
    assert "Short Sword" in output, f"Guard should see the sword:\n{output}"
    print("PASS  test_place_item_visible_on_ground")
    return output


# ---------------------------------------------------------------------------
# Test 4b: pick_up — actor retrieves a floor item
# ---------------------------------------------------------------------------

def test_pickup_floor_item():
    """An actor can pick up an unowned item from an adjacent tile."""
    world_map = _make_map()
    guard = _guard(x=5, y=5)
    world_map.actors = [guard]

    sword = _sword()
    # Place sword on tile (5,4) — adjacent to guard (Chebyshev 1)
    world_map.world_items = [WorldItem(item=sword, x=5, y=4, owner_id=None)]

    engine = _engine(world_map)
    pickup_action = PlannedAction(
        action_type="pick_up",
        params={"item_id": "sword_short"},
    )
    result = engine._execute_action(guard, pickup_action)
    assert "picked up" in result.lower(), f"Unexpected pickup result: {result}"

    # Sword should be in guard's inventory
    guard_item_ids = [i.id for i in guard.inventory]
    assert "sword_short" in guard_item_ids, f"Guard inventory: {guard_item_ids}"
    # And no longer on the ground
    remaining = [gi.item.id for gi in world_map.world_items]
    assert "sword_short" not in remaining, f"Sword still on ground: {remaining}"
    print("PASS  test_pickup_floor_item")


# ---------------------------------------------------------------------------
# Test 4c: pick_up blocked for owned shelf item
# ---------------------------------------------------------------------------

def test_pickup_owned_shelf_item_counts_as_theft():
    """A player can pick up an owned shelf item, but it is treated as theft."""
    world_map = _make_map()
    sk = _shopkeeper(x=3, y=3)
    player = _player(x=2, y=2)  # adjacent to shelf tile (1,1) and (2,1)
    world_map.actors = [sk, player]

    potion = _potion()
    world_map.world_items = [WorldItem(item=potion, x=2, y=1, owner_id=sk.id)]

    engine = _engine(world_map)
    pickup_action = PlannedAction(
        action_type="pick_up",
        params={"item_id": "potion_health"},
    )
    result = engine._execute_action(player, pickup_action)

    # Theft succeeds — item removed from shelf and moved to player inventory
    remaining = [gi.item.id for gi in world_map.world_items]
    assert "potion_health" not in remaining, f"Owned item should be removed from shelf: {remaining}"
    player_items = [i.id for i in player.inventory]
    assert "potion_health" in player_items, f"Player should have stolen item: {player_items}"
    assert "warning" in result.lower() or "steal" in result.lower(), (
        f"Expected a theft warning message, got: {result}"
    )
    assert player.infamy > 0, f"Theft should increase infamy, got: {player.infamy}"
    print("PASS  test_pickup_owned_shelf_item_counts_as_theft")


# ---------------------------------------------------------------------------
# Test 5: ASCII render — stocked shelves show 's', floor items show 'i'
# ---------------------------------------------------------------------------

def test_ascii_render_symbols():
    """Stocked shelf tiles render as 's', floor items as 'i'."""
    world_map = _make_map()
    sk = _shopkeeper(x=3, y=3)
    world_map.actors = [sk]

    potion = _potion()
    sword = _sword()
    # Shelf item on (1,1) — shop tile
    world_map.world_items = [
        WorldItem(item=potion, x=1, y=1, owner_id=sk.id),
        WorldItem(item=sword, x=3, y=5, owner_id=None),      # floor item
    ]

    ascii_map = render_ascii(world_map, show_actors=False)
    rows = ascii_map.splitlines()

    shelf_char = rows[1][1]   # (x=1, y=1) — shop tile with stocked item
    floor_char = rows[5][3]   # (x=3, y=5) — floor tile with item
    empty_shelf = rows[1][2]  # (x=2, y=1) — shop tile WITHOUT an item

    assert shelf_char == "s", f"Stocked shelf should render 's', got '{shelf_char}'"
    assert floor_char == "i", f"Floor item should render 'i', got '{floor_char}'"
    assert empty_shelf == "S", f"Empty shop tile should render 'S', got '{empty_shelf}'"
    print("PASS  test_ascii_render_symbols")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = []

    tests = [
        test_shelf_item_visible_in_observation,
        test_shelf_trade_transfers_item,
        test_inventory_not_in_world_items,
        test_place_item_visible_on_ground,
        test_pickup_floor_item,
        test_pickup_owned_shelf_item_counts_as_theft,
        test_ascii_render_symbols,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {t.__name__}: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
