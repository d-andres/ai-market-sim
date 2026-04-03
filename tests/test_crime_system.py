"""Crime, witness, and social reputation tests.

Covers:
1. Owned shelf pickup is allowed and recorded as theft.
2. Witnessed theft increases offender infamy and harms witness likeness.
3. Unwitnessed theft does not create direct witness punishment.
4. Witnessed high-value theft triggers guard intervention.
5. Retaliation within self-defense window is not recorded as a second assault crime.
6. Unprovoked witnessed assault triggers guard response.

No LLM calls are made; AI is disabled for all tests.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import Actor, ActorRole, Item, Map, PlannedAction, Tile, TileType, WorldItem
from src.simulation.engine import SimulationEngine


def _make_map(width: int = 12, height: int = 12) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                tt = TileType.WALL
            elif y == 1 and x in (1, 2, 3, 4):
                tt = TileType.SHOP
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _actor(
    aid: str,
    name: str,
    role: ActorRole,
    x: int,
    y: int,
    gold: int = 100,
    hp: int = 100,
    base_attack: int = 8,
    base_defense: int = 2,
    max_hp: int = 100,
) -> Actor:
    return Actor(
        id=aid,
        name=name,
        role=role,
        x=x,
        y=y,
        gold=gold,
        hp=hp,
        max_hp=max_hp,
        base_attack=base_attack,
        base_defense=base_defense,
    )


def _item(iid: str, name: str, price: int = 20) -> Item:
    return Item(id=iid, name=name, base_price=price, quantity=1, metadata={"category": "test"})


def _engine(world_map: Map) -> SimulationEngine:
    return SimulationEngine(world_map, enable_ai=False)


def test_theft_allowed_for_owned_shelf_item() -> None:
    world_map = _make_map()
    shopkeeper = _actor("sk", "Shopkeeper", ActorRole.SHOPKEEPER, x=3, y=3)
    thief = _actor("th", "Thief", ActorRole.PLAYER, x=2, y=2)
    world_map.actors = [shopkeeper, thief]
    world_map.world_items = [WorldItem(item=_item("ring", "Gold Ring", 60), x=2, y=1, owner_id=shopkeeper.id)]

    engine = _engine(world_map)
    result = engine._execute_action(thief, PlannedAction(action_type="pick_up", params={"item_id": "ring"}))

    assert "warning" in result.lower() or "steal" in result.lower(), result
    assert "ring" in [i.id for i in thief.inventory], f"Inventory: {[i.id for i in thief.inventory]}"
    assert "ring" not in [gi.item.id for gi in world_map.world_items]
    print("PASS  test_theft_allowed_for_owned_shelf_item")


def test_witnessed_theft_updates_infamy_and_likeness() -> None:
    world_map = _make_map()
    shopkeeper = _actor("sk", "Shopkeeper", ActorRole.SHOPKEEPER, x=3, y=3)
    thief = _actor("th", "Thief", ActorRole.PLAYER, x=2, y=2)
    witness = _actor("wi", "Witness", ActorRole.PLAYER, x=4, y=2)
    world_map.actors = [shopkeeper, thief, witness]
    world_map.world_items = [WorldItem(item=_item("amulet", "Amulet", 80), x=2, y=1, owner_id=shopkeeper.id)]

    engine = _engine(world_map)
    engine._execute_action(thief, PlannedAction(action_type="pick_up", params={"item_id": "amulet"}))

    assert thief.infamy > 0, f"Infamy should increase on witnessed theft, got {thief.infamy}"
    assert witness.likeness.get(thief.id, 0) < 0, f"Witness likeness should decrease: {witness.likeness}"
    crime_events = [e for e in engine.events if e.event_type == "crime"]
    assert crime_events, "Expected crime event"
    assert any(e.data.get("crime_type") == "theft" for e in crime_events)
    print("PASS  test_witnessed_theft_updates_infamy_and_likeness")


def test_unwitnessed_theft_no_direct_witnesses() -> None:
    world_map = _make_map(width=24, height=24)
    shopkeeper = _actor("sk", "Shopkeeper", ActorRole.SHOPKEEPER, x=20, y=20)
    thief = _actor("th", "Thief", ActorRole.PLAYER, x=2, y=2)
    world_map.actors = [shopkeeper, thief]
    world_map.world_items = [WorldItem(item=_item("gem", "Gem", 50), x=2, y=1, owner_id=shopkeeper.id)]

    engine = _engine(world_map)
    engine._execute_action(thief, PlannedAction(action_type="pick_up", params={"item_id": "gem"}))

    # No direct witnesses should mean no direct notoriety gain in this model.
    assert thief.infamy == 0, f"Unwitnessed theft should not add public infamy, got {thief.infamy}"
    crime_events = [e for e in engine.events if e.event_type == "crime"]
    assert crime_events, "Crime event should still be recorded"
    assert crime_events[-1].data.get("witnesses") == [], f"Expected no witnesses: {crime_events[-1].data}"
    print("PASS  test_unwitnessed_theft_no_direct_witnesses")


def test_guard_intervenes_on_witnessed_high_value_theft() -> None:
    world_map = _make_map()
    shopkeeper = _actor("sk", "Shopkeeper", ActorRole.SHOPKEEPER, x=3, y=3)
    thief = _actor("th", "Thief", ActorRole.PLAYER, x=2, y=2, hp=100)
    guard = _actor("gd", "Guard", ActorRole.GUARD, x=3, y=2, base_attack=12)
    world_map.actors = [shopkeeper, thief, guard]
    world_map.world_items = [WorldItem(item=_item("crown", "Crown", 1200), x=2, y=1, owner_id=shopkeeper.id)]

    engine = _engine(world_map)
    engine._execute_action(thief, PlannedAction(action_type="pick_up", params={"item_id": "crown"}))

    guard_events = [e for e in engine.events if e.event_type.startswith("guard_")]
    assert guard_events, "Expected guard intervention event"
    assert thief.hp < 100 or thief.gold < 100 or thief.needs_replan, "Guard should have penalized or warned offender"
    print("PASS  test_guard_intervenes_on_witnessed_high_value_theft")


def test_self_defense_not_double_crime() -> None:
    world_map = _make_map()
    attacker = _actor("a", "Aggressor", ActorRole.PLAYER, x=5, y=5)
    defender = _actor("d", "Defender", ActorRole.PLAYER, x=6, y=5)
    world_map.actors = [attacker, defender]
    engine = _engine(world_map)

    engine._execute_action(attacker, PlannedAction(action_type="attack", params={"target_actor_id": defender.id}))
    crimes_after_first = len([e for e in engine.events if e.event_type == "crime" and e.data.get("crime_type") == "assault"])

    engine._execute_action(defender, PlannedAction(action_type="attack", params={"target_actor_id": attacker.id}))
    crimes_after_second = len([e for e in engine.events if e.event_type == "crime" and e.data.get("crime_type") == "assault"])

    assert crimes_after_first >= 1, "First unprovoked attack should be assault crime"
    assert crimes_after_second == crimes_after_first, "Retaliation should not add another assault crime"
    print("PASS  test_self_defense_not_double_crime")


def test_guard_intervenes_on_witnessed_assault() -> None:
    world_map = _make_map()
    guard = _actor("gd", "Guard", ActorRole.GUARD, x=4, y=5, base_attack=12)
    attacker = _actor("a", "Aggressor", ActorRole.PLAYER, x=5, y=5)
    victim = _actor("v", "Victim", ActorRole.PLAYER, x=6, y=5)
    world_map.actors = [guard, attacker, victim]

    engine = _engine(world_map)
    engine._execute_action(attacker, PlannedAction(action_type="attack", params={"target_actor_id": victim.id}))

    guard_events = [e for e in engine.events if e.event_type.startswith("guard_")]
    assert guard_events, "Guard should respond to witnessed assault"
    assert attacker.infamy > 0, "Assault should increase infamy when witnessed"
    print("PASS  test_guard_intervenes_on_witnessed_assault")


if __name__ == "__main__":
    tests = [
        test_theft_allowed_for_owned_shelf_item,
        test_witnessed_theft_updates_infamy_and_likeness,
        test_unwitnessed_theft_no_direct_witnesses,
        test_guard_intervenes_on_witnessed_high_value_theft,
        test_self_defense_not_double_crime,
        test_guard_intervenes_on_witnessed_assault,
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
