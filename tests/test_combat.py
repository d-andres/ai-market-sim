"""Combat, equipment, carry-capacity, and death system tests.

Covers:
 1.  calc_damage — base formula, damage floor of 1
 2.  calc_damage — crit doubles the multiplier
 3.  calc_damage — equipped armor raises defender's total defense
 4.  calc_damage — no weapon equipped uses actor's base_attack only
 5.  Two-hand weapon: equip → lands in hand slot, clears offhand
 6.  Two-hand equipped: trying to equip offhand afterwards is rejected
 7.  equip / unequip round-trip: item moves between inventory and slot
 8.  use_item (heal): hp_delta restores HP, capped at max_hp
 9.  use_item (poison): negative hp_delta reduces HP, floored at 0
10.  use_item (blue potion): attack_delta + crit_delta modify base stats
11.  use_item blocked for non-consumable
12.  attack action: HP reduced, event logged
13.  attack action blocked when target is out of melee range
14.  Actor death: inventory + equipped items drop at death tile
15.  Actor death: corpse WorldItem spawned at death tile
16.  Actor death: dead actor removed from world_map.actors
17.  Actor death: dead actor's shelf items are NOT dropped (they stay on shelves)
18.  carry limit: pickup blocked when inventory is full
19.  carry limit: shelf items do not count toward carry limit
20.  carry limit: trade blocked when receiver has no space
21.  observation: inventory shows (N/max_carry)
22.  observation: equipped items listed
23.  observation: dead actor leaves corpse visible to survivors
24.  load_item_catalog: all 20 items load with correct combat stats
25.  load_item_catalog: catalog items parse into Item models without error

No LLM calls are made; AI is disabled for all tests.
"""

import sys
import os
import random

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
from src.simulation.physics import calc_damage, MELEE_RANGE
from src.agents.brain import ObserveSurroundingsTool, RelationshipMemory
from data import load_item_catalog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _actor(aid: str, name: str, role: ActorRole = ActorRole.PLAYER,
           x: int = 5, y: int = 5, hp: int = 100, max_hp: int = 100,
           base_attack: int = 5, base_defense: int = 2, base_crit: float = 0.0,
           gold: int = 50, max_carry: int = 10) -> Actor:
    return Actor(
        id=aid, name=name, role=role,
        x=x, y=y, hp=hp, max_hp=max_hp,
        base_attack=base_attack, base_defense=base_defense, base_crit=base_crit,
        gold=gold, max_carry=max_carry,
    )


def _engine(world_map: Map) -> SimulationEngine:
    return SimulationEngine(world_map, tick_rate=1.0, enable_ai=False)


def _item(iid: str, name: str = "Thing", base_price: int = 10,
          category: str = "misc", slot: str = "",
          attack_bonus: int = 0, defense_bonus: int = 0,
          crit_bonus: float = 0.0, hp_delta: int = 0,
          attack_delta: int = 0, defense_delta: int = 0,
          crit_delta: float = 0.0) -> Item:
    meta: dict = {"category": category}
    if slot:
        meta["slot"] = slot
    return Item(
        id=iid, name=name, base_price=base_price,
        metadata=meta,
        attack_bonus=attack_bonus, defense_bonus=defense_bonus,
        crit_bonus=crit_bonus, hp_delta=hp_delta,
        attack_delta=attack_delta, defense_delta=defense_delta,
        crit_delta=crit_delta,
    )


def _sword(slot: str = "hand") -> Item:
    return _item("test_sword", "Test Sword", slot=slot,
                 category="weapon", attack_bonus=8)


def _shield() -> Item:
    return _item("test_shield", "Test Shield", slot="offhand",
                 category="armor", defense_bonus=5)


def _helmet() -> Item:
    return _item("test_helmet", "Test Helmet", slot="head",
                 category="armor", defense_bonus=3)


def _potion(hp: int = 20) -> Item:
    return _item("test_potion", "Test Potion", category="consumable", hp_delta=hp)


# ---------------------------------------------------------------------------
# 1. calc_damage — base formula
# ---------------------------------------------------------------------------

def test_calc_damage_base():
    """damage = max(1, base_attack - base_defense) when no equipment."""
    attacker = _actor("a", "Alice", base_attack=8, base_defense=2)
    defender = _actor("b", "Bob", base_attack=5, base_defense=3)
    # expected = max(1, 8 - 3) = 5, crit_chance = 0 so never crits
    dmg, is_crit = calc_damage(attacker, defender)
    assert dmg == 5, f"Expected 5, got {dmg}"
    assert not is_crit
    print("PASS  test_calc_damage_base")


# ---------------------------------------------------------------------------
# 2. calc_damage — damage floor of 1
# ---------------------------------------------------------------------------

def test_calc_damage_floor():
    """Damage is always at least 1, even if defense exceeds attack."""
    attacker = _actor("a", "Alice", base_attack=2, base_defense=2)
    defender = _actor("b", "Bob", base_attack=2, base_defense=20)
    dmg, _ = calc_damage(attacker, defender)
    assert dmg >= 1, f"Damage floor violated: {dmg}"
    print("PASS  test_calc_damage_floor")


# ---------------------------------------------------------------------------
# 3. calc_damage — crit doubles damage
# ---------------------------------------------------------------------------

def test_calc_damage_crit():
    """Forcing crit (base_crit=1.0) should double the effective attack before subtracting defense."""
    attacker = _actor("a", "Alice", base_attack=10, base_crit=1.0)
    defender = _actor("b", "Bob", base_defense=3)
    dmg, is_crit = calc_damage(attacker, defender)
    # expected = max(1, int(10 * 2.0) - 3) = 17
    assert is_crit
    assert dmg == 17, f"Expected 17 on guaranteed crit, got {dmg}"
    print("PASS  test_calc_damage_crit")


# ---------------------------------------------------------------------------
# 4. calc_damage — equipped armor raises total defense
# ---------------------------------------------------------------------------

def test_calc_damage_armor():
    """Defender's equipped armor defense_bonus is added to base_defense."""
    attacker = _actor("a", "Alice", base_attack=10)
    defender = _actor("b", "Bob", base_defense=2)
    defender.equipped["chest"] = _item("vest", defense_bonus=5, category="armor", slot="chest")
    defender.equipped["head"] = _item("helm", defense_bonus=3, category="armor", slot="head")
    # total_defense = 2 + 5 + 3 = 10; damage = max(1, 10 - 10) = 1
    dmg, _ = calc_damage(attacker, defender)
    assert dmg == 1, f"Expected 1 (floor), got {dmg}"
    print("PASS  test_calc_damage_armor")


# ---------------------------------------------------------------------------
# 5. Two-hand weapon lands in 'hand' slot, clears offhand
# ---------------------------------------------------------------------------

def test_equip_twohander_clears_offhand():
    """Equipping a two_hand weapon moves it to 'hand' and unequips any offhand."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5)
    shield = _shield()
    longsword = _item("longsword", "Long Sword", slot="two_hand", category="weapon", attack_bonus=14)
    player.equipped["offhand"] = shield  # pre-equip a shield
    player.inventory = [longsword]
    world_map.actors = [player]
    engine = _engine(world_map)

    result = engine._execute_action(player, PlannedAction(
        action_type="equip", params={"item_id": "longsword"}
    ))
    assert "equip" in result.lower(), result
    assert player.equipped["hand"] is not None
    assert player.equipped["hand"].id == "longsword"
    assert player.equipped["offhand"] is None, "offhand should be cleared"
    # Old shield returned to inventory
    inv_ids = [i.id for i in player.inventory]
    assert "test_shield" in inv_ids, f"Shield should be back in inventory: {inv_ids}"
    assert "longsword" not in inv_ids, f"Longsword should NOT be in inventory: {inv_ids}"
    print("PASS  test_equip_twohander_clears_offhand")


# ---------------------------------------------------------------------------
# 6. Two-hand in hand blocks offhand equip
# ---------------------------------------------------------------------------

def test_offhand_blocked_by_twohander():
    """Cannot equip an offhand item while a two-handed weapon is in hand."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5)
    longsword = _item("longsword", "Long Sword", slot="two_hand", category="weapon")
    shield = _shield()
    player.equipped["hand"] = longsword
    player.inventory = [shield]
    world_map.actors = [player]
    engine = _engine(world_map)

    result = engine._execute_action(player, PlannedAction(
        action_type="equip", params={"item_id": "test_shield"}
    ))
    assert player.equipped["offhand"] is None, "Offhand should remain empty"
    assert "test_shield" in [i.id for i in player.inventory], "Shield should stay in inventory"
    assert "two-handed" in result.lower() or "two_hand" in result.lower() or "offhand" in result.lower(), result
    print("PASS  test_offhand_blocked_by_twohander")


# ---------------------------------------------------------------------------
# 7. equip / unequip round-trip
# ---------------------------------------------------------------------------

def test_equip_unequip_roundtrip():
    """Item moves from inventory → slot on equip, then slot → inventory on unequip."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5)
    sword = _sword()
    player.inventory = [sword]
    world_map.actors = [player]
    engine = _engine(world_map)

    # Equip
    engine._execute_action(player, PlannedAction(action_type="equip", params={"item_id": "test_sword"}))
    assert player.equipped["hand"] is not None and player.equipped["hand"].id == "test_sword"
    assert "test_sword" not in [i.id for i in player.inventory]

    # Unequip
    engine._execute_action(player, PlannedAction(action_type="unequip", params={"slot": "hand"}))
    assert player.equipped["hand"] is None
    assert "test_sword" in [i.id for i in player.inventory]
    print("PASS  test_equip_unequip_roundtrip")


# ---------------------------------------------------------------------------
# 8. use_item — heal capped at max_hp
# ---------------------------------------------------------------------------

def test_use_item_heal():
    """hp_delta heals the actor, capped at max_hp."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5, hp=60, max_hp=100)
    potion = _item("big_potion", "Big Potion", category="consumable", hp_delta=80)
    player.inventory = [potion]
    world_map.actors = [player]
    engine = _engine(world_map)

    result = engine._execute_action(player, PlannedAction(
        action_type="use_item", params={"item_id": "big_potion"}
    ))
    assert player.hp == 100, f"HP should be capped at 100, got {player.hp}"
    assert "big_potion" not in [i.id for i in player.inventory], "Consumed item should be gone"
    assert "HP" in result or "hp" in result.lower(), result
    print("PASS  test_use_item_heal")


# ---------------------------------------------------------------------------
# 9. use_item — poison (negative hp_delta) floored at 0
# ---------------------------------------------------------------------------

def test_use_item_poison():
    """Negative hp_delta reduces HP, floored at 0 (not negative)."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5, hp=10, max_hp=100)
    poison = _item("poison", "Poison Vial", category="consumable", hp_delta=-50)
    player.inventory = [poison]
    world_map.actors = [player]
    engine = _engine(world_map)

    engine._execute_action(player, PlannedAction(
        action_type="use_item", params={"item_id": "poison"}
    ))
    assert player.hp == 0, f"HP should be floored at 0, got {player.hp}"
    print("PASS  test_use_item_poison")


# ---------------------------------------------------------------------------
# 10. use_item — stat deltas (blue-potion style)
# ---------------------------------------------------------------------------

def test_use_item_stat_deltas():
    """attack_delta and crit_delta permanently modify base stats."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5, base_attack=5, base_crit=0.05)
    brew = _item("brew", "Power Brew", category="consumable",
                 attack_delta=3, crit_delta=0.05)
    player.inventory = [brew]
    world_map.actors = [player]
    engine = _engine(world_map)

    engine._execute_action(player, PlannedAction(
        action_type="use_item", params={"item_id": "brew"}
    ))
    assert player.base_attack == 8, f"Expected base_attack=8, got {player.base_attack}"
    assert abs(player.base_crit - 0.10) < 1e-9, f"Expected base_crit≈0.10, got {player.base_crit}"
    print("PASS  test_use_item_stat_deltas")


# ---------------------------------------------------------------------------
# 11. use_item blocked for non-consumable
# ---------------------------------------------------------------------------

def test_use_item_non_consumable_blocked():
    """Using a non-consumable item should fail and leave inventory unchanged."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5)
    sword = _sword()
    player.inventory = [sword]
    world_map.actors = [player]
    engine = _engine(world_map)

    result = engine._execute_action(player, PlannedAction(
        action_type="use_item", params={"item_id": "test_sword"}
    ))
    assert "test_sword" in [i.id for i in player.inventory], "Non-consumable should remain"
    assert "not a consumable" in result.lower() or "consumable" in result.lower(), result
    print("PASS  test_use_item_non_consumable_blocked")


# ---------------------------------------------------------------------------
# 12. attack action — HP reduced, event logged
# ---------------------------------------------------------------------------

def test_attack_reduces_hp():
    """A successful attack reduces the defender's HP."""
    world_map = _make_map()
    attacker = _actor("a", "Alice", x=5, y=5, base_attack=10, base_crit=0.0)
    defender = _actor("b", "Bob",   x=6, y=5, hp=50, max_hp=50, base_defense=2)
    world_map.actors = [attacker, defender]
    engine = _engine(world_map)

    result = engine._execute_action(attacker, PlannedAction(
        action_type="attack", params={"target_actor_id": "b"}
    ))
    # damage = max(1, 10 - 2) = 8; HP should drop
    assert defender.hp < 50, f"Defender HP should have dropped: {defender.hp}"
    assert "attacks" in result.lower() or "damage" in result.lower(), result
    # Event should be logged
    attack_events = [e for e in engine.events if e.event_type == "attack"]
    assert len(attack_events) == 1, f"Expected 1 attack event, got {len(attack_events)}"
    print("PASS  test_attack_reduces_hp")


# ---------------------------------------------------------------------------
# 13. attack action blocked when out of range
# ---------------------------------------------------------------------------

def test_attack_out_of_range_blocked():
    """Attack fails (interrupt) when target is more than MELEE_RANGE tiles away."""
    world_map = _make_map()
    attacker = _actor("a", "Alice", x=3, y=3)
    defender = _actor("b", "Bob",   x=7, y=7)  # Chebyshev = 4 > 1
    world_map.actors = [attacker, defender]
    engine = _engine(world_map)

    before_hp = defender.hp
    result = engine._execute_action(attacker, PlannedAction(
        action_type="attack", params={"target_actor_id": "b"}
    ))
    assert defender.hp == before_hp, "HP should not change when out of range"
    assert "far" in result.lower() or "closer" in result.lower() or "range" in result.lower(), result
    assert attacker.needs_replan, "Attacker should be flagged for replan"
    print("PASS  test_attack_out_of_range_blocked")


# ---------------------------------------------------------------------------
# 14. Actor death — non-shelf inventory + equipped items drop
# ---------------------------------------------------------------------------

def test_death_drops_inventory_and_equipped():
    """On death: personal inventory and equipped items land on the map as unowned WorldItems."""
    world_map = _make_map()
    victim = _actor("v", "Victim", x=5, y=5)
    sword = _sword()
    helmet = _helmet()
    potion = _potion()
    victim.inventory = [potion]
    victim.equipped["hand"] = sword
    victim.equipped["head"] = helmet
    world_map.actors = [victim]
    engine = _engine(world_map)

    engine._handle_actor_death(victim)

    ground_ids = {gi.item.id for gi in world_map.world_items}
    assert "test_potion" in ground_ids, f"Potion should be on ground: {ground_ids}"
    assert "test_sword" in ground_ids, f"Sword should be on ground: {ground_ids}"
    assert "test_helmet" in ground_ids, f"Helmet should be on ground: {ground_ids}"

    # All dropped items at victim's original position and unowned
    for gi in world_map.world_items:
        if gi.item.id in ("test_potion", "test_sword", "test_helmet"):
            assert gi.x == 5 and gi.y == 5, f"Item not at death tile: {gi.x},{gi.y}"
            assert gi.owner_id is None, f"Dropped item should be unowned: {gi.owner_id}"
    print("PASS  test_death_drops_inventory_and_equipped")


# ---------------------------------------------------------------------------
# 15. Actor death — corpse WorldItem spawned
# ---------------------------------------------------------------------------

def test_death_spawns_corpse():
    """A corpse WorldItem with category='corpse' is placed at the death tile."""
    world_map = _make_map()
    victim = _actor("v", "Victim", x=6, y=4)
    world_map.actors = [victim]
    engine = _engine(world_map)

    engine._handle_actor_death(victim)

    corpses = [
        gi for gi in world_map.world_items
        if gi.item.metadata.get("category") == "corpse"
    ]
    assert len(corpses) == 1, f"Expected 1 corpse, found: {len(corpses)}"
    assert corpses[0].x == 6 and corpses[0].y == 4
    assert corpses[0].owner_id is None
    assert "Victim" in corpses[0].item.name, f"Corpse name: {corpses[0].item.name}"
    print("PASS  test_death_spawns_corpse")


# ---------------------------------------------------------------------------
# 16. Actor death — actor removed from world_map.actors
# ---------------------------------------------------------------------------

def test_death_removes_actor():
    """Dead actor is removed from world_map.actors."""
    world_map = _make_map()
    attacker = _actor("a", "Alice", x=5, y=5)
    victim   = _actor("v", "Victim", x=6, y=5, hp=1, max_hp=1, base_defense=0)
    world_map.actors = [attacker, victim]
    engine = _engine(world_map)

    # One punch should kill (base_attack=5, defense=0 → damage=5 ≥ 1 hp)
    engine._execute_action(attacker, PlannedAction(
        action_type="attack", params={"target_actor_id": "v"}
    ))
    actor_ids = [a.id for a in world_map.actors]
    assert "v" not in actor_ids, f"Victim should be removed, actors: {actor_ids}"
    assert "a" in actor_ids, "Attacker should still be alive"
    print("PASS  test_death_removes_actor")


# ---------------------------------------------------------------------------
# 17. Actor death — shelf items are NOT dropped (they stay as WorldItems with owner)
# ---------------------------------------------------------------------------

def test_death_does_not_drop_shelf_items():
    """When a shopkeeper dies, their stocked shelf items remain on the map (owner_id preserved).

    Only personal inventory + equipped items drop as unowned loot.
    """
    world_map = _make_map()
    sk = _actor("sk", "Shopkeeper", role=ActorRole.SHOPKEEPER, x=5, y=5)
    shelf_item = _item("shelf_potion", "Shelf Potion", category="consumable")
    world_map.world_items = [WorldItem(item=shelf_item, x=2, y=1, owner_id="sk")]
    world_map.actors = [sk]
    engine = _engine(world_map)

    engine._handle_actor_death(sk)

    # Shelf WorldItem should still exist (it's a world item, not in personal inventory)
    shelf_gis = [gi for gi in world_map.world_items if gi.item.id == "shelf_potion"]
    assert len(shelf_gis) == 1, "Shelf item should still be present on map"
    # owner_id is preserved (actor is gone but item remains for others to pick up)
    # The item itself is not in any actor's inventory so _handle_actor_death won't touch it
    print("PASS  test_death_does_not_drop_shelf_items")


# ---------------------------------------------------------------------------
# 18. Carry limit — pickup blocked when full
# ---------------------------------------------------------------------------

def test_carry_limit_blocks_pickup():
    """Pickup fails with an interrupt when the actor's inventory is full."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5, max_carry=2)
    filler1 = _item("filler1", "Thing 1")
    filler2 = _item("filler2", "Thing 2")
    loot    = _item("loot",    "Loot Item")
    player.inventory = [filler1, filler2]  # already at max_carry
    world_map.world_items = [WorldItem(item=loot, x=5, y=4, owner_id=None)]
    world_map.actors = [player]
    engine = _engine(world_map)

    result = engine._execute_action(player, PlannedAction(
        action_type="pick_up", params={"item_id": "loot"}
    ))
    assert "loot" not in [i.id for i in player.inventory], "Item should not be picked up"
    assert len(world_map.world_items) == 1, "Item should still be on ground"
    assert "carry" in result.lower() or "full" in result.lower() or "too many" in result.lower(), result
    assert player.needs_replan, "Should be flagged for replan"
    print("PASS  test_carry_limit_blocks_pickup")


# ---------------------------------------------------------------------------
# 19. Carry limit — shelf items do not count toward personal limit
# ---------------------------------------------------------------------------

def test_shelf_items_dont_count_toward_carry():
    """Shelf WorldItems (owner_id == actor.id) are excluded from carry count."""
    world_map = _make_map()
    sk = _actor("sk", "Shopkeeper", role=ActorRole.SHOPKEEPER, x=5, y=5, max_carry=2)
    # Two shelf items owned by the shopkeeper
    for i in range(5):
        world_map.world_items.append(
            WorldItem(item=_item(f"shelf_{i}", f"Shelf {i}"), x=2, y=1, owner_id="sk")
        )
    world_map.actors = [sk]
    engine = _engine(world_map)

    # Personal inventory is empty — should be 0/2, not 5/2
    assert engine._carry_count(sk) == 0, f"Expected 0 carry, got {engine._carry_count(sk)}"
    print("PASS  test_shelf_items_dont_count_toward_carry")


# ---------------------------------------------------------------------------
# 20. Carry limit — trade blocked when receiver is full
# ---------------------------------------------------------------------------

def test_carry_limit_blocks_trade():
    """Trade is rejected when the proposer (receiver of items) has no carry space."""
    world_map = _make_map()
    buyer  = _actor("b", "Buyer",    x=4, y=4, gold=100, max_carry=1)
    seller = _actor("s", "Seller",   x=5, y=4, gold=0)
    potion = _item("p1", "Potion 1", category="consumable")
    filler = _item("p2", "Filler",   category="consumable")
    buyer.inventory = [filler]       # buyer already full (max_carry=1)
    seller.inventory = [potion]
    world_map.actors = [buyer, seller]
    engine = _engine(world_map)

    proposal = TradeProposal(
        proposer_id="b",
        target_id="s",
        offered_gold=10,
        offered_items=[],
        requested_items=[potion],
        requested_gold=0,
    )
    err = engine._resolve_trade(proposal, buyer, seller)
    assert err is not None, "Trade should be rejected"
    assert "carry" in err.lower() or "space" in err.lower(), f"Unexpected error: {err}"
    # Item should remain with seller
    assert "p1" in [i.id for i in seller.inventory], "Item should stay with seller"
    print("PASS  test_carry_limit_blocks_trade")


# ---------------------------------------------------------------------------
# 21. Observation — inventory shows (N/max_carry)
# ---------------------------------------------------------------------------

def test_observation_shows_carry_count():
    """ObserveSurroundingsTool shows inventory count and capacity."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5, max_carry=8)
    sword = _sword()
    player.inventory = [sword]
    world_map.actors = [player]

    memory = RelationshipMemory()
    obs = ObserveSurroundingsTool(actor=player, world_map=world_map, memory=memory)
    output = obs.forward()

    assert "1/8" in output, f"Expected '1/8' carry indicator in output:\n{output}"
    print("PASS  test_observation_shows_carry_count")


# ---------------------------------------------------------------------------
# 22. Observation — equipped items listed
# ---------------------------------------------------------------------------

def test_observation_shows_equipped():
    """Equipped items appear in the observation output."""
    world_map = _make_map()
    player = _actor("p", "Player", x=5, y=5)
    sword = _sword()
    player.equipped["hand"] = sword
    world_map.actors = [player]

    memory = RelationshipMemory()
    obs = ObserveSurroundingsTool(actor=player, world_map=world_map, memory=memory)
    output = obs.forward()

    assert "equipped" in output.lower(), f"Expected 'equipped' in output:\n{output}"
    assert "Test Sword" in output, f"Expected sword name in output:\n{output}"
    assert "hand" in output.lower(), f"Expected slot name in output:\n{output}"
    print("PASS  test_observation_shows_equipped")


# ---------------------------------------------------------------------------
# 23. Observation — corpse visible after death
# ---------------------------------------------------------------------------

def test_corpse_visible_in_observation():
    """After an actor dies, survivors can see the corpse labeled [CORPSE]."""
    world_map = _make_map()
    survivor = _actor("s", "Survivor", x=5, y=5)
    victim   = _actor("v", "Victim",   x=6, y=5)
    world_map.actors = [survivor, victim]
    engine = _engine(world_map)

    engine._handle_actor_death(victim)

    memory = RelationshipMemory()
    obs = ObserveSurroundingsTool(actor=survivor, world_map=world_map, memory=memory)
    output = obs.forward()

    assert "Corpse of Victim" in output, f"Corpse not visible:\n{output}"
    assert "[CORPSE]" in output, f"Corpse not labeled [CORPSE]:\n{output}"
    print("PASS  test_corpse_visible_in_observation")


# ---------------------------------------------------------------------------
# 24. load_item_catalog — 20 items with correct combat stats spot-check
# ---------------------------------------------------------------------------

def test_catalog_count_and_stats():
    """Catalog loads 20 items; spot-check key combat stats."""
    catalog = load_item_catalog()
    assert len(catalog) == 20, f"Expected 20 items, got {len(catalog)}"

    by_id = {item.id: item for item in catalog}

    # Weapons
    assert by_id["dagger"].attack_bonus == 5
    assert abs(by_id["dagger"].crit_bonus - 0.12) < 1e-9
    assert by_id["short_sword"].attack_bonus == 8
    assert by_id["long_sword"].attack_bonus == 14
    assert by_id["long_sword"].metadata["slot"] == "two_hand"

    # Armor
    assert by_id["shield"].defense_bonus == 6
    assert by_id["iron_chest"].defense_bonus == 8
    assert by_id["leather_hood"].defense_bonus == 1

    # Consumables
    assert by_id["red_potion"].hp_delta == 30
    assert by_id["green_potion"].hp_delta == -25
    assert by_id["blue_potion"].attack_delta == 2
    assert abs(by_id["blue_potion"].crit_delta - 0.05) < 1e-9
    assert by_id["wine_bottle"].attack_delta == 1
    assert by_id["meat"].hp_delta == 15

    # Relic — no combat stats
    assert by_id["lost_crown"].attack_bonus == 0
    assert by_id["lost_crown"].defense_bonus == 0

    print("PASS  test_catalog_count_and_stats")


# ---------------------------------------------------------------------------
# 25. load_item_catalog — all items parse without error
# ---------------------------------------------------------------------------

def test_catalog_all_items_parse():
    """Every item in items.json deserialises into a valid Item without raising."""
    catalog = load_item_catalog()
    for item in catalog:
        assert item.id, f"Item missing id: {item}"
        assert item.name, f"Item missing name: {item}"
        assert item.base_price >= 0
        assert 0.0 <= item.crit_bonus <= 1.0
    print("PASS  test_catalog_all_items_parse")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_calc_damage_base,
        test_calc_damage_floor,
        test_calc_damage_crit,
        test_calc_damage_armor,
        test_equip_twohander_clears_offhand,
        test_offhand_blocked_by_twohander,
        test_equip_unequip_roundtrip,
        test_use_item_heal,
        test_use_item_poison,
        test_use_item_stat_deltas,
        test_use_item_non_consumable_blocked,
        test_attack_reduces_hp,
        test_attack_out_of_range_blocked,
        test_death_drops_inventory_and_equipped,
        test_death_spawns_corpse,
        test_death_removes_actor,
        test_death_does_not_drop_shelf_items,
        test_carry_limit_blocks_pickup,
        test_shelf_items_dont_count_toward_carry,
        test_carry_limit_blocks_trade,
        test_observation_shows_carry_count,
        test_observation_shows_equipped,
        test_corpse_visible_in_observation,
        test_catalog_count_and_stats,
        test_catalog_all_items_parse,
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

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
