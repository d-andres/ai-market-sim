"""Dialogue-driven social state tests.

Validates that free-form conversation text can:
- Increase/decrease likeness
- Increase suspicion toward named suspects
- Trigger guard responses when accusations are strong enough
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.brain import ObserveSurroundingsTool, RelationshipMemory
from src.models.schema import Actor, ActorRole, Map, PlannedAction, Tile, TileType
from src.simulation.engine import SimulationEngine


def _make_map(width: int = 16, height: int = 16) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                tt = TileType.WALL
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _actor(aid: str, name: str, role: ActorRole, x: int, y: int) -> Actor:
    return Actor(id=aid, name=name, role=role, x=x, y=y, gold=20)


def _engine(world_map: Map) -> SimulationEngine:
    return SimulationEngine(world_map, enable_ai=False)


def test_polite_dialogue_increases_likeness() -> None:
    world_map = _make_map()
    a = _actor("a", "Alice", ActorRole.PLAYER, 5, 5)
    b = _actor("b", "Borin", ActorRole.PLAYER, 6, 5)
    world_map.actors = [a, b]
    engine = _engine(world_map)

    engine._execute_action(a, PlannedAction(
        action_type="converse",
        params={"target_actor_id": "b", "opening_line": "Please help me, friend. Thanks!"},
    ))

    assert b.likeness.get("a", 0) > 0, f"Expected positive likeness, got: {b.likeness}"
    print("PASS  test_polite_dialogue_increases_likeness")


def test_hostile_dialogue_decreases_likeness() -> None:
    world_map = _make_map()
    a = _actor("a", "Alice", ActorRole.PLAYER, 5, 5)
    b = _actor("b", "Borin", ActorRole.PLAYER, 6, 5)
    world_map.actors = [a, b]
    engine = _engine(world_map)

    engine._execute_action(a, PlannedAction(
        action_type="converse",
        params={"target_actor_id": "b", "opening_line": "You liar, idiot. I hate you."},
    ))

    assert b.likeness.get("a", 0) < 0, f"Expected negative likeness, got: {b.likeness}"
    print("PASS  test_hostile_dialogue_decreases_likeness")


def test_accusation_increases_suspicion() -> None:
    world_map = _make_map()
    accuser = _actor("acc", "Accuser", ActorRole.PLAYER, 5, 5)
    listener = _actor("lst", "Listener", ActorRole.PLAYER, 6, 5)
    suspect = _actor("sus", "Suspect", ActorRole.PLAYER, 7, 5)
    listener.likeness[accuser.id] = 45
    world_map.actors = [accuser, listener, suspect]
    engine = _engine(world_map)

    engine._execute_action(accuser, PlannedAction(
        action_type="converse",
        params={"target_actor_id": "lst", "opening_line": "sus stole my coin. sus is a thief and criminal."},
    ))

    score = engine.suspicion_score("lst", "sus")
    assert score > 0, f"Expected suspicion > 0, got {score}"
    dialogue_events = [e for e in engine.events if e.event_type == "dialogue_suspicion"]
    assert dialogue_events, "Expected dialogue_suspicion event"
    print("PASS  test_accusation_increases_suspicion")


def test_guard_can_react_to_dialogue_accusation() -> None:
    world_map = _make_map()
    accuser = _actor("acc", "Accuser", ActorRole.PLAYER, 5, 5)
    guard = _actor("grd", "Guard Thorne", ActorRole.GUARD, 6, 5)
    suspect = _actor("sus", "Suspect", ActorRole.PLAYER, 7, 5)
    # Trusted source: guard is more willing to act on accusation.
    guard.likeness[accuser.id] = 70
    world_map.actors = [accuser, guard, suspect]
    engine = _engine(world_map)

    for _ in range(4):
        engine._execute_action(accuser, PlannedAction(
            action_type="converse",
            params={
                "target_actor_id": "grd",
                "opening_line": "sus stole, attacked, murdered and is a criminal thief.",
            },
        ))

    guard_events = [e for e in engine.events if e.event_type.startswith("guard_")]
    assert guard_events, "Expected guard reaction event after repeated accusation"
    print("PASS  test_guard_can_react_to_dialogue_accusation")


def test_guard_ignores_low_trust_accusation() -> None:
    world_map = _make_map()
    accuser = _actor("acc", "Accuser", ActorRole.PLAYER, 5, 5)
    guard = _actor("grd", "Guard Thorne", ActorRole.GUARD, 6, 5)
    suspect = _actor("sus", "Suspect", ActorRole.PLAYER, 7, 5)
    # Untrusted source: accusation should not immediately escalate.
    guard.likeness[accuser.id] = -30
    world_map.actors = [accuser, guard, suspect]
    engine = _engine(world_map)

    engine._execute_action(accuser, PlannedAction(
        action_type="converse",
        params={
            "target_actor_id": "grd",
            "opening_line": "sus is a criminal thief and murderer.",
        },
    ))

    guard_events = [e for e in engine.events if e.event_type.startswith("guard_")]
    assert not guard_events, f"Low-trust accusation should not trigger guard punishment: {guard_events}"
    print("PASS  test_guard_ignores_low_trust_accusation")


def test_observation_exposes_suspicion_score() -> None:
    world_map = _make_map()
    accuser = _actor("acc", "Accuser", ActorRole.PLAYER, 5, 5)
    listener = _actor("lst", "Listener", ActorRole.PLAYER, 6, 5)
    suspect = _actor("sus", "Suspect", ActorRole.PLAYER, 7, 5)
    world_map.actors = [accuser, listener, suspect]
    engine = _engine(world_map)

    engine._execute_action(accuser, PlannedAction(
        action_type="converse",
        params={"target_actor_id": "lst", "opening_line": "sus stole from me."},
    ))

    obs = ObserveSurroundingsTool(actor=listener, world_map=world_map, memory=RelationshipMemory(), engine=engine)
    output = obs.forward(vision_range=10)

    assert "your_suspicion" in output, f"Expected suspicion field in observation output:\n{output}"
    print("PASS  test_observation_exposes_suspicion_score")


if __name__ == "__main__":
    tests = [
        test_polite_dialogue_increases_likeness,
        test_hostile_dialogue_decreases_likeness,
        test_accusation_increases_suspicion,
        test_guard_can_react_to_dialogue_accusation,
        test_guard_ignores_low_trust_accusation,
        test_observation_exposes_suspicion_score,
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
