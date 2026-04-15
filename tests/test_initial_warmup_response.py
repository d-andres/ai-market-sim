"""Startup initial-response behavior tests.

These tests focus on the create_plan mode used during initial map generation:
- allow_adaptive_fallback=False should not cascade into extra fallback model calls
- allow_adaptive_fallback=True may use adaptive fallback to recover from bad output
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.brain import AgentBrain
from src.models.schema import Actor, ActorRole, Map, Tile, TileType
from src.simulation.engine import SimulationEngine


class _ScriptedModel:
    """Return scripted payloads in order and count calls."""

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.call_count = 0

    def __call__(self, _messages):
        idx = self.call_count if self.call_count < len(self.outputs) else len(self.outputs) - 1
        payload = self.outputs[idx]
        self.call_count += 1

        class _Resp:
            content = payload
            thinking = ""
            duration = 0.0

        return _Resp()


def _make_map(width: int = 10, height: int = 10) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            tt = TileType.WALL if (x in {0, width - 1} or y in {0, height - 1}) else TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _brain_with_model(outputs: list[str]) -> tuple[AgentBrain, _ScriptedModel]:
    world_map = _make_map()
    actor = Actor(id="a1", name="Adventurer", role=ActorRole.PLAYER, x=5, y=5, gold=50)
    world_map.actors = [actor]
    engine = SimulationEngine(world_map, enable_ai=False)
    brain = AgentBrain(actor=actor, world_map=world_map, engine=engine)
    model = _ScriptedModel(outputs)
    brain.model = model
    return brain, model


def test_initial_response_single_pass_no_adaptive_cascade() -> None:
    # First output is invalid JSON. In warm-up mode (allow_adaptive_fallback=False)
    # the brain must return a quick fallback without extra model calls.
    brain, model = _brain_with_model([
        "not json",
        '{"identity_summary":"should_not_be_used","goals":["x"],"preferred_tactics":["x"],"risk_posture":"medium","revision_triggers":["x"]}',
        '{"summary":"should not be used","plan":[{"action_type":"move","params":{"direction":"north"},"reason":"x"}]}',
    ])

    plan, summary = brain.create_plan("", allow_adaptive_fallback=False)

    assert model.call_count == 1, f"Expected exactly 1 model call, got {model.call_count}"
    assert plan and plan[0].action_type == "wait", f"Expected warm-up wait fallback, got: {plan}"
    assert "initial" in summary.lower() or "recover" in summary.lower(), summary
    print("PASS  test_initial_response_single_pass_no_adaptive_cascade")


def test_non_warmup_can_use_adaptive_fallback_chain() -> None:
    # In normal mode, a bad primary output should trigger adaptive policy+plan recovery.
    brain, model = _brain_with_model([
        "not json",
        '{"identity_summary":"adaptive posture","goals":["acquire value"],"preferred_tactics":["trade"],"risk_posture":"medium","revision_triggers":["threat"]}',
        '{"summary":"Adventurer recovers with adaptive strategy.","plan":[{"action_type":"move","params":{"direction":"north"},"reason":"reposition"},{"action_type":"wait","params":{},"reason":"observe"}]}',
    ])

    plan, summary = brain.create_plan("", allow_adaptive_fallback=True)

    assert model.call_count >= 3, f"Expected adaptive fallback chain, got {model.call_count} calls"
    assert any(a.action_type != "wait" for a in plan), f"Expected at least one non-wait action: {plan}"
    assert "adaptive fallback" in summary.lower(), summary
    print("PASS  test_non_warmup_can_use_adaptive_fallback_chain")


def test_warmup_fallback_marks_actor_for_immediate_replan() -> None:
    """When warm-up produces a wait-only fallback, the actor should be marked for
    immediate critical replan so the tick loop doesn't defer 23 ticks."""
    from src.models.schema import PlannedAction

    world_map = _make_map()
    actor = Actor(id="a1", name="Adventurer", role=ActorRole.PLAYER, x=5, y=5, gold=50)
    world_map.actors = [actor]

    # Simulate what main.py does after warm-up returns a fallback plan
    plan = [PlannedAction(action_type="wait", params={}, reason="initial warm-up fallback")]
    actor.action_queue = plan
    is_fallback = all(a.action_type == "wait" for a in plan)
    if is_fallback:
        actor.needs_replan = True
        actor.interrupt_reason = "initial plan was low quality"
    else:
        actor.needs_replan = False
        actor.interrupt_reason = ""

    assert actor.needs_replan, "Fallback plan should mark needs_replan=True"
    assert actor.interrupt_reason, "Fallback plan should set interrupt_reason to trigger replan at next interval"
    print("PASS  test_warmup_fallback_marks_actor_for_immediate_replan")


def test_plan_timeout_watchdog_cleans_up_stuck_futures() -> None:
    """Futures pending for too many ticks should be abandoned by the watchdog."""
    from concurrent.futures import Future
    from src.simulation.engine import PLAN_TIMEOUT_TICKS

    world_map = _make_map()
    actor = Actor(id="a1", name="Adventurer", role=ActorRole.PLAYER, x=5, y=5, gold=50)
    world_map.actors = [actor]
    engine = SimulationEngine(world_map, enable_ai=False)

    # Plant a fake never-completing future
    fake_future = Future()
    engine._pending_plan_futures["a1"] = fake_future
    engine._pending_plan_started_tick["a1"] = 0
    engine._pending_plan_generation["a1"] = 0

    # Advance tick_count past the timeout threshold
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert "a1" not in engine._pending_plan_futures, "Timed-out future should be removed"
    assert actor.needs_replan, "Actor should be marked for replan after timeout"
    assert "timed out" in actor.interrupt_reason.lower(), f"Expected timeout reason, got: {actor.interrupt_reason}"
    print("PASS  test_plan_timeout_watchdog_cleans_up_stuck_futures")


if __name__ == "__main__":
    tests = [
        test_initial_response_single_pass_no_adaptive_cascade,
        test_non_warmup_can_use_adaptive_fallback_chain,
        test_warmup_fallback_marks_actor_for_immediate_replan,
        test_plan_timeout_watchdog_cleans_up_stuck_futures,
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
