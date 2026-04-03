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


if __name__ == "__main__":
    tests = [
        test_initial_response_single_pass_no_adaptive_cascade,
        test_non_warmup_can_use_adaptive_fallback_chain,
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
