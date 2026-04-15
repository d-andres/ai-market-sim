"""Adaptive fallback tests for planning.

Ensures fallback behavior stays AI-driven:
- model authors a fallback policy
- model uses that policy to produce a recovery plan
- policy refreshes when context changes (interrupt)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.brain import AgentBrain
from src.models.schema import Actor, ActorRole, Item, Map, Tile, TileType, WorldItem
from src.simulation.engine import SimulationEngine


class _ScriptedModel:
    """Fake model that returns scripted responses in order."""

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


def _make_map(width: int = 12, height: int = 12) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                tt = TileType.WALL
            elif y == 1 and x in (1, 2, 3):
                tt = TileType.SHOP
            elif (x, y) == (1, 10):
                tt = TileType.ENTRANCE
            else:
                tt = TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def test_adaptive_fallback_is_ai_authored_and_refreshes() -> None:
    world_map = _make_map()

    shopkeeper = Actor(
        id="shopkeeper_1",
        name="Merchant Elara",
        role=ActorRole.SHOPKEEPER,
        x=5,
        y=4,
        gold=100,
        personality_traits=["shrewd", "protective"],
    )
    player = Actor(
        id="player_1",
        name="Adventurer",
        role=ActorRole.PLAYER,
        x=6,
        y=4,
        gold=60,
        personality_traits=["ambitious", "cunning"],
    )
    world_map.actors = [shopkeeper, player]

    gem = Item(id="royal_gem", name="Royal Gem", description="Very valuable.", base_price=80, quantity=1)
    world_map.world_items = [WorldItem(item=gem, x=6, y=5, owner_id=shopkeeper.id)]

    engine = SimulationEngine(world_map, enable_ai=False)

    player_brain = AgentBrain(actor=player, world_map=world_map, engine=engine)

    scripted_outputs = [
        # Primary plan fails to parse -> triggers adaptive fallback path.
        "this is not valid json",
        # Fallback policy authored by AI.
        '{"identity_summary":"Aggressive opportunist with social flexibility.","goals":["Acquire rare goods quickly","Exploit openings near merchants"],"preferred_tactics":["probe with dialogue","attempt trade before theft"],"risk_posture":"medium","revision_triggers":["injured","guard pressure"]}',
        # Fallback plan authored by AI using policy above.
        '{"summary":"Adventurer pivots to an opportunistic acquisition plan built from current leverage.","plan":[{"action_type":"converse","params":{"target_actor_id":"shopkeeper_1","opening_line":"Show me your best item first."},"reason":"probe merchant posture"},{"action_type":"propose_trade","params":{"target_actor_id":"shopkeeper_1","offered_gold":60,"offered_item_ids":"","requested_item_ids":"royal_gem","requested_gold":0},"reason":"attempt purchase"},{"action_type":"wait","params":{},"reason":"observe response"}]}',
        # Second primary plan fails again.
        "still invalid",
        # Interrupt should refresh policy with a different posture.
        '{"identity_summary":"Now under pressure, favor decisive actions.","goals":["Secure target item immediately"],"preferred_tactics":["rapid pressure","high-risk move"],"risk_posture":"high","revision_triggers":["target moved","critical injury"]}',
        # Second fallback plan.
        '{"summary":"Adventurer switches to a high-risk pressure approach after disruption.","plan":[{"action_type":"converse","params":{"target_actor_id":"shopkeeper_1","opening_line":"Last chance to sell fairly."},"reason":"final warning"},{"action_type":"pick_up","params":{"item_id":"royal_gem"},"reason":"force acquisition if exposed"},{"action_type":"wait","params":{},"reason":"reassess witnesses"}]}',
    ]
    player_brain.model = _ScriptedModel(scripted_outputs)

    first_plan, first_summary = player_brain.create_plan()
    second_plan, second_summary = player_brain.create_plan(interrupt_reason="You were threatened by a guard")

    assert first_plan, "Adaptive fallback plan should not be empty"
    assert second_plan, "Adaptive fallback plan after interrupt should not be empty"

    assert any(a.action_type != "wait" for a in first_plan), first_plan
    assert any(a.action_type != "wait" for a in second_plan), second_plan
    assert "adaptive fallback" in first_summary.lower(), first_summary
    assert "adaptive fallback" in second_summary.lower(), second_summary

    policy = player_brain._adaptive_fallback_policy
    assert policy is not None, "Expected adaptive fallback policy to be generated"
    assert policy.get("risk_posture") == "high", f"Expected refreshed policy posture to be high, got: {policy}"
    assert any("Secure target item immediately" in g for g in policy.get("goals", [])), policy

    assert player_brain.model.call_count >= 6, "Expected scripted model calls for policy + fallback plans"

    print("PASS  test_adaptive_fallback_is_ai_authored_and_refreshes")


if __name__ == "__main__":
    try:
        test_adaptive_fallback_is_ai_authored_and_refreshes()
        print("\n=======================================================")
        print("Results: 1 passed, 0 failed out of 1 tests")
    except Exception as exc:
        import traceback

        print(f"FAIL  test_adaptive_fallback_is_ai_authored_and_refreshes: {exc}")
        traceback.print_exc()
        sys.exit(1)
