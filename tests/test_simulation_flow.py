"""End-to-end simulation flow tests.

Section 1 — Mocked flow tests (always run):
  Exercises the full lifecycle: initialisation → plan submission → action
  execution → timeout → health check → cooldown → normal replanning,
  all without touching a real LLM.

Section 2 — Live LLM integration tests (run on request only):
  Requires a running Ollama instance.  Skipped by default unless the
  environment variable ``RUN_LIVE_LLM_TESTS=1`` is set:

      RUN_LIVE_LLM_TESTS=1 pytest tests/test_simulation_flow.py -v -k live
"""

import os
import sys
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import Actor, ActorRole, Map, PlannedAction, Tile, TileType
from src.simulation.engine import (
    MAX_CONCURRENT_PLAN_JOBS,
    PLAN_FAILURE_COOLDOWN_TICKS,
    PLAN_TIMEOUT_TICKS,
    REPLAN_CADENCE_TICKS,
    TIMEOUT_REPLAN_COOLDOWN_TICKS,
    TIMEOUT_UNHEALTHY_COOLDOWN_TICKS,
    SimulationEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_map(width: int = 10, height: int = 10) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            tt = TileType.WALL if (x in {0, width - 1} or y in {0, height - 1}) else TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _make_engine_with_actors(n: int = 1) -> tuple[SimulationEngine, list[Actor]]:
    """Build an engine with *n* actors placed on floor tiles, AI disabled."""
    world_map = _make_map()
    actors = []
    for i in range(n):
        a = Actor(
            id=f"a{i + 1}",
            name=f"Actor{i + 1}",
            role=ActorRole.PLAYER,
            x=2 + i,
            y=2,
            gold=50,
        )
        actors.append(a)
    world_map.actors = actors
    engine = SimulationEngine(world_map, enable_ai=False)
    return engine, actors


def _install_mock_brain(engine: SimulationEngine, actor_id: str, plan_result=None) -> MagicMock:
    """Install a MagicMock brain for the given actor.

    *plan_result* is the value ``brain.create_plan()`` will return.
    Defaults to a single wait action with a summary.
    """
    if plan_result is None:
        plan_result = (
            [PlannedAction(action_type="wait", params={}, reason="default mock")],
            "Actor waits.",
        )
    brain = MagicMock()
    brain.create_plan = MagicMock(return_value=plan_result)
    brain.health_check = MagicMock(return_value=(True, "OK"))
    brain.last_request_prompt = "(mock plan prompt)"
    brain.last_response_raw = "(mock raw response)"
    engine.agent_brains[actor_id] = brain
    return brain


# ═══════════════════════════════════════════════════════════════════════════
# Section 1 — Mocked end-to-end flow tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """Walk through the entire planning → execution → timeout → recovery cycle."""

    def test_init_to_plan_to_action_execution(self):
        """Fresh engine → submit plan → plan completes → actions execute."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [
                PlannedAction(action_type="move", params={"direction": "east"}, reason="explore"),
                PlannedAction(action_type="wait", params={}, reason="look around"),
            ],
            "Actor1 heads east then waits.",
        ))

        # Enable AI so tick() uses the planning path.
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "initial warm-up"

        # Tick 0: should submit a plan request.
        updates = engine.tick()
        assert engine.llm_call_count == 1, "First tick should submit a plan"
        assert "a1" in engine._pending_plan_futures

        # Simulate the future completing instantly.
        future = engine._pending_plan_futures["a1"]
        future.result()  # MagicMock executor — the future is real so grab result

        # Actually, the ThreadPoolExecutor runs the mock synchronously in a thread,
        # so by the time we reach tick 1 the future should already be done.
        # Tick 1: collect completed plan and start executing actions.
        updates = engine.tick()
        # Plan should have been applied.
        assert engine.applied_plan_count == 1
        assert actor.needs_replan is False
        # The first action (move east) should have executed this tick.
        # Actor started at (2, 2); east = (+1, 0) → (3, 2).
        assert actor.x == 3 and actor.y == 2, f"Expected (3,2), got ({actor.x},{actor.y})"

        # Tick 2: second action (wait) should execute.
        updates = engine.tick()
        # Actor stays at (3, 2).
        assert actor.x == 3 and actor.y == 2
        # Queue should be empty now; actor needs a new plan eventually.

    def test_action_queue_drains_then_replans(self):
        """After the queue drains the actor becomes a candidate for replanning."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [PlannedAction(action_type="wait", params={}, reason="one-step plan")],
            "Quick plan.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"

        # Tick 0: submit plan.
        engine.tick()
        assert engine.llm_call_count == 1

        # Tick 1: plan arrives, action executes — queue now empty.
        engine.tick()
        assert engine.applied_plan_count == 1
        assert len(actor.action_queue) == 0

        # Fast-forward to a replan-interval boundary so the actor is eligible again.
        engine.tick_count = engine.replan_interval
        brain.create_plan.return_value = (
            [PlannedAction(action_type="wait", params={}, reason="second plan")],
            "Second plan.",
        )
        engine.tick()
        assert engine.llm_call_count == 2, "Should submit a second plan on replan interval tick"

    def test_timeout_triggers_health_check_then_recovery(self):
        """Plan timeout → health check submitted → passes → short cooldown → replan."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1")
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"

        # Tick 0: submit plan.
        engine.tick()
        assert engine.llm_call_count == 1

        # Sabotage the future so it never completes (simulate a hung LLM).
        stuck_future = Future()
        engine._pending_plan_futures["a1"] = stuck_future
        engine._pending_plan_started_tick["a1"] = engine.tick_count

        # Fast-forward past the timeout horizon.
        engine.tick_count = PLAN_TIMEOUT_TICKS + 2
        updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
        engine._collect_completed_plan_requests(updates)

        assert engine.llm_timeout_count == 1
        assert "a1" not in engine._pending_plan_futures, "Timed-out future should be removed"
        assert "a1" in engine._pending_health_checks, "Health check should be submitted"

        # Simulate the health check future completing successfully.
        hc_future = Future()
        hc_future.set_result((True, "OK"))
        engine._pending_health_checks["a1"] = hc_future

        engine._collect_health_check_results()

        assert "a1" not in engine._pending_health_checks
        assert actor.needs_replan is True
        cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
        assert cooldown_until == engine.tick_count + TIMEOUT_REPLAN_COOLDOWN_TICKS

        # Advance to the next replan interval boundary at or after the cooldown.
        engine.tick_count = ((cooldown_until) // engine.replan_interval + 1) * engine.replan_interval
        brain.create_plan.return_value = (
            [PlannedAction(action_type="move", params={"direction": "south"}, reason="recovered")],
            "Actor1 resumes activity.",
        )
        engine.tick()
        assert engine.llm_call_count == 2, "Should replan at replan interval after cooldown"

    def test_timeout_unhealthy_extends_cooldown(self):
        """Timeout → health check FAILS → longer cooldown before replan."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1")
        engine.enable_ai = True

        # Plant a stuck future directly.
        stuck_future = Future()
        engine._pending_plan_futures["a1"] = stuck_future
        engine._pending_plan_started_tick["a1"] = 0
        engine._pending_plan_started_time["a1"] = time.monotonic() - 300
        engine._pending_plan_generation["a1"] = 0
        engine.tick_count = PLAN_TIMEOUT_TICKS + 1

        updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
        engine._collect_completed_plan_requests(updates)

        # Simulate a FAILED health check.
        hc_future = Future()
        hc_future.set_result((False, "Connection refused"))
        engine._pending_health_checks["a1"] = hc_future

        engine._collect_health_check_results()

        cooldown_until = engine._plan_failure_cooldown_until["a1"]
        expected = engine.tick_count + TIMEOUT_UNHEALTHY_COOLDOWN_TICKS
        assert cooldown_until == expected, f"Expected cooldown until {expected}, got {cooldown_until}"

        # Verify actor is NOT a candidate until cooldown expires.
        engine.tick_count = cooldown_until - 1
        candidates = [
            a for a in engine.world_map.actors
            if (a.needs_replan or not a.action_queue)
            and a.id not in engine._pending_plan_futures
            and a.id not in engine._pending_health_checks
            and engine.agent_brains.get(a.id)
            and engine._plan_failure_cooldown_until.get(a.id, 0) <= engine.tick_count
        ]
        assert len(candidates) == 0, "Should still be on cooldown"

        # Now past cooldown.
        engine.tick_count = cooldown_until + 1
        candidates = [
            a for a in engine.world_map.actors
            if (a.needs_replan or not a.action_queue)
            and a.id not in engine._pending_plan_futures
            and a.id not in engine._pending_health_checks
            and engine.agent_brains.get(a.id)
            and engine._plan_failure_cooldown_until.get(a.id, 0) <= engine.tick_count
        ]
        assert len(candidates) == 1, "Should be eligible after cooldown expires"

    def test_multi_actor_independence(self):
        """Two actors plan and act independently; one timing out doesn't block the other."""
        engine, actors = _make_engine_with_actors(2)
        a1, a2 = actors
        brain1 = _install_mock_brain(engine, "a1", plan_result=(
            [PlannedAction(action_type="wait", params={}, reason="a1 waits")],
            "A1 waits.",
        ))
        brain2 = _install_mock_brain(engine, "a2", plan_result=(
            [PlannedAction(action_type="move", params={"direction": "east"}, reason="a2 moves")],
            "A2 moves east.",
        ))
        engine.enable_ai = True
        a1.needs_replan = True
        a1.interrupt_reason = "init"
        a2.needs_replan = True
        a2.interrupt_reason = "init"

        # Tick 0: both actors submit plans.
        engine.tick()
        assert engine.llm_call_count == 2

        # Wait for the real futures to complete (mock returns instantly in thread).
        for aid in list(engine._pending_plan_futures):
            engine._pending_plan_futures[aid].result(timeout=5)

        # Sabotage a1's future with a stuck one.
        stuck = Future()
        engine._pending_plan_futures["a1"] = stuck
        engine._pending_plan_started_tick["a1"] = engine.tick_count

        # Tick 1: a2's plan completes and executes. a1 is still pending.
        engine.tick()
        assert a2.x == 4, f"a2 should have moved east to x=4, got {a2.x}"

        # Fast-forward so a1 times out.
        engine.tick_count = PLAN_TIMEOUT_TICKS + 5
        engine.tick()
        assert engine.llm_timeout_count >= 1, "a1 should have timed out"
        assert "a1" in engine._pending_health_checks

    def test_interrupt_clears_plan_and_replans(self):
        """When an actor is interrupted, their current plan is discarded and they replan."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [
                PlannedAction(action_type="wait", params={}, reason="step 1"),
                PlannedAction(action_type="wait", params={}, reason="step 2"),
                PlannedAction(action_type="wait", params={}, reason="step 3"),
            ],
            "Three waits.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"

        # Tick 0: submit plan. Tick 1: plan arrives, first action executes.
        engine.tick()
        engine.tick()
        assert engine.applied_plan_count == 1
        assert len(actor.action_queue) == 2, "Should have 2 actions remaining"

        # Interrupt the actor — queue should be cleared, generation bumped.
        old_gen = engine._plan_generation.get("a1", 0)
        engine.interrupt_actor("a1", "something unexpected")
        assert actor.action_queue == []
        assert actor.needs_replan is True
        assert engine._plan_generation["a1"] > old_gen

        # Advance to the next replan interval boundary to submit the new plan.
        engine.tick_count = engine.replan_interval
        brain.create_plan.return_value = (
            [PlannedAction(action_type="wait", params={}, reason="recovery")],
            "Recovering.",
        )
        engine.tick()
        assert engine.llm_call_count == 2

    def test_move_blocked_triggers_interrupt_and_replan(self):
        """When a move action is blocked by a wall, the actor is interrupted and replans."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        # Place actor next to a wall — (1,1) is a wall tile.
        actor.x, actor.y = 1, 2

        brain = _install_mock_brain(engine, "a1", plan_result=(
            [
                PlannedAction(action_type="move", params={"direction": "west"}, reason="go west"),
                PlannedAction(action_type="wait", params={}, reason="unreachable"),
            ],
            "Go west.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"

        # Tick 0: submit. Tick 1: plan arrives, tries move west into wall (0,2).
        engine.tick()
        engine.tick()

        # Actor should have been interrupted (move into wall).
        assert actor.needs_replan is True
        assert actor.action_queue == [], "Queue should be cleared on interrupt"

    def test_failure_cooldown_prevents_rapid_replanning(self):
        """A plan failure applies a cooldown that prevents immediate resubmission."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1")
        # Make the plan call raise an exception.
        brain.create_plan = MagicMock(side_effect=RuntimeError("LLM crashed"))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"

        # Tick 0: submit plan that will fail.
        engine.tick()
        assert engine.llm_call_count == 1

        # Tick 1: the failed future is collected, cooldown is set.
        engine.tick()
        cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
        assert cooldown_until > engine.tick_count, "Cooldown should be active"

        # Tick 2: actor should NOT be resubmitted (still on cooldown).
        old_call_count = engine.llm_call_count
        engine.tick()
        assert engine.llm_call_count == old_call_count, "Should not resubmit during cooldown"

        # Advance to the next replan interval boundary at or after the cooldown.
        brain.create_plan = MagicMock(return_value=(
            [PlannedAction(action_type="wait", params={}, reason="recovered")],
            "Recovered.",
        ))
        engine.tick_count = ((cooldown_until) // engine.replan_interval + 1) * engine.replan_interval
        engine.tick()
        assert engine.llm_call_count == old_call_count + 1, "Should resubmit at replan interval after cooldown"

    def test_metrics_accumulate_across_full_lifecycle(self):
        """Metrics (call count, response count, timeout count) track the full flow."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [PlannedAction(action_type="wait", params={}, reason="plan1")],
            "First plan.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "init"

        # Phase 1: successful plan.
        engine.tick()  # submit
        engine.tick()  # collect + execute
        assert engine.llm_call_count == 1
        assert engine.llm_response_count == 1
        assert engine.llm_timeout_count == 0

        # Phase 2: plan that times out.
        # Set tick_count to a replan_interval boundary so non-critical replans fire.
        engine.tick_count = engine.replan_interval
        brain.create_plan.return_value = (
            [PlannedAction(action_type="wait", params={}, reason="plan2")],
            "Second plan.",
        )
        engine.tick()  # submit second plan
        assert engine.llm_call_count == 2

        # Sabotage the future.
        stuck = Future()
        engine._pending_plan_futures["a1"] = stuck
        engine._pending_plan_started_tick["a1"] = engine.tick_count

        engine.tick_count += PLAN_TIMEOUT_TICKS + 1
        engine.tick()  # timeout triggers
        assert engine.llm_timeout_count == 1

        # Phase 3: health check pass → recovery.
        hc = Future()
        hc.set_result((True, "OK"))
        engine._pending_health_checks["a1"] = hc
        engine._collect_health_check_results()

        cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
        engine.tick_count = ((cooldown_until) // engine.replan_interval + 1) * engine.replan_interval
        engine.tick()  # replan after recovery at next interval boundary
        assert engine.llm_call_count == 3
        assert engine.llm_response_count == 1  # only the first plan succeeded so far

        # Wait for the executor thread to finish before the next collection tick.
        for aid in list(engine._pending_plan_futures):
            engine._pending_plan_futures[aid].result(timeout=5)

        # Let the third plan complete.
        engine.tick()
        assert engine.llm_response_count == 2

        # Verify log has entries for all phases.
        statuses = [entry["status"] for entry in engine.llm_request_log]
        assert "ok" in statuses
        assert "timeout" in statuses

    def test_health_check_blocks_replanning_until_resolved(self):
        """While a health check is pending, the actor cannot be resubmitted for planning."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1")
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "timeout recovery"

        # Put a pending (not-done) health check in place.
        engine._pending_health_checks["a1"] = Future()  # not resolved

        old_count = engine.llm_call_count
        engine.tick()
        assert engine.llm_call_count == old_count, "Should NOT submit plan while health check pending"

    def test_replan_interval_suppresses_non_critical_replans(self):
        """Non-critical replans are suppressed between replan_interval boundaries."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [PlannedAction(action_type="wait", params={}, reason="plan")],
            "Plan.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"
        engine.replan_interval = 10

        # Tick 0 (critical): plan submitted.
        engine.tick()
        assert engine.llm_call_count == 1

        # Tick 1: plan collected, action executed, queue now empty.
        engine.tick()
        assert engine.applied_plan_count == 1
        assert not actor.action_queue
        assert not actor.interrupt_reason  # cleared on successful plan apply

        # Ticks 2-9: non-interval, non-critical — no new plan should be submitted.
        count_before = engine.llm_call_count
        for tick in range(2, 10):
            engine.tick()
        assert engine.llm_call_count == count_before, (
            f"Expected no new LLM calls between replan intervals, "
            f"got {engine.llm_call_count - count_before} extra call(s) over ticks 2-9"
        )

    def test_interrupt_queued_until_replan_interval(self):
        """Interrupts mark an actor for replan but the LLM call only fires at the next interval."""
        engine, actors = _make_engine_with_actors(1)
        actor = actors[0]
        brain = _install_mock_brain(engine, "a1", plan_result=(
            [PlannedAction(action_type="wait", params={}, reason="plan")],
            "Plan.",
        ))
        engine.enable_ai = True
        actor.needs_replan = True
        actor.interrupt_reason = "startup"
        engine.replan_interval = 10

        # Tick 0 (replan tick): plan submitted and executed.
        engine.tick()
        engine.tick()
        assert engine.llm_call_count == 1

        # Interrupt at tick 2: actor is flagged for replan but no LLM call fires yet.
        engine.interrupt_actor("a1", "attacked by enemy")
        brain.create_plan.return_value = (
            [PlannedAction(action_type="wait", params={}, reason="urgent")],
            "Urgent replan.",
        )
        engine.tick()  # tick 2 — not a replan tick
        assert engine.llm_call_count == 1, "Interrupt must wait for replan interval — no call at tick 2"

        # Advance to the next replan interval tick; the call should fire then.
        engine.tick_count = engine.replan_interval
        engine.tick()
        assert engine.llm_call_count == 2, "Should submit at next replan interval tick"


# ═══════════════════════════════════════════════════════════════════════════
# Section 2 — Live LLM integration tests (run with RUN_LIVE_LLM_TESTS=1)
# ═══════════════════════════════════════════════════════════════════════════

LIVE_LLM = os.environ.get("RUN_LIVE_LLM_TESTS", "0") == "1"
skip_unless_live = pytest.mark.skipif(not LIVE_LLM, reason="RUN_LIVE_LLM_TESTS not set")


@skip_unless_live
class TestLiveLLM:
    """Integration tests that make real LLM calls via Ollama.

    Run with:
        RUN_LIVE_LLM_TESTS=1 pytest tests/test_simulation_flow.py::TestLiveLLM -v -s
    """

    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    def _make_live_engine(self, actor_count: int = 1) -> tuple[SimulationEngine, list[Actor]]:
        world_map = _make_map()
        actors = []
        for i in range(actor_count):
            a = Actor(
                id=f"live{i + 1}",
                name=f"Traveler{i + 1}",
                role=ActorRole.PLAYER,
                x=3 + i,
                y=3,
                gold=100,
            )
            actors.append(a)
        world_map.actors = actors
        engine = SimulationEngine(
            world_map,
            enable_ai=True,
            ollama_model=self.OLLAMA_MODEL,
            ollama_base_url=self.OLLAMA_URL,
        )
        return engine, actors

    def test_live_health_check(self):
        """Verify that AgentBrain.health_check() returns a truthy response from the real LLM."""
        engine, actors = self._make_live_engine(1)
        brain = engine.agent_brains[actors[0].id]
        alive, detail = brain.health_check()
        assert alive, f"Health check should pass against a running LLM — got: {detail}"
        print(f"  Health check response: {detail!r}")

    def test_live_plan_creation(self):
        """Verify that create_plan() returns a non-empty action list from the real LLM."""
        engine, actors = self._make_live_engine(1)
        actor = actors[0]
        brain = engine.agent_brains[actor.id]
        plan, summary = brain.create_plan("initial warm-up")
        assert isinstance(plan, list), f"Expected list, got {type(plan)}"
        assert len(plan) > 0, "Plan should contain at least one action"
        print(f"  Plan ({len(plan)} steps): {summary}")
        for step in plan:
            assert hasattr(step, "action_type")
            print(f"    - {step.action_type}: {step.reason}")

    def test_live_full_tick_cycle(self):
        """Run several ticks with a real LLM and verify the engine cycles correctly.

        This exercises: plan submission → LLM response → action execution → replan.
        """
        engine, actors = self._make_live_engine(1)
        actor = actors[0]
        actor.needs_replan = True
        actor.interrupt_reason = "initial warm-up"

        # Tick 0: submits a plan request.
        updates = engine.tick()
        assert engine.llm_call_count >= 1, "Should have submitted at least one plan"
        print(f"  Tick 0 — calls: {engine.llm_call_count}, pending: {engine.llm_pending_actors}")

        # Wait for the plan to complete (poll up to 200s for slow models).
        deadline = time.monotonic() + 200
        while actor.id in engine._pending_plan_futures and time.monotonic() < deadline:
            time.sleep(1)
            engine.tick()

        assert engine.llm_response_count >= 1 or engine.llm_timeout_count >= 1, (
            "Should have received a response or timed out"
        )
        print(
            f"  After plan — responses: {engine.llm_response_count}, "
            f"timeouts: {engine.llm_timeout_count}, "
            f"applied: {engine.applied_plan_count}"
        )

        if engine.applied_plan_count > 0:
            # Execute a few more ticks to drain the action queue.
            for _ in range(5):
                engine.tick()
            print(f"  Final position: ({actor.x}, {actor.y}), queue: {len(actor.action_queue)}")

    def test_live_timeout_and_health_check_recovery(self):
        """Simulate a timeout scenario and verify the health check recovers with real LLM."""
        engine, actors = self._make_live_engine(1)
        actor = actors[0]
        brain = engine.agent_brains[actor.id]

        # Manually trigger a health check (as would happen after a timeout).
        alive, detail = brain.health_check()
        print(f"  Health check: alive={alive}, detail={detail!r}")
        assert alive, "LLM should be reachable for health check"

        # Now verify a normal plan still works after the health check.
        plan, summary = brain.create_plan("recovery after timeout health check passed")
        assert isinstance(plan, list) and len(plan) > 0, "Plan should succeed after health check"
        print(f"  Recovery plan ({len(plan)} steps): {summary}")


# ---------------------------------------------------------------------------
# Manual runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mocked_tests = [
        ("init_to_plan_to_action_execution", TestFullLifecycle().test_init_to_plan_to_action_execution),
        ("action_queue_drains_then_replans", TestFullLifecycle().test_action_queue_drains_then_replans),
        ("timeout_triggers_health_check_then_recovery", TestFullLifecycle().test_timeout_triggers_health_check_then_recovery),
        ("timeout_unhealthy_extends_cooldown", TestFullLifecycle().test_timeout_unhealthy_extends_cooldown),
        ("multi_actor_independence", TestFullLifecycle().test_multi_actor_independence),
        ("interrupt_clears_plan_and_replans", TestFullLifecycle().test_interrupt_clears_plan_and_replans),
        ("move_blocked_triggers_interrupt_and_replan", TestFullLifecycle().test_move_blocked_triggers_interrupt_and_replan),
        ("failure_cooldown_prevents_rapid_replanning", TestFullLifecycle().test_failure_cooldown_prevents_rapid_replanning),
        ("metrics_accumulate_across_full_lifecycle", TestFullLifecycle().test_metrics_accumulate_across_full_lifecycle),
        ("health_check_blocks_replanning_until_resolved", TestFullLifecycle().test_health_check_blocks_replanning_until_resolved),
        ("interrupt_queued_until_replan_interval", TestFullLifecycle().test_interrupt_queued_until_replan_interval),
        ("replan_interval_suppresses_non_critical_replans", TestFullLifecycle().test_replan_interval_suppresses_non_critical_replans),
    ]

    passed = 0
    failed = 0
    for name, fn in mocked_tests:
        try:
            fn()
            passed += 1
            print(f"PASS  {name}")
        except Exception as exc:
            import traceback
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Mocked flow tests: {passed} passed, {failed} failed out of {len(mocked_tests)}")

    if LIVE_LLM:
        print("\n--- Live LLM tests ---")
        live_tests = [
            ("live_health_check", TestLiveLLM().test_live_health_check),
            ("live_plan_creation", TestLiveLLM().test_live_plan_creation),
            ("live_full_tick_cycle", TestLiveLLM().test_live_full_tick_cycle),
            ("live_timeout_and_health_check_recovery", TestLiveLLM().test_live_timeout_and_health_check_recovery),
        ]
        for name, fn in live_tests:
            try:
                fn()
                passed += 1
                print(f"PASS  {name}")
            except Exception as exc:
                import traceback
                print(f"FAIL  {name}: {exc}")
                traceback.print_exc()
                failed += 1
        print(f"\nLive tests: {passed} passed, {failed} failed")
    else:
        print("\nSkipped live LLM tests (set RUN_LIVE_LLM_TESTS=1 to enable)")

    if failed:
        sys.exit(1)
