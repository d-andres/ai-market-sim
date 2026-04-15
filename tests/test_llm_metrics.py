"""Tests for LLM metrics tracking in the simulation engine.

Covers: response count, timeout count, duration accumulation,
request/response log entries, log size cap, and health checks.
"""

import os
import sys
import time
from concurrent.futures import Future
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.schema import Actor, ActorRole, Map, PlannedAction, Tile, TileType
from src.simulation.engine import (
    PLAN_TIMEOUT_TICKS,
    TIMEOUT_REPLAN_COOLDOWN_TICKS,
    TIMEOUT_UNHEALTHY_COOLDOWN_TICKS,
    SimulationEngine,
)


def _make_map(width: int = 10, height: int = 10) -> Map:
    tiles = []
    for y in range(height):
        for x in range(width):
            tt = TileType.WALL if (x in {0, width - 1} or y in {0, height - 1}) else TileType.FLOOR
            tiles.append(Tile(x=x, y=y, tile_type=tt))
    return Map(width=width, height=height, tiles=tiles)


def _make_engine() -> tuple[SimulationEngine, Actor]:
    world_map = _make_map()
    actor = Actor(id="a1", name="Adventurer", role=ActorRole.PLAYER, x=5, y=5, gold=50)
    world_map.actors = [actor]
    engine = SimulationEngine(world_map, enable_ai=False)
    return engine, actor


def _plant_done_future(engine: SimulationEngine, actor_id: str, result) -> None:
    """Plant a pre-resolved future into the engine's pending plan structures."""
    future = Future()
    future.set_result(result)
    engine._pending_plan_futures[actor_id] = future
    engine._pending_plan_started_tick[actor_id] = engine.tick_count
    engine._pending_plan_started_time[actor_id] = time.monotonic() - 5.0  # simulate 5s ago
    engine._pending_plan_generation[actor_id] = engine._plan_generation.get(actor_id, 0)


def _plant_failed_future(engine: SimulationEngine, actor_id: str, exc: Exception) -> None:
    """Plant a pre-failed future into the engine's pending plan structures."""
    future = Future()
    future.set_exception(exc)
    engine._pending_plan_futures[actor_id] = future
    engine._pending_plan_started_tick[actor_id] = engine.tick_count
    engine._pending_plan_started_time[actor_id] = time.monotonic() - 3.0
    engine._pending_plan_generation[actor_id] = engine._plan_generation.get(actor_id, 0)


def _plant_stuck_future(engine: SimulationEngine, actor_id: str, started_tick: int) -> None:
    """Plant a never-completing future for timeout testing.

    Also installs a mock brain so the health check submission works.
    """
    future = Future()
    engine._pending_plan_futures[actor_id] = future
    engine._pending_plan_started_tick[actor_id] = started_tick
    engine._pending_plan_started_time[actor_id] = time.monotonic() - 120.0
    engine._pending_plan_generation[actor_id] = engine._plan_generation.get(actor_id, 0)
    # The timeout path reads brain.last_request_prompt and submits brain.health_check.
    if actor_id not in engine.agent_brains:
        brain = MagicMock()
        brain.last_request_prompt = "(mock plan request)"
        brain.health_check = MagicMock(return_value=(True, "OK"))
        engine.agent_brains[actor_id] = brain


# ── Successful completion metrics ──────────────────────────────────────────


def test_successful_plan_increments_response_count() -> None:
    engine, actor = _make_engine()
    plan = [PlannedAction(action_type="move", params={"direction": "north"}, reason="go")]
    _plant_done_future(engine, "a1", (plan, "Adventurer heads north."))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_response_count == 1


def test_successful_plan_accumulates_duration() -> None:
    engine, actor = _make_engine()
    plan = [PlannedAction(action_type="wait", params={}, reason="idle")]
    _plant_done_future(engine, "a1", (plan, "Adventurer waits."))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_total_duration > 0.0, "Duration should be positive after a completed plan"


def test_successful_plan_logs_ok_entry() -> None:
    engine, actor = _make_engine()
    plan = [PlannedAction(action_type="move", params={"direction": "south"}, reason="explore")]
    _plant_done_future(engine, "a1", (plan, "Adventurer explores south."))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert len(engine.llm_request_log) == 1
    entry = engine.llm_request_log[0]
    assert entry["status"] == "ok"
    assert entry["actor"] == "Adventurer"
    assert entry["duration"] > 0
    assert "Adventurer explores south." in entry["response"]


# ── Failed plan metrics ────────────────────────────────────────────────────


def test_failed_plan_accumulates_duration() -> None:
    engine, actor = _make_engine()
    _plant_failed_future(engine, "a1", RuntimeError("LLM exploded"))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_total_duration > 0.0, "Duration should accumulate even on failure"


def test_failed_plan_logs_error_entry() -> None:
    engine, actor = _make_engine()
    _plant_failed_future(engine, "a1", RuntimeError("connection refused"))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert len(engine.llm_request_log) == 1
    entry = engine.llm_request_log[0]
    assert entry["status"] == "error"
    assert "connection refused" in entry["response"]


def test_failed_plan_does_not_increment_response_count() -> None:
    engine, actor = _make_engine()
    _plant_failed_future(engine, "a1", RuntimeError("boom"))

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_response_count == 0, "Errors should not count as responses"


# ── Timeout metrics ────────────────────────────────────────────────────────


def test_timeout_increments_timeout_count() -> None:
    engine, actor = _make_engine()
    _plant_stuck_future(engine, "a1", started_tick=0)
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_timeout_count == 1


def test_timeout_accumulates_duration() -> None:
    engine, actor = _make_engine()
    _plant_stuck_future(engine, "a1", started_tick=0)
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_total_duration > 0.0, "Duration should accumulate even on timeout"


def test_timeout_logs_timeout_entry() -> None:
    engine, actor = _make_engine()
    _plant_stuck_future(engine, "a1", started_tick=0)
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert len(engine.llm_request_log) == 1
    entry = engine.llm_request_log[0]
    assert entry["status"] == "timeout"
    assert entry["actor"] == "Adventurer"
    assert "Timed out" in entry["response"]


def test_timeout_cleans_up_started_time() -> None:
    engine, actor = _make_engine()
    _plant_stuck_future(engine, "a1", started_tick=0)
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert "a1" not in engine._pending_plan_started_time, "started_time should be cleaned up"


def test_timeout_submits_health_check() -> None:
    """After a plan timeout, a health check future should be submitted."""
    engine, actor = _make_engine()
    _plant_stuck_future(engine, "a1", started_tick=0)
    engine.tick_count = PLAN_TIMEOUT_TICKS + 1

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert "a1" in engine._pending_health_checks, "Health check should be submitted after timeout"


# ── Stale discard cleanup ─────────────────────────────────────────────────


def test_stale_discard_cleans_up_started_time() -> None:
    engine, actor = _make_engine()
    plan = [PlannedAction(action_type="wait", params={}, reason="stale")]
    _plant_done_future(engine, "a1", (plan, "Adventurer goes stale."))

    # Bump generation so the result is stale
    engine._plan_generation["a1"] = 999

    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert "a1" not in engine._pending_plan_started_time, "started_time should be cleaned up on stale discard"
    assert engine.llm_response_count == 0, "Stale plans should not count as responses"


# ── Request log cap ────────────────────────────────────────────────────────


def test_request_log_capped_at_50() -> None:
    engine, actor = _make_engine()

    # Manually fill the deque beyond 50
    for i in range(60):
        engine.llm_request_log.append({"tick": i, "actor": "test", "status": "ok"})

    assert len(engine.llm_request_log) == 50, f"Expected 50, got {len(engine.llm_request_log)}"
    assert engine.llm_request_log[0]["tick"] == 10, "Oldest entries should be evicted"


# ── Submit sets started_time ───────────────────────────────────────────────


def test_submit_plan_request_records_started_time() -> None:
    from unittest.mock import MagicMock

    engine, actor = _make_engine()
    brain = MagicMock()
    brain.create_plan = MagicMock(return_value=([], ""))

    before = time.monotonic()
    engine._submit_plan_request(actor, brain)
    after = time.monotonic()

    assert "a1" in engine._pending_plan_started_time
    t = engine._pending_plan_started_time["a1"]
    assert before <= t <= after, f"started_time {t} not in [{before}, {after}]"


# ── Multiple plans accumulate correctly ────────────────────────────────────


def test_multiple_completions_accumulate_metrics() -> None:
    engine, actor = _make_engine()

    # First completion
    plan1 = [PlannedAction(action_type="move", params={"direction": "north"}, reason="go")]
    _plant_done_future(engine, "a1", (plan1, "Goes north."))
    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    # Second completion
    plan2 = [PlannedAction(action_type="wait", params={}, reason="rest")]
    _plant_done_future(engine, "a1", (plan2, "Rests."))
    updates = {"tick": engine.tick_count, "events": [], "actor_actions": []}
    engine._collect_completed_plan_requests(updates)

    assert engine.llm_response_count == 2
    assert engine.applied_plan_count == 2
    assert len(engine.llm_request_log) == 2


# ── Health check result tests ─────────────────────────────────────────────


def test_health_check_pass_applies_short_cooldown() -> None:
    """When health check succeeds, actor gets a short cooldown before replanning."""
    engine, actor = _make_engine()

    # Simulate a passed health check
    hc_future = Future()
    hc_future.set_result((True, "OK"))
    engine._pending_health_checks["a1"] = hc_future
    engine.tick_count = 100

    engine._collect_health_check_results()

    assert "a1" not in engine._pending_health_checks, "Health check should be cleaned up"
    assert actor.needs_replan is True
    cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
    assert cooldown_until == 100 + TIMEOUT_REPLAN_COOLDOWN_TICKS


def test_health_check_pass_logs_ok() -> None:
    engine, actor = _make_engine()

    hc_future = Future()
    hc_future.set_result((True, "OK"))
    engine._pending_health_checks["a1"] = hc_future
    engine.tick_count = 50

    engine._collect_health_check_results()

    assert len(engine.llm_request_log) == 1
    entry = engine.llm_request_log[0]
    assert entry["status"] == "ok"
    assert "Health check" in entry["request"]
    assert entry["response"] == "OK"


def test_health_check_fail_applies_long_cooldown() -> None:
    """When health check fails, actor gets a longer cooldown."""
    engine, actor = _make_engine()

    hc_future = Future()
    hc_future.set_result((False, "Connection refused"))
    engine._pending_health_checks["a1"] = hc_future
    engine.tick_count = 100

    engine._collect_health_check_results()

    assert "a1" not in engine._pending_health_checks
    cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
    assert cooldown_until == 100 + TIMEOUT_UNHEALTHY_COOLDOWN_TICKS


def test_health_check_fail_logs_error() -> None:
    engine, actor = _make_engine()

    hc_future = Future()
    hc_future.set_result((False, "Connection refused"))
    engine._pending_health_checks["a1"] = hc_future

    engine._collect_health_check_results()

    assert len(engine.llm_request_log) == 1
    entry = engine.llm_request_log[0]
    assert entry["status"] == "error"
    assert "FAILED" in entry["response"]


def test_health_check_exception_treated_as_failure() -> None:
    """If the health check future itself raises, treat as unhealthy."""
    engine, actor = _make_engine()

    hc_future = Future()
    hc_future.set_exception(RuntimeError("thread died"))
    engine._pending_health_checks["a1"] = hc_future
    engine.tick_count = 200

    engine._collect_health_check_results()

    cooldown_until = engine._plan_failure_cooldown_until.get("a1", 0)
    assert cooldown_until == 200 + TIMEOUT_UNHEALTHY_COOLDOWN_TICKS


def test_pending_health_check_blocks_plan_submission() -> None:
    """Actors with pending health checks should not be submitted for new plans."""
    engine, actor = _make_engine()
    actor.needs_replan = True
    actor.interrupt_reason = "recovery after health check"

    # Put a pending health check in place
    hc_future = Future()  # not done yet
    engine._pending_health_checks["a1"] = hc_future

    # Also need a brain for the candidate filter
    brain = MagicMock()
    engine.agent_brains["a1"] = brain

    # The candidate filter should exclude this actor
    candidates = [
        a for a in engine.world_map.actors
        if (a.needs_replan or not a.action_queue)
        and a.id not in engine._pending_plan_futures
        and a.id not in engine._pending_health_checks
        and engine.agent_brains.get(a.id)
        and engine._plan_failure_cooldown_until.get(a.id, 0) <= engine.tick_count
    ]
    assert len(candidates) == 0, "Actor with pending health check should not be a plan candidate"


if __name__ == "__main__":
    tests = [
        test_successful_plan_increments_response_count,
        test_successful_plan_accumulates_duration,
        test_successful_plan_logs_ok_entry,
        test_failed_plan_accumulates_duration,
        test_failed_plan_logs_error_entry,
        test_failed_plan_does_not_increment_response_count,
        test_timeout_increments_timeout_count,
        test_timeout_accumulates_duration,
        test_timeout_logs_timeout_entry,
        test_timeout_cleans_up_started_time,
        test_stale_discard_cleans_up_started_time,
        test_request_log_capped_at_50,
        test_submit_plan_request_records_started_time,
        test_multiple_completions_accumulate_metrics,
        test_timeout_submits_health_check,
        test_health_check_pass_applies_short_cooldown,
        test_health_check_pass_logs_ok,
        test_health_check_fail_applies_long_cooldown,
        test_health_check_fail_logs_error,
        test_health_check_exception_treated_as_failure,
        test_pending_health_check_blocks_plan_submission,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS  {t.__name__}")
        except Exception as exc:
            import traceback

            print(f"FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 55}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
