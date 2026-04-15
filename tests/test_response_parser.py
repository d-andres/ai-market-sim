"""Response parser tests.

Covers:
1. clean() strips <think>...</think> tags
2. clean() strips markdown code fences
3. clean() strips <reasoning>...</reasoning> tags
4. parse_plan() handles format A (object with summary + plan)
5. parse_plan() handles format B (bare array)
6. parse_plan() returns wait fallback on garbage input
7. parse_trade_decision() parses ACCEPT correctly
8. parse_trade_decision() parses DECLINE correctly
9. parse_trade_decision() defaults to decline on garbage
10. _build_plan() skips invalid entries gracefully

No LLM calls are made.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.response_parser import clean, parse_plan, parse_trade_decision


def test_clean_strips_think_tags():
    raw = "<think>Let me reason about this.</think>Here is the real output."
    result = clean(raw)
    assert "<think>" not in result
    assert "Here is the real output." in result
    print("PASS  test_clean_strips_think_tags")


def test_clean_strips_markdown_fences():
    raw = '```json\n{"key": "value"}\n```'
    result = clean(raw)
    assert "```" not in result
    assert '"key"' in result
    print("PASS  test_clean_strips_markdown_fences")


def test_clean_strips_reasoning_tags():
    raw = "<reasoning>Some model reasoning.</reasoning>Actual content."
    result = clean(raw)
    assert "<reasoning>" not in result
    assert "Actual content." in result
    print("PASS  test_clean_strips_reasoning_tags")


def test_clean_handles_nested_think():
    raw = "<think>step 1\nstep 2\n</think>\n{\"plan\": []}"
    result = clean(raw)
    assert "<think>" not in result
    assert '{"plan": []}' in result
    print("PASS  test_clean_handles_nested_think")


def test_parse_plan_format_a():
    raw = '{"summary": "Guard patrols east.", "plan": [{"action_type": "move", "params": {"direction": "east"}, "reason": "patrol"}]}'
    plan, summary = parse_plan(raw, "Guard")
    assert len(plan) == 1
    assert plan[0].action_type == "move"
    assert plan[0].params["direction"] == "east"
    assert "patrols east" in summary
    print("PASS  test_parse_plan_format_a")


def test_parse_plan_format_b():
    raw = '[{"action_type": "wait", "params": {}, "reason": "observing"}]'
    plan, summary = parse_plan(raw, "Bob")
    assert len(plan) == 1
    assert plan[0].action_type == "wait"
    print("PASS  test_parse_plan_format_b")


def test_parse_plan_garbage_returns_wait():
    raw = "I don't know what to do today."
    plan, summary = parse_plan(raw, "Alice")
    assert len(plan) == 1
    assert plan[0].action_type == "wait"
    print("PASS  test_parse_plan_garbage_returns_wait")


def test_parse_plan_with_think_tags():
    raw = '<think>Planning steps...</think>{"summary": "Moving north.", "plan": [{"action_type": "move", "params": {"direction": "north"}, "reason": ""}]}'
    plan, summary = parse_plan(raw, "Test")
    assert len(plan) == 1
    assert plan[0].action_type == "move"
    assert plan[0].params["direction"] == "north"
    print("PASS  test_parse_plan_with_think_tags")


def test_parse_trade_accept():
    raw = "DECISION: ACCEPT\nRESPONSE: Sounds like a fair deal!"
    accepted, spoken = parse_trade_decision(raw, "Elara")
    assert accepted is True
    assert "fair deal" in spoken
    print("PASS  test_parse_trade_accept")


def test_parse_trade_decline():
    raw = "DECISION: DECLINE\nRESPONSE: That offer insults me."
    accepted, spoken = parse_trade_decision(raw, "Elara")
    assert accepted is False
    assert "insults" in spoken
    print("PASS  test_parse_trade_decline")


def test_parse_trade_garbage_defaults_decline():
    raw = "I have no idea what you're saying."
    accepted, spoken = parse_trade_decision(raw, "Bob")
    assert accepted is False
    print("PASS  test_parse_trade_garbage_defaults_decline")


def test_parse_plan_skips_invalid_entries():
    raw = '[{"action_type": "move", "params": {"direction": "north"}, "reason": ""}, {"bad": "entry"}, {"action_type": "wait", "params": {}, "reason": ""}]'
    plan, summary = parse_plan(raw, "Test")
    assert len(plan) == 2
    assert plan[0].action_type == "move"
    assert plan[1].action_type == "wait"
    print("PASS  test_parse_plan_skips_invalid_entries")


if __name__ == "__main__":
    tests = [
        test_clean_strips_think_tags,
        test_clean_strips_markdown_fences,
        test_clean_strips_reasoning_tags,
        test_clean_handles_nested_think,
        test_parse_plan_format_a,
        test_parse_plan_format_b,
        test_parse_plan_garbage_returns_wait,
        test_parse_plan_with_think_tags,
        test_parse_trade_accept,
        test_parse_trade_decline,
        test_parse_trade_garbage_defaults_decline,
        test_parse_plan_skips_invalid_entries,
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
