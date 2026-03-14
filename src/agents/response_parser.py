"""LLM response normalizer and parser.

All model-specific quirks are handled here so that brain.py stays clean.
To support a new model, add its artifact patterns to _clean() — nothing else
needs to change.

Public API
----------
clean(raw)                       -> str
parse_plan(raw, actor_name)      -> tuple[list[PlannedAction], str]
parse_trade_decision(raw, actor_name) -> tuple[bool, str]
"""

from __future__ import annotations

import json
import re

from src.models.schema import PlannedAction


# ---------------------------------------------------------------------------
# Step 1 – clean raw model output of all model-specific artifacts
# ---------------------------------------------------------------------------

# Patterns stripped in order before any parsing is attempted.
# Add new model artifacts here — brain.py never needs to change.
_ARTIFACT_PATTERNS: list[str] = [
    # Reasoning models: qwen3, deepseek-r1, o1-style
    r"<think>[\s\S]*?</think>",
    # Some models emit <reasoning>...</reasoning>
    r"<reasoning>[\s\S]*?</reasoning>",
    # Markdown code fences: ```json ... ``` or ``` ... ```
    r"```[a-zA-Z]*\s*",
    r"```",
    # XML/HTML-style preamble some models add
    r"<\?xml[^>]*\?>",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _ARTIFACT_PATTERNS]


def clean(raw: str) -> str:
    """Strip all known model-specific artifacts and return normalised text."""
    for pattern in _COMPILED:
        raw = pattern.sub("", raw)
    return raw.strip()


# ---------------------------------------------------------------------------
# Step 2 – parse normalised text into typed structures
# ---------------------------------------------------------------------------

def parse_plan(raw: str, actor_name: str) -> tuple[list[PlannedAction], str]:
    """Parse a plan response into (list[PlannedAction], summary_sentence).

    Accepts two formats the model might return:
      A)  {"summary": "...", "plan": [...]}   ← preferred new format
      B)  [{"action_type": ...}, ...]          ← bare array fallback

    Returns a guaranteed non-empty plan and a human-readable summary.
    Falls back to a single wait action on any parse failure so the engine
    always has something to execute.
    """
    content = clean(raw)
    fallback_summary = f"{actor_name} pauses, unsure what to do next."

    # --- Format A: object with "summary" + "plan" keys ---
    obj_match = re.search(r"\{[\s\S]*\}", content)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if "plan" in parsed:
                summary: str = parsed.get("summary") or fallback_summary
                plan = _build_plan(parsed.get("plan", []))
                if plan:
                    return plan, summary
        except (json.JSONDecodeError, TypeError):
            pass

    # --- Format B: bare array ---
    arr_match = re.search(r"\[[\s\S]*\]", content)
    if arr_match:
        try:
            plan = _build_plan(json.loads(arr_match.group()))
            if plan:
                return plan, fallback_summary
        except (json.JSONDecodeError, TypeError):
            pass

    return [PlannedAction(action_type="wait", reason="LLM returned no parseable JSON")], fallback_summary


def parse_trade_decision(raw: str, actor_name: str) -> tuple[bool, str]:
    """Parse a trade evaluation response into (accepted, spoken_reply).

    Expected format (two lines, order-independent):
        DECISION: ACCEPT
        RESPONSE: Sure, that sounds fair.

    Falls back to decline + silent response on parse failure.
    """
    content = clean(raw)
    accepted = False
    spoken = f"{actor_name} considers the offer silently."

    for line in content.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("DECISION:"):
            accepted = "ACCEPT" in upper
        elif upper.startswith("RESPONSE:"):
            spoken = line.split(":", 1)[1].strip()

    return accepted, spoken


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_plan(actions_data: list) -> list[PlannedAction]:
    """Convert a raw list of dicts into validated PlannedAction objects.

    Silently skips entries that are missing required fields or have wrong types.
    """
    plan: list[PlannedAction] = []
    for entry in actions_data:
        if not isinstance(entry, dict) or "action_type" not in entry:
            continue
        try:
            plan.append(PlannedAction(**entry))
        except Exception:
            continue
    return plan
