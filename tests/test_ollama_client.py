"""Tests for the OllamaClient and message types."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.ollama_client import ChatMessage, MessageRole, OllamaClient, OllamaResponse


# ---------------------------------------------------------------------------
# Message / response types
# ---------------------------------------------------------------------------

def test_message_role_values():
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"


def test_chat_message_with_string_content():
    msg = ChatMessage(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"


def test_chat_message_with_list_content():
    msg = ChatMessage(
        role=MessageRole.USER,
        content=[{"type": "text", "text": "Hello world"}],
    )
    assert isinstance(msg.content, list)
    assert msg.content[0]["text"] == "Hello world"


def test_ollama_response_defaults():
    resp = OllamaResponse()
    assert resp.content == ""
    assert resp.thinking == ""
    assert resp.duration == 0.0


def test_ollama_response_with_values():
    resp = OllamaResponse(content="answer", thinking="thought", duration=1.5)
    assert resp.content == "answer"
    assert resp.thinking == "thought"
    assert resp.duration == 1.5


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------

def test_client_strips_ollama_prefix():
    client = OllamaClient(model_id="ollama/qwen3:4b")
    assert client.model_id == "qwen3:4b"


def test_client_no_prefix():
    client = OllamaClient(model_id="qwen3:4b")
    assert client.model_id == "qwen3:4b"


def test_client_strips_trailing_slash():
    client = OllamaClient(api_base="http://localhost:11434/")
    assert client.api_base == "http://localhost:11434"


def test_client_default_params():
    client = OllamaClient()
    assert client.timeout == 180
    assert client.max_tokens == 4096
    assert client.num_ctx == 8192


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def test_convert_string_content():
    msgs = [ChatMessage(role=MessageRole.USER, content="Hi")]
    converted = OllamaClient._convert_messages(msgs)
    assert converted == [{"role": "user", "content": "Hi"}]


def test_convert_list_content():
    msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[{"type": "text", "text": "You are helpful"}],
        ),
    ]
    converted = OllamaClient._convert_messages(msgs)
    assert converted == [{"role": "system", "content": "You are helpful"}]


def test_convert_multiple_messages():
    msgs = [
        ChatMessage(role=MessageRole.SYSTEM, content="System prompt"),
        ChatMessage(role=MessageRole.USER, content="User question"),
    ]
    converted = OllamaClient._convert_messages(msgs)
    assert len(converted) == 2
    assert converted[0]["role"] == "system"
    assert converted[1]["role"] == "user"


def test_convert_list_content_multiple_parts():
    msgs = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        ),
    ]
    converted = OllamaClient._convert_messages(msgs)
    assert converted[0]["content"] == "Part 1\nPart 2"


# ---------------------------------------------------------------------------
# think parameter
# ---------------------------------------------------------------------------

def test_client_think_default_true():
    client = OllamaClient()
    assert client.think is True


def test_client_think_false():
    client = OllamaClient(think=False)
    assert client.think is False


# ---------------------------------------------------------------------------
# _extract_json_from_thinking
# ---------------------------------------------------------------------------

def test_extract_json_simple():
    thinking = 'Let me think... {"summary": "hello", "plan": []}'
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"summary"' in result
    assert '"plan"' in result


def test_extract_json_with_trailing_whitespace():
    thinking = 'Reasoning... {"key": "value"}  \n  '
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"key"' in result


def test_extract_json_nested():
    thinking = 'Some text {"summary": "ok", "plan": [{"action_type": "move"}]}'
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"action_type"' in result


def test_extract_json_multiple_objects_takes_valid():
    # Intermediate braces in reasoning shouldn't grab too much
    thinking = 'Comparing {"x": 1} vs {"y": 2}. Final: {"answer": "done"}'
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"answer"' in result


def test_extract_json_empty_thinking():
    assert OllamaClient._extract_json_from_thinking("") == ""


def test_extract_json_no_json():
    assert OllamaClient._extract_json_from_thinking("Just plain text.") == ""


def test_extract_json_incomplete():
    thinking = "Some reasoning { incomplete"
    assert OllamaClient._extract_json_from_thinking(thinking) == ""


def test_extract_json_trailing_text_after_json():
    """When thinking continues past the JSON (model got cut off), still extract it."""
    thinking = (
        'Let me think about this.\n'
        '{"summary": "hello", "plan": [{"action_type": "wait"}]}\n'
        'This is valid. However the problem says: "exactly two'
    )
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"summary"' in result
    assert '"plan"' in result


def test_extract_json_prefers_outermost_object():
    """Scan backwards: find the largest valid JSON at each { position."""
    thinking = (
        'Analysis: {"action_type": "move"} is one option.\n'
        'Final: {"summary": "go north", "plan": [{"action_type": "move", "params": {"direction": "north"}, "reason": "explore"}]}'
    )
    result = OllamaClient._extract_json_from_thinking(thinking)
    assert '"summary"' in result
    assert '"plan"' in result


if __name__ == "__main__":
    test_message_role_values()
    test_chat_message_with_string_content()
    test_chat_message_with_list_content()
    test_ollama_response_defaults()
    test_ollama_response_with_values()
    test_client_strips_ollama_prefix()
    test_client_no_prefix()
    test_client_strips_trailing_slash()
    test_client_default_params()
    test_convert_string_content()
    test_convert_list_content()
    test_convert_multiple_messages()
    test_convert_list_content_multiple_parts()
    print("All ollama client tests passed!")
