"""Direct Ollama HTTP client that captures thinking content.

Ollama's ``/api/chat`` endpoint returns a separate ``thinking`` field for
reasoning models (qwen3, deepseek-r1, etc.).  This lightweight wrapper
calls the endpoint directly via ``requests`` and exposes both ``content``
and ``thinking`` from the response.

Includes simple :class:`ChatMessage` / :class:`MessageRole` types so the
rest of the codebase has zero dependency on ``smolagents`` or ``litellm``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import requests


# ---------------------------------------------------------------------------
# Lightweight message types
# ---------------------------------------------------------------------------

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Minimal chat message compatible with Ollama's ``/api/chat`` format."""
    role: MessageRole
    content: str | list[dict[str, Any]]


@dataclass
class OllamaResponse:
    """Container returned by :class:`OllamaClient.__call__`."""
    content: str = ""
    thinking: str = ""
    duration: float = 0.0


class OllamaClient:
    """Minimal synchronous Ollama chat client.

    Parameters
    ----------
    model_id : str
        Model name as known by Ollama (e.g. ``"qwen3:4b"``).
        An ``"ollama/"`` prefix is stripped automatically for convenience.
    api_base : str
        Ollama server base URL (default ``http://localhost:11434``).
    timeout : int
        HTTP request timeout in seconds.
    max_tokens : int
        Maps to Ollama ``num_predict`` option.
    num_ctx : int
        Context window size passed to Ollama.
    think : bool
        Enable the model's thinking/reasoning mode.  When *False*,
        ``"think": false`` is sent in the request payload so all output
        goes directly to the ``content`` field.
    """

    def __init__(
        self,
        model_id: str = "qwen3:4b",
        api_base: str = "http://localhost:11434",
        timeout: int = 180,
        max_tokens: int = 4096,
        num_ctx: int = 8192,
        think: bool = True,
    ) -> None:
        # Strip the "ollama/" prefix for convenience.
        self.model_id = model_id.removeprefix("ollama/")
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.think = think

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[ChatMessage]) -> list[dict]:
        """Convert :class:`ChatMessage` objects to Ollama's wire format."""
        converted: list[dict] = []
        for msg in messages:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            # ChatMessage.content can be a plain str or a list of typed parts
            # (e.g. [{"type": "text", "text": "..."}]).
            if isinstance(msg.content, str):
                text = msg.content
            elif isinstance(msg.content, list):
                parts = []
                for part in msg.content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    else:
                        parts.append(str(part))
                text = "\n".join(parts)
            else:
                text = str(msg.content)
            converted.append({"role": role, "content": text})
        return converted

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, messages: list[ChatMessage]) -> OllamaResponse:
        """Send a chat completion request and return the response.

        Raises ``requests.RequestException`` on network errors and
        ``RuntimeError`` if the response payload is malformed.
        """
        payload: dict = {
            "model": self.model_id,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if not self.think:
            payload["think"] = False

        t0 = time.monotonic()
        resp = requests.post(
            f"{self.api_base}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        elapsed = round(time.monotonic() - t0, 3)
        resp.raise_for_status()

        data = resp.json()
        message = data.get("message", {})

        content = message.get("content", "")
        thinking = message.get("thinking", "")

        # Thinking models (qwen3, deepseek-r1, etc.) may exhaust num_predict
        # on reasoning tokens, leaving content empty.  When that happens,
        # extract the last JSON object from the thinking field so callers
        # still receive a parseable response.
        if not content.strip() and thinking:
            content = self._extract_json_from_thinking(thinking)

        return OllamaResponse(
            content=content,
            thinking=thinking,
            duration=elapsed,
        )

    @staticmethod
    def _extract_json_from_thinking(thinking: str) -> str:
        """Extract the last valid JSON object from thinking text.

        Thinking models often place the final JSON answer inside the
        reasoning stream and may continue writing after it (or get cut
        off mid-sentence).  This method scans backwards through ``{``
        positions, pairs each with the last ``}`` that follows it, and
        returns the first candidate that parses as valid JSON.
        """
        import json as _json

        text = thinking.strip()
        for i in range(len(text) - 1, -1, -1):
            if text[i] == "{":
                last_brace = text.rfind("}", i)
                if last_brace == -1:
                    continue
                candidate = text[i : last_brace + 1]
                try:
                    _json.loads(candidate)
                    return candidate
                except (ValueError, _json.JSONDecodeError):
                    continue
        return ""
