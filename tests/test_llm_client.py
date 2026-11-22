"""Unit tests for src/llm_client.py."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import src.llm_client as llm_client
from src.config import litellm_model_name, settings
from src.llm_client import StructuredOutputError, generate_structured, get_chat_model


class SampleDecision(BaseModel):
    """Test Pydantic model for structured output validation."""

    status: str
    score: float


def _make_response(content: Any) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_get_chat_model_uses_settings_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure get_chat_model reads settings.LLM_MODEL and forwards parameters."""
    captured: dict[str, Any] = {}

    class DummyChatLiteLLM:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm_client, "ChatLiteLLM", DummyChatLiteLLM)

    model = get_chat_model(temperature=0.2, max_tokens=256)
    assert isinstance(model, DummyChatLiteLLM)
    assert captured["model"] == settings.LLM_MODEL
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 256


def test_generate_structured_success_plain_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return validated Pydantic model when LLM returns valid JSON."""
    recorded_calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        recorded_calls.append(kwargs)
        return _make_response('{"status": "approve", "score": 0.92}')

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    result = generate_structured("Evaluate SME application", SampleDecision)
    assert result == SampleDecision(status="approve", score=0.92)
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["model"] == litellm_model_name(
        settings.LLM_MODEL, settings.LLM_PROVIDER
    )
    assert recorded_calls[0]["response_format"] == {"type": "json_object"}


def test_generate_structured_strips_markdown_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle JSON wrapped in markdown code fences gracefully."""
    fenced_payload = '```json\n{"status": "borderline", "score": 0.61}\n```'

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        return _make_response(fenced_payload)

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    result = generate_structured("Evaluate SME application", SampleDecision)
    assert result == SampleDecision(status="borderline", score=0.61)


def test_generate_structured_handles_list_content_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Extract JSON text when provider returns structured content block lists."""

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        return _make_response(
            [{"type": "text", "text": '{"status": "approve", "score": 0.85}'}]
        )

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    result = generate_structured("Evaluate SME application", SampleDecision)
    assert result == SampleDecision(status="approve", score=0.85)


def test_generate_structured_repairs_invalid_output_within_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass validation error back to LLM and succeed on bounded retry."""
    calls: list[list[dict[str, str]]] = []
    responses = iter(
        [
            _make_response('{"status": "approve", "score": "not-a-float"}'),
            _make_response('{"status": "approve", "score": 0.88}'),
        ]
    )

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        calls.append([dict(msg) for msg in kwargs["messages"]])
        return next(responses)

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    result = generate_structured(
        "Evaluate SME application",
        SampleDecision,
        max_retries=2,
        system_prompt="You are a credit underwriter.",
    )
    assert result == SampleDecision(status="approve", score=0.88)
    assert len(calls) == 2
    assert calls[1][-2]["role"] == "assistant"
    assert calls[1][-1]["role"] == "user"
    assert "failed JSON/schema validation" in calls[1][-1]["content"]


def test_generate_structured_exhausts_retries_raises_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise StructuredOutputError once max_retries is exhausted."""
    attempt_count = 0

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        nonlocal attempt_count
        attempt_count += 1
        return _make_response('{"status": "approve"}')  # missing required 'score'

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    with pytest.raises(StructuredOutputError, match="SampleDecision"):
        generate_structured("Evaluate SME application", SampleDecision, max_retries=2)

    assert attempt_count == 3


def test_generate_structured_response_format_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry without response_format if provider rejects json_object mode."""
    seen_response_formats: list[Any] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        seen_response_formats.append(kwargs.get("response_format"))
        if "response_format" in kwargs:
            raise ValueError("Unsupported parameter: response_format")
        return _make_response('{"status": "decline", "score": 0.15}')

    monkeypatch.setattr(llm_client.litellm, "completion", fake_completion)

    result = generate_structured("Evaluate SME application", SampleDecision)
    assert result == SampleDecision(status="decline", score=0.15)
    assert seen_response_formats == [{"type": "json_object"}, None]


def test_generate_structured_rejects_invalid_arguments() -> None:
    """Reject negative max_retries and blank prompts immediately."""
    with pytest.raises(ValueError, match="max_retries"):
        generate_structured("Valid prompt", SampleDecision, max_retries=-1)

    with pytest.raises(ValueError, match="prompt"):
        generate_structured("   ", SampleDecision)

