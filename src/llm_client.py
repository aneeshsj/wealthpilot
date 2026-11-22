"""Provider-agnostic LLM wrapper with structured Pydantic output and bounded repair."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv(override=True)


import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

import litellm
from langchain_litellm import ChatLiteLLM
from pydantic import BaseModel, ValidationError

from src.config import litellm_model_name, settings
from src.observability import trace_span, annotate_trace

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class StructuredOutputError(ValueError):
    """Raised when the LLM fails to produce schema-valid JSON after bounded retries."""


def get_chat_model(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> ChatLiteLLM:
    """Return a configured LangChain ChatLiteLLM instance using centralized settings."""
    init_kwargs: dict[str, Any] = {
        "model": settings.LLM_MODEL,
        "temperature": (
            settings.LLM_TEMPERATURE if temperature is None else temperature
        ),
        **kwargs,
    }
    if getattr(settings, "LLM_PROVIDER", None):
        init_kwargs["model_provider"] = settings.LLM_PROVIDER
    if getattr(settings, "LLM_API_BASE", None):
        init_kwargs["api_base"] = settings.LLM_API_BASE
    if max_tokens is not None:
        init_kwargs["max_tokens"] = max_tokens

    logger.debug("Initializing ChatLiteLLM with model=%s", settings.LLM_MODEL)
    return ChatLiteLLM(**init_kwargs)


def _extract_text_content(response: Any) -> str:
    """Safely extract string content from a LiteLLM completion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("LLM response contained no choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("LLM response choice contained no message.")

    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        text = "\n".join(parts).strip()
    else:
        text = ""

    if not text:
        raise ValueError("LLM returned empty message content.")
    return text


def _strip_json_fences(raw_text: str) -> str:
    """Strip optional markdown code fences and isolate JSON payload if wrapped."""
    cleaned = raw_text.strip()
    match = _CODE_FENCE_RE.match(cleaned)
    if match:
        cleaned = match.group(1).strip()

    if not (cleaned.startswith("{") and cleaned.endswith("}")):
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def _call_completion(
    messages: list[dict[str, str]],
    temperature: Optional[float] = None,
    use_json_mode: bool = True,
    max_tokens: Optional[int] = None,
) -> Any:
    """Invoke litellm.completion using settings.LLM_MODEL with safe JSON-mode fallback."""
    completion_kwargs: dict[str, Any] = {
        "model": litellm_model_name(settings.LLM_MODEL, settings.LLM_PROVIDER),
        "messages": messages,
        "temperature": (
            settings.LLM_TEMPERATURE if temperature is None else temperature
        ),
    }
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens
    api_base = getattr(settings, "LLM_API_BASE", None)
    if api_base:
        completion_kwargs["api_base"] = api_base
    api_key = getattr(settings, "LLM_API_KEY", None) or getattr(
        settings, "GEMINI_API_KEY", None
    )
    if api_key:
        completion_kwargs["api_key"] = api_key
    if use_json_mode:
        completion_kwargs["response_format"] = {"type": "json_object"}

    prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
    with trace_span(
        "llm.completion",
        kind="generation",
        inputs={"model": completion_kwargs["model"], "message_count": len(messages), "prompt_chars": prompt_chars},
    ):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            if use_json_mode and "response_format" in str(exc).lower():
                logger.debug(
                    "Model %s rejected response_format=json_object; retrying without it.",
                    settings.LLM_MODEL,
                )
                completion_kwargs.pop("response_format", None)
                response = litellm.completion(**completion_kwargs)
            else:
                raise
        annotate_trace(
            output={
                "model": completion_kwargs["model"],
                "response_chars": len(str(getattr(response.choices[0].message, "content", ""))),
            }
        )
        return response


def generate_structured(
    prompt: str,
    response_model: Type[T],
    max_retries: int = 2,
    *,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> T:
    """Generate a Pydantic-validated object from the configured LLM with bounded repair.

    Args:
        prompt: User/task prompt describing the required extraction or decision.
        response_model: Pydantic BaseModel class defining the target JSON schema.
        max_retries: Maximum number of repair attempts after the initial call.
        system_prompt: Optional domain system instructions prepended to schema rules.
        temperature: Sampling temperature passed to litellm.completion.
        max_tokens: Optional maximum output token limit.

    Returns:
        An instance of `response_model` validated against its Pydantic schema.

    Raises:
        ValueError: If `max_retries` is negative or `prompt` is empty.
        StructuredOutputError: If parsing/validation fails across all attempts.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    schema_str = json.dumps(response_model.model_json_schema(), indent=2)
    base_system = (
        f"{system_prompt.strip()}\n\n" if system_prompt and system_prompt.strip() else ""
    )
    system_message = (
        f"{base_system}"
        "You must respond with ONLY a single valid JSON object matching the following "
        "JSON Schema. Do not include markdown fences, commentary, or extra keys unless "
        "permitted by the schema.\n\n"
        f"JSON Schema ({response_model.__name__}):\n{schema_str}"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    last_error: Optional[Exception] = None
    total_attempts = max_retries + 1

    for attempt in range(total_attempts):
        raw_text = ""
        try:
            logger.debug(
                "Requesting structured output for %s (attempt %d/%d, model=%s)",
                response_model.__name__,
                attempt + 1,
                total_attempts,
                settings.LLM_MODEL,
            )
            response = _call_completion(
                messages=messages,
                temperature=temperature,
                use_json_mode=True,
                max_tokens=max_tokens,
            )
            raw_text = _extract_text_content(response)
            cleaned_json = _strip_json_fences(raw_text)
            parsed = response_model.model_validate_json(cleaned_json)
            logger.info(
                "Successfully parsed %s on attempt %d/%d",
                response_model.__name__,
                attempt + 1,
                total_attempts,
            )
            return parsed
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "Structured output validation failed for %s on attempt %d/%d: %s",
                response_model.__name__,
                attempt + 1,
                total_attempts,
                exc,
            )
            if attempt >= max_retries:
                break

            if raw_text:
                messages.append({"role": "assistant", "content": raw_text})

            repair_instruction = (
                "Your previous response failed JSON/schema validation with the following "
                f"error:\n{exc}\n\n"
                "Return ONLY a corrected, valid JSON object that strictly conforms to the "
                f"{response_model.__name__} JSON Schema:\n{schema_str}"
            )
            messages.append({"role": "user", "content": repair_instruction})

    raise StructuredOutputError(
        f"Failed to generate valid {response_model.__name__} after {total_attempts} "
        f"attempt(s): {last_error}"
    ) from last_error
