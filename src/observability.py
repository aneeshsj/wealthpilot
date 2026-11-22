"""Observability, PII/secret-safe structured logging, circuit breaking, and resilient calls."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import functools
import inspect
import json
import logging
import random
import re
import sys
import threading
import time
import os
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, ParamSpec, TextIO, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

_ACTIVE_TRACE: ContextVar["TraceRun | None"] = ContextVar("wealthpilot_trace", default=None)


@dataclass(slots=True)
class TraceSpan:
    """PII-safe local span used for trace export and KPI aggregation."""

    name: str
    kind: str
    started_at: float
    ended_at: float | None = None
    status: str = "ok"
    input_summary: dict[str, Any] = field(default_factory=dict)
    output_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    parent: str | None = None
    langfuse_context: Any = field(default=None, repr=False)
    langfuse_observation: Any = field(default=None, repr=False)

    @property
    def latency_ms(self) -> float:
        return round(((self.ended_at or time.perf_counter()) - self.started_at) * 1000.0, 3)


@dataclass(slots=True)
class TraceRun:
    """One request trace with nested node/tool/LLM spans and run KPIs."""

    trace_id: str
    session_id: str
    user_id: str
    started_at: float = field(default_factory=time.perf_counter)
    spans: list[TraceSpan] = field(default_factory=list)
    error_count: int = 0
    fallback_count: int = 0
    langfuse_context: Any = field(default=None, repr=False)
    langfuse_observation: Any = field(default=None, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": mask_sensitive_text(self.session_id),
            "user_id": mask_sensitive_text(self.user_id),
            "latency_ms": round((time.perf_counter() - self.started_at) * 1000.0, 3),
            "span_count": len(self.spans),
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "spans": [
                {
                    "name": span.name,
                    "kind": span.kind,
                    "latency_ms": span.latency_ms,
                    "status": span.status,
                    "input": mask_sensitive_data(span.input_summary),
                    "output": mask_sensitive_data(span.output_summary),
                    "error": mask_sensitive_text(span.error or ""),
                    "parent": span.parent,
                }
                for span in self.spans
            ],
        }


def start_trace(session_id: str, user_id: str, trace_id: str | None = None) -> TraceRun:
    """Start a local trace and, when configured, its Langfuse root observation."""
    import uuid

    run = TraceRun(trace_id or uuid.uuid4().hex, session_id, user_id)
    langfuse_context, langfuse_observation = _start_langfuse_observation(
        as_type="trace",
        name="wealthpilot-underwriting",
        trace_id=run.trace_id,
        session_id=session_id,
        user_id=user_id,
        input_summary={"session_id": session_id, "user_id": user_id},
    )
    run.langfuse_context = langfuse_context
    run.langfuse_observation = langfuse_observation
    _ACTIVE_TRACE.set(run)
    return run


def end_trace(run: TraceRun) -> dict[str, Any]:
    """Finalize a trace and emit a compact KPI record through structured logging."""
    summary = run.summary()
    get_logger("wealthpilot.trace").info("Trace completed", extra={"trace": summary})
    output_path = os.getenv("TRACE_OUTPUT_PATH")
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, default=str) + "\n")
    if run.langfuse_observation is not None:
        try:
            run.langfuse_observation.update(output=mask_sensitive_data(summary))
        except Exception as exc:  # Langfuse must never break underwriting.
            get_logger("wealthpilot.langfuse").debug("Unable to update Langfuse trace: %s", exc)
    _close_langfuse_observation(run.langfuse_context)
    _ACTIVE_TRACE.set(None)
    _flush_langfuse()
    return summary


def active_trace() -> TraceRun | None:
    """Return the current request trace, if one is active."""
    return _ACTIVE_TRACE.get()


@contextmanager
def trace_span(name: str, *, kind: str = "node", inputs: dict[str, Any] | None = None):
    """Record a nested, secret-safe span around an agent, tool, or LLM call."""
    run = _ACTIVE_TRACE.get()
    if run is None:
        yield None
        return
    parent = run.spans[-1].name if run.spans else None
    span = TraceSpan(name=name, kind=kind, started_at=time.perf_counter(), parent=parent)
    span.input_summary = mask_sensitive_data(inputs or {})
    run.spans.append(span)
    span.langfuse_context, span.langfuse_observation = _start_langfuse_observation(
        as_type="generation" if kind == "generation" else "span",
        name=name,
        input_summary=span.input_summary,
    )
    try:
        yield span
    except Exception as exc:
        span.status = "error"
        span.error = str(exc)
        run.error_count += 1
        if span.langfuse_observation is not None:
            try:
                span.langfuse_observation.update(
                    level="ERROR", status_message=mask_sensitive_text(str(exc))
                )
            except Exception:
                pass
        raise
    finally:
        span.ended_at = time.perf_counter()
        if span.langfuse_observation is not None:
            try:
                span.langfuse_observation.update(output=mask_sensitive_data(span.output_summary))
            except Exception:
                pass
        _close_langfuse_observation(span.langfuse_context)


def annotate_trace(*, output: dict[str, Any] | None = None, fallback: bool = False) -> None:
    """Attach output/fallback facts to the current span."""
    run = _ACTIVE_TRACE.get()
    if run is None or not run.spans:
        return
    run.spans[-1].output_summary = mask_sensitive_data(output or {})
    if run.spans[-1].langfuse_observation is not None:
        try:
            run.spans[-1].langfuse_observation.update(
                output=mask_sensitive_data(run.spans[-1].output_summary)
            )
        except Exception:
            pass
    if fallback:
        run.fallback_count += 1


def _langfuse_client() -> Any | None:
    """Return a configured Langfuse client, or None when tracing is not configured."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import get_client

        return get_client()
    except Exception as exc:  # Optional telemetry cannot affect core execution.
        get_logger("wealthpilot.langfuse").debug("Langfuse unavailable: %s", exc)
        return None


def _start_langfuse_observation(
    *,
    as_type: str,
    name: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    input_summary: dict[str, Any] | None = None,
) -> tuple[Any | None, Any | None]:
    """Open a Langfuse observation using the workbook's current SDK API."""
    client = _langfuse_client()
    if client is None:
        return None, None
    try:
        kwargs: dict[str, Any] = {
            "as_type": as_type,
            "name": name,
            "input": mask_sensitive_data(input_summary or {}),
        }
        if trace_id:
            kwargs["trace_id"] = trace_id
        if session_id:
            kwargs["session_id"] = mask_sensitive_text(session_id)
        if user_id:
            kwargs["user_id"] = mask_sensitive_text(user_id)
        context = client.start_as_current_observation(**kwargs)
        return context, context.__enter__()
    except Exception as exc:
        get_logger("wealthpilot.langfuse").debug("Unable to start Langfuse observation: %s", exc)
        return None, None


def _close_langfuse_observation(context: Any | None) -> None:
    if context is None:
        return
    try:
        context.__exit__(None, None, None)
    except Exception as exc:
        get_logger("wealthpilot.langfuse").debug("Unable to close Langfuse observation: %s", exc)


def _flush_langfuse() -> None:
    client = _langfuse_client()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        get_logger("wealthpilot.langfuse").debug("Unable to flush Langfuse events: %s", exc)

_RESERVED_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)

_SENSITIVE_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "secret",
        "secret_key",
        "client_secret",
        "password",
        "passwd",
        "private_key",
        "pan",
        "pan_number",
        "aadhaar",
        "aadhaar_number",
        "proprietor_name",
        "applicant_name",
        "phone",
        "mobile",
        "email",
        "account_number",
        "bank_account",
    }
)

_TEXT_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Google / Gemini / GCP style API keys
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"), "[MASKED_API_KEY]"),
    # OpenAI / Anthropic / Stripe style secret keys
    (re.compile(r"\bsk-(?:proj-|live-|test-)?[A-Za-z0-9\-_]{16,}\b"), "[MASKED_API_KEY]"),
    # GitHub tokens
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b"), "[MASKED_TOKEN]"),
    # Bearer tokens
    (re.compile(r"(?i)\bBearer\s+(?!Bearer\b)[A-Za-z0-9\-._~+/]+=*"), "Bearer [MASKED_TOKEN]"),
    # Key=value or JSON-style credential assignments
    (
        re.compile(
            r"(?i)(['\"]?(?:api[_-]?key|secret|token|password|authorization|client_secret)['\"]?\s*[:=]\s*['\"]?)([^\s,'\"\}]{6,})(['\"]?)"
        ),
        r"\1[REDACTED]\3",
    ),
    # Indian PAN (5 letters, 4 digits, 1 letter)
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[MASKED_PAN]"),
    # Indian Aadhaar (12 digits starting 2-9, optional spaces/hyphens)
    (re.compile(r"\b[2-9][0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b"), "[MASKED_AADHAAR]"),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[MASKED_EMAIL]"),
    # Indian mobile numbers (+91 optional, 10 digits starting 6-9)
    (re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9][0-9]{9}(?!\d)"), "[MASKED_PHONE]"),
)

_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
_NON_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 422})


class TransientError(RuntimeError):
    """Error indicating a temporary failure eligible for retry and backoff."""


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit breaker is OPEN."""


class CircuitState(str, Enum):
    """Lifecycle states of a circuit breaker."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


def mask_sensitive_text(text: str) -> str:
    """Redact API keys, tokens, and PII patterns from arbitrary log text."""
    masked = text
    for pattern, replacement in _TEXT_REDACTION_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked


def mask_sensitive_data(value: Any) -> Any:
    """Recursively mask sensitive dictionary keys and string patterns."""
    if isinstance(value, str):
        return mask_sensitive_text(value)
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for k, v in value.items():
            key_lower = str(k).strip().lower()
            if key_lower in _SENSITIVE_KEY_NAMES:
                sanitized[k] = "[REDACTED]"
            else:
                sanitized[k] = mask_sensitive_data(v)
        return sanitized
    if isinstance(value, list):
        return [mask_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_sensitive_data(item) for item in value)
    return value


class PIIMaskingFilter(logging.Filter):
    """Logging filter that scrubs PII and secrets from messages and extra fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered_message = record.getMessage()
            record.msg = mask_sensitive_text(rendered_message)
            record.args = ()
        except Exception:
            record.msg = mask_sensitive_text(str(record.msg))
            record.args = ()

        for attr_name in list(vars(record).keys()):
            if attr_name in _RESERVED_LOG_RECORD_ATTRS:
                continue
            attr_val = getattr(record, attr_name)
            if attr_name.strip().lower() in _SENSITIVE_KEY_NAMES:
                setattr(record, attr_name, "[REDACTED]")
            else:
                setattr(record, attr_name, mask_sensitive_data(attr_val))
        return True


class StructuredJsonFormatter(logging.Formatter):
    """JSON-line formatter suitable for Linux log aggregators and stderr MCP logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_text(record.getMessage()),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        extra_fields: dict[str, Any] = {}
        for key, val in vars(record).items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            extra_fields[key] = mask_sensitive_data(val)
        if extra_fields:
            payload["extra"] = extra_fields

        if record.exc_info:
            payload["exception"] = mask_sensitive_text(self.formatException(record.exc_info))

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: int | str = logging.INFO,
    stream: TextIO | None = None,
    json_format: bool = True,
    force: bool = True,
) -> logging.Logger:
    """Configure root logging with PII/API-key masking and strict stderr output by default."""
    target_stream: TextIO = sys.stderr if stream is None else stream
    resolved_level: int = (
        getattr(logging, level.upper(), logging.INFO) if isinstance(level, str) else level
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)

    if force:
        for existing in list(root_logger.handlers):
            root_logger.removeHandler(existing)

    if not root_logger.handlers:
        handler = logging.StreamHandler(stream=target_stream)
        handler.setLevel(resolved_level)
        handler.addFilter(PIIMaskingFilter())
        if json_format:
            handler.setFormatter(StructuredJsonFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
                )
            )
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.setStream(target_stream)
            if not any(isinstance(f, PIIMaskingFilter) for f in handler.filters):
                handler.addFilter(PIIMaskingFilter())

    return logging.getLogger("wealthpilot")


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring PII masking is active if root has no handlers."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging(stream=sys.stderr, force=False)
    return logging.getLogger(name)


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker supporting CLOSED, OPEN, and HALF_OPEN states."""

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)
    _half_open_calls: int = field(default=0, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be > 0")
        if self.half_open_max_calls < 1:
            raise ValueError("half_open_max_calls must be >= 1")
        self._logger = get_logger(f"wealthpilot.circuit_breaker.{self.name}")

    @property
    def state(self) -> CircuitState:
        """Current circuit state, promoting OPEN -> HALF_OPEN after recovery_timeout."""
        with self._lock:
            if (
                self._state is CircuitState.OPEN
                and self._opened_at is not None
                and (time.monotonic() - self._opened_at) >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._logger.info("Circuit breaker transitioned to HALF_OPEN", extra={"breaker": self.name})
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def allow_request(self) -> bool:
        """Check whether an execution attempt is permitted right now."""
        with self._lock:
            current = self.state
            if current is CircuitState.CLOSED:
                return True
            if current is CircuitState.OPEN:
                return False
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False

    def record_success(self) -> None:
        """Record a successful invocation and close the circuit."""
        with self._lock:
            previous = self._state
            self._failure_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED
            if previous is not CircuitState.CLOSED:
                self._logger.info(
                    "Circuit breaker closed after successful probe",
                    extra={"breaker": self.name, "previous_state": previous.value},
                )

    def record_failure(self) -> None:
        """Record a transient/service failure and open the circuit when threshold is reached."""
        with self._lock:
            current = self.state
            self._failure_count += 1
            if current is CircuitState.HALF_OPEN or self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_calls = 0
                self._logger.warning(
                    "Circuit breaker opened",
                    extra={
                        "breaker": self.name,
                        "failure_count": self._failure_count,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._opened_at = None
            self._half_open_calls = 0


def is_transient_error(
    exc: BaseException,
    transient_exceptions: tuple[type[BaseException], ...],
    non_transient_exceptions: tuple[type[BaseException], ...],
) -> bool:
    """Determine if an exception represents a transient failure safe to retry."""
    status_code = getattr(exc, "status_code", getattr(exc, "code", None))
    if isinstance(status_code, int):
        if status_code in _NON_TRANSIENT_HTTP_CODES:
            return False
        if status_code in _TRANSIENT_HTTP_CODES:
            return True

    # Evaluate non-transient exceptions before broad bases like OSError (e.g., PermissionError).
    if isinstance(exc, non_transient_exceptions):
        return False

    if isinstance(exc, (TimeoutError, FuturesTimeoutError, asyncio.TimeoutError)):
        return True

    return isinstance(exc, transient_exceptions)


def _compute_backoff_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    backoff_factor: float,
    jitter: bool,
) -> float:
    delay = min(max_delay, base_delay * (backoff_factor**attempt))
    if jitter:
        delay *= random.uniform(0.5, 1.0)
    return max(0.0, delay)


def _resolve_fallback(fallback: Callable[..., T] | T | None, args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
    if callable(fallback):
        return fallback(*args, **kwargs)
    return fallback  # type: ignore[return-value]


def resilient_call(
    *,
    timeout_seconds: float = 15.0,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    transient_exceptions: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
        TransientError,
    ),
    non_transient_exceptions: tuple[type[BaseException], ...] = (
        ValueError,
        TypeError,
        KeyError,
        PermissionError,
    ),
    circuit_breaker: CircuitBreaker | None = None,
    fallback: Callable[..., T] | T | None = None,
) -> Callable[[Callable[P, T]], Callable[P, tuple[T, bool]]]:
    """Decorator adding timeouts, transient-only exponential backoff, and circuit breaking.

    Decorated functions return ``(result, used_fallback)``.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    def decorator(func: Callable[P, T]) -> Callable[P, tuple[T, bool]]:
        logger = get_logger(f"wealthpilot.resilience.{func.__module__}.{func.__qualname__}")

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, bool]:
                if circuit_breaker is not None and not circuit_breaker.allow_request():
                    logger.warning("Call short-circuited by open circuit breaker", extra={"func": func.__qualname__})
                    if fallback is not None:
                        fb_val = _resolve_fallback(fallback, args, kwargs)
                        if inspect.isawaitable(fb_val):
                            fb_val = await fb_val
                        return fb_val, True
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{circuit_breaker.name}' is OPEN for {func.__qualname__}"
                    )

                last_exc: BaseException | None = None
                for attempt in range(max_retries + 1):
                    try:
                        result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                        if circuit_breaker is not None:
                            circuit_breaker.record_success()
                        return result, False
                    except BaseException as exc:
                        last_exc = exc
                        transient = is_transient_error(exc, transient_exceptions, non_transient_exceptions)
                        if not transient:
                            logger.error(
                                "Non-transient error encountered; skipping retries",
                                extra={"func": func.__qualname__, "error": str(exc)},
                            )
                            raise

                        if circuit_breaker is not None:
                            circuit_breaker.record_failure()

                        if attempt < max_retries and (circuit_breaker is None or circuit_breaker.state is not CircuitState.OPEN):
                            delay = _compute_backoff_delay(attempt, base_delay, max_delay, backoff_factor, jitter)
                            logger.warning(
                                "Transient error; retrying with exponential backoff",
                                extra={
                                    "func": func.__qualname__,
                                    "attempt": attempt + 1,
                                    "max_retries": max_retries,
                                    "delay_seconds": round(delay, 3),
                                    "error": str(exc),
                                },
                            )
                            await asyncio.sleep(delay)
                            continue
                        break

                if fallback is not None:
                    logger.warning(
                        "Exhausted retries; returning fallback response",
                        extra={"func": func.__qualname__, "error": str(last_exc)},
                    )
                    fb_val = _resolve_fallback(fallback, args, kwargs)
                    if inspect.isawaitable(fb_val):
                        fb_val = await fb_val
                    return fb_val, True

                assert last_exc is not None
                raise last_exc

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, bool]:
            if circuit_breaker is not None and not circuit_breaker.allow_request():
                logger.warning("Call short-circuited by open circuit breaker", extra={"func": func.__qualname__})
                if fallback is not None:
                    return _resolve_fallback(fallback, args, kwargs), True
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{circuit_breaker.name}' is OPEN for {func.__qualname__}"
                )

            last_exc: BaseException | None = None
            for attempt in range(max_retries + 1):
                executor = ThreadPoolExecutor(max_workers=1)
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timeout=timeout_seconds)
                    if circuit_breaker is not None:
                        circuit_breaker.record_success()
                    return result, False
                except BaseException as exc:
                    if isinstance(exc, FuturesTimeoutError):
                        exc = TimeoutError(
                            f"{func.__qualname__} timed out after {timeout_seconds:.2f}s"
                        )
                    last_exc = exc
                    transient = is_transient_error(exc, transient_exceptions, non_transient_exceptions)
                    if not transient:
                        logger.error(
                            "Non-transient error encountered; skipping retries",
                            extra={"func": func.__qualname__, "error": str(exc)},
                        )
                        raise

                    if circuit_breaker is not None:
                        circuit_breaker.record_failure()

                    if attempt < max_retries and (circuit_breaker is None or circuit_breaker.state is not CircuitState.OPEN):
                        delay = _compute_backoff_delay(attempt, base_delay, max_delay, backoff_factor, jitter)
                        logger.warning(
                            "Transient error; retrying with exponential backoff",
                            extra={
                                "func": func.__qualname__,
                                "attempt": attempt + 1,
                                "max_retries": max_retries,
                                "delay_seconds": round(delay, 3),
                                "error": str(exc),
                            },
                        )
                        time.sleep(delay)
                        continue
                    break
                finally:
                    executor.shutdown(wait=False, cancel_futures=True)

            if fallback is not None:
                logger.warning(
                    "Exhausted retries; returning fallback response",
                    extra={"func": func.__qualname__, "error": str(last_exc)},
                )
                return _resolve_fallback(fallback, args, kwargs), True

            assert last_exc is not None
            raise last_exc

        return sync_wrapper

    return decorator
