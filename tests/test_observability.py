"""Unit tests for src/observability.py."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import time
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.observability import (  # noqa: E402
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    TransientError,
    configure_logging,
    get_logger,
    mask_sensitive_data,
    mask_sensitive_text,
    resilient_call,
)


class _HttpError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class TestObservabilityLogging(unittest.TestCase):
    def test_mask_sensitive_text_redacts_secrets_and_indian_pii(self) -> None:
        fake_google_key = "AIza" + "0123456789abcdefGHIJKL"
        fake_sk_key = "sk-proj-" + "a1b2c3d4e5f6g7h8i9j0"
        fake_gh_token = "ghp_" + "AbCdEf1234567890XyZ"
        sample = (
            f"key={fake_google_key} Authorization: Bearer abcdef1234567890 "
            f"sk={fake_sk_key} gh={fake_gh_token} "
            "PAN=ABCDE1234F Aadhaar=2345 6789 0123 "
            "phone=+91 9876543210 email=founder@wealthpilot.in"
        )
        masked = mask_sensitive_text(sample)

        self.assertNotIn(fake_google_key, masked)
        self.assertNotIn(fake_sk_key, masked)
        self.assertNotIn(fake_gh_token, masked)
        self.assertNotIn("abcdef1234567890", masked)
        self.assertNotIn("ABCDE1234F", masked)
        self.assertNotIn("2345 6789 0123", masked)
        self.assertNotIn("9876543210", masked)
        self.assertNotIn("founder@wealthpilot.in", masked)
        self.assertIn("[MASKED_PAN]", masked)
        self.assertIn("[MASKED_AADHAAR]", masked)
        self.assertIn("[MASKED_PHONE]", masked)
        self.assertIn("[MASKED_EMAIL]", masked)

    def test_mask_sensitive_data_nested_structures(self) -> None:
        payload = {
            "application_id": "WP-101",
            "api_key": "secret-value-12345",
            "applicant": {
                "proprietor_name": "Rohan Mehta",
                "notes": ["Contact 9876543210", {"pan": "ABCDE1234F"}],
            },
            "tags": ("clean", "founder@sme.co.in"),
        }
        masked = mask_sensitive_data(payload)

        self.assertEqual(masked["application_id"], "WP-101")
        self.assertEqual(masked["api_key"], "[REDACTED]")
        self.assertEqual(masked["applicant"]["proprietor_name"], "[REDACTED]")
        self.assertEqual(masked["applicant"]["notes"][0], "Contact [MASKED_PHONE]")
        self.assertEqual(masked["applicant"]["notes"][1]["pan"], "[REDACTED]")
        self.assertEqual(masked["tags"], ("clean", "[MASKED_EMAIL]"))

    def test_configure_logging_writes_only_to_stderr_and_masks_json(self) -> None:
        stderr_buf = io.StringIO()
        stdout_buf = io.StringIO()

        with contextlib.redirect_stdout(stdout_buf):
            configure_logging(level=logging.INFO, stream=stderr_buf, json_format=True, force=True)
            log = get_logger("tests.observability.logging")
            log.info(
                "Processing PAN ABCDE1234F for %s",
                "owner@sme.in",
                extra={"api_key": "sk-test-1234567890abcdef", "loan_id": "LN-42"},
            )

        self.assertEqual(stdout_buf.getvalue(), "")
        lines = [line for line in stderr_buf.getvalue().splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 1)
        record = json.loads(lines[-1])
        self.assertEqual(record["level"], "INFO")
        self.assertNotIn("ABCDE1234F", record["message"])
        self.assertNotIn("owner@sme.in", record["message"])
        self.assertEqual(record["extra"]["api_key"], "[REDACTED]")
        self.assertEqual(record["extra"]["loan_id"], "LN-42")


class TestCircuitBreaker(unittest.TestCase):
    def test_circuit_breaker_state_transitions(self) -> None:
        cb = CircuitBreaker(name="unit_test", failure_threshold=2, recovery_timeout=0.05, half_open_max_calls=1)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())

        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertFalse(cb.allow_request())

        time.sleep(0.06)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())
        self.assertFalse(cb.allow_request())

        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

        time.sleep(0.06)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)
        self.assertTrue(cb.allow_request())
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.failure_count, 0)

    def test_circuit_breaker_invalid_parameters(self) -> None:
        with self.assertRaises(ValueError):
            CircuitBreaker(failure_threshold=0)
        with self.assertRaises(ValueError):
            CircuitBreaker(recovery_timeout=0.0)
        with self.assertRaises(ValueError):
            CircuitBreaker(half_open_max_calls=0)


class TestResilientCallSync(unittest.TestCase):
    def test_retries_transient_errors_and_succeeds(self) -> None:
        calls = {"count": 0}

        @resilient_call(timeout_seconds=1.0, max_retries=2, base_delay=0.005, jitter=False)
        def flaky_service() -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise TransientError("upstream busy")
            return "healthy"

        result, used_fallback = flaky_service()
        self.assertEqual(result, "healthy")
        self.assertFalse(used_fallback)
        self.assertEqual(calls["count"], 3)

    def test_non_transient_errors_do_not_retry(self) -> None:
        for exc_cls in (ValueError, PermissionError):
            calls = {"count": 0}

            @resilient_call(timeout_seconds=1.0, max_retries=3, base_delay=0.005)
            def fail_fast() -> str:
                calls["count"] += 1
                raise exc_cls("non-retryable")

            with self.assertRaises(exc_cls):
                fail_fast()
            self.assertEqual(calls["count"], 1)

    def test_http_status_code_classification(self) -> None:
        transient_calls = {"count": 0}

        @resilient_call(timeout_seconds=1.0, max_retries=1, base_delay=0.005, jitter=False)
        def rate_limited_then_ok() -> str:
            transient_calls["count"] += 1
            if transient_calls["count"] == 1:
                raise _HttpError("rate limited", status_code=429)
            return "done"

        result, used_fallback = rate_limited_then_ok()
        self.assertEqual(result, "done")
        self.assertFalse(used_fallback)
        self.assertEqual(transient_calls["count"], 2)

        client_calls = {"count": 0}

        @resilient_call(timeout_seconds=1.0, max_retries=2, base_delay=0.005)
        def bad_request() -> str:
            client_calls["count"] += 1
            raise _HttpError("bad request", status_code=400)

        with self.assertRaises(_HttpError):
            bad_request()
        self.assertEqual(client_calls["count"], 1)

    def test_timeout_triggers_fallback(self) -> None:
        @resilient_call(timeout_seconds=0.04, max_retries=1, base_delay=0.005, fallback={"status": "degraded"})
        def slow_operation() -> dict[str, str]:
            time.sleep(0.2)
            return {"status": "ok"}

        result, used_fallback = slow_operation()
        self.assertEqual(result, {"status": "degraded"})
        self.assertTrue(used_fallback)

    def test_open_circuit_short_circuits_with_fallback_or_error(self) -> None:
        cb = CircuitBreaker(name="short_circuit", failure_threshold=1, recovery_timeout=10.0)
        cb.record_failure()

        @resilient_call(
            timeout_seconds=1.0,
            max_retries=2,
            circuit_breaker=cb,
            fallback=lambda x: f"fallback:{x}",
        )
        def with_fallback(val: str) -> str:
            return val

        res, used_fb = with_fallback("sme")
        self.assertEqual(res, "fallback:sme")
        self.assertTrue(used_fb)

        @resilient_call(timeout_seconds=1.0, max_retries=2, circuit_breaker=cb)
        def without_fallback() -> str:
            return "ok"

        with self.assertRaises(CircuitBreakerOpenError):
            without_fallback()


class TestResilientCallAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_retries_and_timeout_fallback(self) -> None:
        attempts = {"count": 0}

        @resilient_call(timeout_seconds=0.5, max_retries=2, base_delay=0.005, jitter=False)
        async def async_flaky() -> int:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("reset")
            return 99

        val, used_fb = await async_flaky()
        self.assertEqual(val, 99)
        self.assertFalse(used_fb)
        self.assertEqual(attempts["count"], 2)

        @resilient_call(timeout_seconds=0.03, max_retries=0, fallback=-1)
        async def async_slow() -> int:
            import asyncio

            await asyncio.sleep(0.2)
            return 1

        fb_val, used_fb_slow = await async_slow()
        self.assertEqual(fb_val, -1)
        self.assertTrue(used_fb_slow)


if __name__ == "__main__":
    unittest.main()

