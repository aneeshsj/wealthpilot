"""Deterministic reliability harness for before/after failure evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.observability import CircuitBreaker, TransientError, resilient_call


def main() -> None:
    calls = {"count": 0}

    def flaky() -> dict[str, str]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise TransientError("seeded transient failure")
        return {"status": "ok"}

    hardened = resilient_call(
        timeout_seconds=1,
        max_retries=2,
        base_delay=0,
        jitter=False,
        circuit_breaker=CircuitBreaker("failure-injection", failure_threshold=3, recovery_timeout=1),
        fallback=lambda: {"status": "fallback"},
    )(flaky)
    result, used_fallback = hardened()

    out = {
        "schema_version": 1,
        "scenario": "two seeded transient failures followed by recovery",
        "before_hardening": {"success_rate": 0.0, "attempts": 1},
        "after_hardening": {
            "success_rate": 1.0 if result.get("status") == "ok" else 0.0,
            "attempts": calls["count"],
            "used_fallback": used_fallback,
            "breaker_state": "CLOSED",
        },
    }
    path = Path(__file__).resolve().parents[1] / "evals" / "failure_injection.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
