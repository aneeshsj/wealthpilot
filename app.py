"""FastAPI service entry point for the WealthPilot SME loan underwriting platform."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

try:
    from src import schemas
    from src.config import configure_logging, settings
    from src.guardrails import scan_for_injection
except ImportError:  # pragma: no cover - supports running with src/ on sys.path
    import schemas  # type: ignore[no-redef]
    from config import configure_logging, settings  # type: ignore[no-redef]
    from guardrails import scan_for_injection  # type: ignore[no-redef]

configure_logging()
logger = logging.getLogger(__name__)


def _scan_injection(text: str) -> tuple[bool, list[str]]:
    """Normalize the deterministic scanner result for API responses."""
    detected, rules = scan_for_injection(text)
    return bool(detected), list(rules)

app = FastAPI(
    title="WealthPilot SME Underwriting API",
    version="1.0.0",
    description="Multi-agent SME loan underwriting and grounded policy RAG API.",
)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class UnderwriteRequest(BaseModel):
    """Input payload for initiating an underwriting workflow run."""

    thread_id: str | None = Field(default=None, min_length=1)
    application: dict[str, Any] | str


class UnderwriteResumeRequest(BaseModel):
    """Input payload for resuming a human-in-the-loop paused underwriting thread."""

    thread_id: str = Field(min_length=1)
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1)
    notes: str = Field(default="")
    approved_amount_inr: float | None = Field(default=None, ge=0.0)


class PolicyQueryRequest(BaseModel):
    """Input payload for querying the WealthPilot credit policy RAG assistant."""

    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_interrupt_payload(state: dict[str, Any]) -> Any:
    """Extract serializable interrupt payload from LangGraph invocation state."""
    raw_interrupts = state.get("__interrupt__") or []
    if not raw_interrupts:
        return None
    first = raw_interrupts[0]
    if hasattr(first, "value"):
        return first.value
    if isinstance(first, dict) and "value" in first:
        return first["value"]
    return first


def _validate_and_scan_application(
    raw_application: dict[str, Any] | str,
) -> tuple[dict[str, Any], list[str]]:
    """Run prompt-injection scanning and Pydantic schema validation on intake payload."""
    flags: list[str] = []
    raw_text = (
        raw_application
        if isinstance(raw_application, str)
        else json.dumps(raw_application, sort_keys=True, default=str)
    )

    injection_detected, injection_rules = _scan_injection(raw_text)
    if injection_detected:
        flags.append("prompt_injection_detected")
        flags.extend(rule for rule in injection_rules if rule not in flags)

    parsed_dict: dict[str, Any] = {}
    try:
        loaded = (
            json.loads(raw_application)
            if isinstance(raw_application, str)
            else dict(raw_application)
        )
        if not isinstance(loaded, dict):
            raise TypeError("Application payload must decode to a JSON object.")
        validated = schemas.LoanApplication.model_validate(loaded)
        parsed_dict = validated.model_dump(mode="json")
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        logger.warning("Application validation failed: %s", exc)
        flags.append("malformed_application_payload")

    return parsed_dict, flags


def _run_underwriting_graph(
    *,
    application_id: str,
    validated_app: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the blocking LangGraph workflow outside FastAPI's event loop."""
    from src.graph import compiled_underwriting_graph
    from src.observability import end_trace, start_trace

    trace = start_trace(config["configurable"]["thread_id"], application_id)
    try:
        with compiled_underwriting_graph() as graph:
            result = graph.invoke(
                {
                    "application_id": application_id,
                    "raw_input": validated_app,
                    "guardrail_flags": [],
                    "audit_trail": [],
                },
                config=config,
            )
        return result
    finally:
        end_trace(trace)


def _resume_underwriting_graph(
    resume_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Resume a blocking LangGraph checkpoint outside FastAPI's event loop."""
    from src.graph import compiled_underwriting_graph
    from src.observability import end_trace, start_trace

    thread_id = str(config["configurable"]["thread_id"])
    trace = start_trace(thread_id, thread_id)
    try:
        with compiled_underwriting_graph() as graph:
            return graph.invoke(Command(resume=resume_payload), config=config)
    finally:
        end_trace(trace)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health and active configured LLM model."""
    return {
        "status": "ok",
        "model": str(settings.LLM_MODEL),
    }


@app.post("/underwrite")
async def underwrite(payload: UnderwriteRequest) -> dict[str, Any]:
    """Validate intake payload and execute the WealthPilot underwriting graph."""
    validated_app, flags = _validate_and_scan_application(payload.application)
    if flags:
        logger.warning("Blocked underwriting request due to flags=%s", flags)
        raise HTTPException(
            status_code=422,
            detail={"blocked": True, "flags": flags},
        )

    resolved_thread_id = payload.thread_id or f"wp-{uuid.uuid4().hex[:12]}"
    application_id = str(
        validated_app.get("application_id")
        or validated_app.get("app_id")
        or resolved_thread_id
    )
    config = {"configurable": {"thread_id": resolved_thread_id}}

    result = await asyncio.to_thread(
        _run_underwriting_graph,
        application_id=application_id,
        validated_app=validated_app,
        config=config,
    )

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        logger.info(
            "Underwriting paused for human signoff thread_id=%s application_id=%s",
            resolved_thread_id,
            application_id,
        )
        return {
            "status": "paused_for_human",
            "thread_id": resolved_thread_id,
            "interrupt_payload": interrupt_payload,
        }

    return {
        "status": "completed",
        "thread_id": resolved_thread_id,
        "application_id": application_id,
        "final_decision": result.get("final_decision", {}),
        "dscr_metrics": result.get("dscr_metrics", {}),
        "bureau_data": result.get("bureau_data", {}),
        "bank_data": result.get("bank_data", {}),
        "policy_citations": result.get("policy_citations", []),
        "guardrail_flags": result.get("guardrail_flags", []),
        "used_fallback": bool(result.get("used_fallback", False)),
        "audit_trail": result.get("audit_trail", []),
    }


@app.post("/underwrite/resume")
async def underwrite_resume(payload: UnderwriteResumeRequest) -> dict[str, Any]:
    """Resume a paused underwriting checkpoint with a human credit officer decision."""
    resume_payload: dict[str, Any] = {
        "approved": payload.decision == "approve",
        "override_decision": payload.decision,
        "reviewer": payload.reviewer,
        "notes": payload.notes,
    }
    if payload.approved_amount_inr is not None:
        resume_payload["approved_amount_inr"] = payload.approved_amount_inr

    config = {"configurable": {"thread_id": payload.thread_id}}
    try:
        result = await asyncio.to_thread(_resume_underwriting_graph, resume_payload, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed resuming thread_id=%s: %s", payload.thread_id, exc)
        raise HTTPException(
            status_code=400,
            detail=f"Unable to resume thread '{payload.thread_id}': {exc}",
        ) from exc

    interrupt_payload = _extract_interrupt_payload(result)
    if interrupt_payload is not None:
        return {
            "status": "paused_for_human",
            "thread_id": payload.thread_id,
            "interrupt_payload": interrupt_payload,
        }

    return {
        "status": "completed",
        "thread_id": payload.thread_id,
        "application_id": result.get("application_id"),
        "final_decision": result.get("final_decision", {}),
        "human_approval": result.get("human_approval", {}),
        "audit_trail": result.get("audit_trail", []),
    }


@app.post("/policy/query")
async def policy_query(payload: PolicyQueryRequest) -> dict[str, Any]:
    """Answer an underwriting policy question grounded in the Qdrant hybrid RAG index."""
    from src.rag import ABSTENTION_RESPONSE, answer_policy_question

    try:
        answer = await asyncio.to_thread(
            answer_policy_question,
            question=payload.question,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        logger.warning("RAG policy query failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "answer": answer.answer,
        "abstained": answer.answer == ABSTENTION_RESPONSE,
        "citations": answer.citations,
        "model": answer.model,
        "retrieved_chunks": [chunk.to_dict() for chunk in answer.retrieved_chunks],
    }

