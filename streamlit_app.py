"""Streamlit workspace for the WealthPilot SME underwriting workflow."""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command

from src import schemas
from src.config import configure_logging, settings
from src.guardrails import sanitize_for_bias, scan_for_injection

configure_logging()
logger = logging.getLogger(__name__)

DEFAULT_RECORDS_PATH = Path(settings.DATA_DIR) / "intake" / "records.jsonl"
DEFAULT_EVALS_DIR = Path("evals")


def _file_signature(path: Path) -> tuple[str, int, int]:
    """Return a cache key that changes when a local artifact changes."""
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            return (str(path.resolve()), 0, 0)
        newest_mtime = max(file.stat().st_mtime_ns for file in files)
        total_size = sum(file.stat().st_size for file in files)
        return (str(path.resolve()), newest_mtime, total_size)
    try:
        stat = path.stat()
    except OSError:
        return (str(path), 0, 0)
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def _load_sample_applications(path_key: tuple[str, int, int]) -> list[dict[str, Any]]:
    """Load sample applications once per file version."""
    records_path = Path(path_key[0])
    if not records_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSONL line %d in %s: %s", line_num, records_path, exc)
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    return records


@st.cache_data(show_spinner=False)
def _load_evaluation_artifacts(path_key: tuple[str, int, int]) -> list[tuple[str, Any]]:
    """Read committed JSON evaluation artifacts from the project evals folder."""
    evals_dir = Path(path_key[0])
    if not evals_dir.is_dir():
        return []
    artifacts: list[tuple[str, Any]] = []
    for path in sorted(evals_dir.glob("*.json")):
        try:
            artifacts.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed parsing evaluation artifact %s: %s", path, exc)
    return artifacts


def _format_sample_label(idx: int, record: dict[str, Any]) -> str:
    payload = record.get("application") if isinstance(record.get("application"), dict) else record
    app_id = str(payload.get("application_id") or payload.get("id") or f"REC-{idx + 1}")
    scenario = str(record.get("scenario") or record.get("category") or payload.get("sector") or "sample")
    amount = payload.get("requested_amount_inr")
    amount_text = f" · INR {float(amount):,.0f}" if isinstance(amount, (int, float)) else ""
    return f"{app_id} · {scenario}{amount_text}"


def _extract_application_dict(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("application")
    return dict(payload) if isinstance(payload, dict) else dict(record)


def _compute_masked_fields(raw_app: dict[str, Any], sanitized_app: dict[str, Any]) -> list[str]:
    return [
        f"{key} ({'removed' if key not in sanitized_app else 'redacted'})"
        for key, value in raw_app.items()
        if key not in sanitized_app or sanitized_app[key] != value
    ]


def _extract_interrupt_payload(state: dict[str, Any]) -> Any:
    raw_interrupts = state.get("__interrupt__") or []
    if not raw_interrupts:
        return None
    first = raw_interrupts[0]
    if hasattr(first, "value"):
        return first.value
    return first.get("value", first) if isinstance(first, dict) else first


def _format_pct_or_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if 0 <= value <= 1 else f"{value:.2f}"


def _find_metric_recursive(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        lowered = {str(k).lower(): value for k, value in payload.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in payload.values():
            found = _find_metric_recursive(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_metric_recursive(value, keys)
            if found is not None:
                return found
    return None


def _find_component_metric(
    payloads: list[Any], component: str, key: str
) -> float | None:
    """Read a metric from the evaluator's component payload."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        value = (payload.get("components") or {}).get(component, {}).get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _model_for_provider(provider: str | None) -> str:
    """Resolve the model shown to reviewers for a configured provider."""
    normalized = (provider or "").strip().lower()
    if normalized == "gemini":
        return str(settings.GEMINI_MODEL)
    if normalized == "groq":
        return str(settings.GROQ_API_MODEL)
    return str(settings.LLM_MODEL)


def _route_label(flags: list[str], stage: str) -> str:
    """Describe the actual route used by one committee stage."""
    if f"{stage}_llm_fallback" in flags:
        return "Deterministic safety fallback (no LLM response)"
    if f"{stage}_provider_fallback" in flags:
        provider = str(settings.LLM_FALLBACK_PROVIDER or "configured fallback").lower()
        return f"Provider fallback · {provider} · {_model_for_provider(provider)}"
    provider = str(settings.LLM_PROVIDER or "configured").lower()
    return f"Primary · {provider} · {_model_for_provider(provider)}"


def _render_sidebar(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    st.sidebar.header("Workspace")
    primary_provider = str(settings.LLM_PROVIDER or "configured").lower()
    st.sidebar.caption(f"Primary model: {_model_for_provider(primary_provider)}")
    if settings.LLM_FALLBACK_ENABLED and settings.LLM_FALLBACK_PROVIDER:
        fallback_provider = str(settings.LLM_FALLBACK_PROVIDER).lower()
        st.sidebar.caption(
            f"Fallback model: {_model_for_provider(fallback_provider)} ({fallback_provider})"
        )
    else:
        st.sidebar.caption("Fallback model: disabled")
    st.sidebar.caption("Generated cases and policy documents are synthetic.")

    selected_record: dict[str, Any] | None = None
    if records:
        labels = [_format_sample_label(i, record) for i, record in enumerate(records)]
        selected_idx = st.sidebar.selectbox(
            "Sample application",
            range(len(records)),
            format_func=lambda idx: labels[idx],
            help="Choose a generated case to load into the underwriting form.",
        )
        selected_record = records[selected_idx]
        if st.sidebar.button("Load selected case", use_container_width=True):
            st.session_state["raw_app_json"] = json.dumps(_extract_application_dict(selected_record), indent=2)
            st.session_state.pop("underwriting_result", None)
            st.session_state.pop("underwriting_thread_id", None)
            st.rerun()
    else:
        st.sidebar.warning(f"No cases found at {DEFAULT_RECORDS_PATH}.")

    with st.sidebar.expander("Policy index", expanded=False):
        st.caption("The index powers policy questions and policy citations in underwriting.")
        if st.button(
            "Build / refresh policy index",
            use_container_width=True,
            help="Rebuild the local Qdrant index from the generated policy documents.",
        ):
            from src.rag import build_index

            with st.spinner("Refreshing policy index…"):
                try:
                    stats = build_index()
                    st.session_state["index_stats"] = stats
                    st.success(f"Indexed {stats.document_count} documents and {stats.chunk_count} chunks.")
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Index rebuild failed: %s", exc)
                    st.error(f"Index refresh failed: {exc}")
        stats = st.session_state.get("index_stats")
        if stats:
            st.caption(f"{stats.chunk_count} chunks · {stats.embedding_backend} embeddings · {stats.collection_name}")

    return _extract_application_dict(selected_record) if selected_record else None


def _render_underwriting(default_sample: dict[str, Any] | None) -> None:
    st.subheader("Underwrite an application")
    st.caption("Validate the intake, calculate risk signals, review policy evidence, and complete any required human sign-off.")

    if "raw_app_json" not in st.session_state:
        initial = default_sample or {
            "application_id": "APP-1001", "applicant_id": "APL-2001", "business_name": "Precision Components Pvt Ltd",
            "sector": "manufacturing", "requested_amount_inr": 2500000.0, "requested_tenor_months": 36,
            "annual_revenue_inr": 45000000.0, "ebitda_inr": 8500000.0, "existing_debt_service_inr": 3200000.0,
        }
        st.session_state["raw_app_json"] = json.dumps(initial, indent=2)

    raw_json = st.text_area(
        "Application details (JSON)",
        key="raw_app_json",
        height=220,
        help="Paste or edit one SME loan application. Validation runs before the committee is called.",
    )
    parsed: dict[str, Any] | None = None
    sanitized: dict[str, Any] | None = None
    validation_error: str | None = None
    injection_detected = False
    try:
        injection_detected, _ = scan_for_injection(raw_json)
        loaded = json.loads(raw_json)
        if not isinstance(loaded, dict):
            raise ValueError("The application must be a JSON object.")
        validated = schemas.LoanApplication.model_validate(loaded)
        parsed = loaded
        sanitized_candidate = sanitize_for_bias(validated.model_dump(mode="json"))
        sanitized = dict(sanitized_candidate) if isinstance(sanitized_candidate, dict) else validated.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        validation_error = str(exc)

    status_col, preview_col = st.columns(2)
    with status_col:
        st.markdown("**Pre-flight checks**")
        if injection_detected:
            st.error("Possible prompt injection found in intake text.")
        else:
            st.success("No prompt injection pattern found.")
        if validation_error:
            st.warning(f"Validation needed: {validation_error}")
        else:
            st.success("Application matches the LoanApplication schema.")
    with preview_col:
        st.markdown("**Fairness protection**")
        if sanitized is not None:
            masked = _compute_masked_fields(parsed or {}, sanitized)
            st.caption(f"Masked before agent reasoning: {', '.join(masked) if masked else 'no proxy fields present'}")
        else:
            st.caption("A sanitized preview appears after the application passes validation.")

    if st.button(
        "Run underwriting committee",
        type="primary",
        disabled=parsed is None,
        help="Runs intake, document checks, deterministic risk tools, policy retrieval, and committee review.",
    ):
        from src.graph import compiled_underwriting_graph

        thread_id = f"ui-{uuid.uuid4().hex[:10]}"
        with st.spinner("Assessing the application…"):
            try:
                with compiled_underwriting_graph() as graph:
                    result = graph.invoke(
                        {"application_id": str(parsed.get("application_id") or thread_id), "raw_input": parsed, "guardrail_flags": [], "audit_trail": []},
                        config={"configurable": {"thread_id": thread_id}},
                    )
                st.session_state["underwriting_thread_id"] = thread_id
                st.session_state["underwriting_result"] = result
            except Exception as exc:  # noqa: BLE001
                logger.exception("Underwriting workflow failed: %s", exc)
                st.error(f"Assessment failed: {exc}")

    result = st.session_state.get("underwriting_result")
    if result:
        _render_underwriting_result(result, st.session_state.get("underwriting_thread_id"))


def _render_underwriting_result(result: dict[str, Any], thread_id: str | None) -> None:
    dscr = result.get("dscr_metrics") or {}
    bureau = result.get("bureau_data") or {}
    bank = result.get("bank_data") or {}
    draft = result.get("draft_verdict") or {}
    final = result.get("final_decision") or {}
    flags = result.get("guardrail_flags") or []
    decision = final.get("decision") or draft.get("decision") or "pending"

    st.divider()
    st.markdown("### Credit assessment")
    assessment_card, decision_card = st.columns(2, gap="large")
    with assessment_card:
        with st.container(border=True):
            st.markdown("#### Credit assessment")
            st.caption("Core financial and credit signals used by the committee.")
            assessment_top = st.columns(2)
            assessment_top[0].metric(
                "DSCR",
                f"{dscr.get('dscr'):.2f}" if dscr.get("dscr") is not None else "—",
                help="Debt service coverage ratio calculated by the deterministic risk tool.",
            )
            assessment_top[1].metric(
                "Bureau score",
                bureau.get("bureau_score", "—"),
                help="Credit-bureau score used as one risk signal.",
            )
            assessment_bottom = st.columns(2)
            assessment_bottom[0].metric(
                "Bounced instruments",
                bank.get("total_bounced_instruments", "—"),
                help="Count from the bank-statement analysis tool.",
            )
            assessment_bottom[1].metric(
                "Policy citations",
                len(result.get("policy_citations") or []),
                help="Number of policy references attached to this assessment.",
            )
    with decision_card:
        with st.container(border=True):
            st.markdown("#### Decision")
            st.caption("Committee recommendation and approval position.")
            decision_top = st.columns(2)
            decision_top[0].metric(
                "Recommendation",
                str(decision).upper(),
                help="Draft until any required human approval is completed.",
            )
            decision_top[1].metric(
                "Risk grade",
                draft.get("risk_grade", "—"),
                help="Risk grade assigned by the analyst or deterministic review path.",
            )
            decision_bottom = st.columns(2)
            amount = draft.get("recommended_amount_inr")
            decision_bottom[0].metric(
                "Recommended amount",
                f"INR {float(amount):,.0f}" if isinstance(amount, (int, float)) else "—",
                help="Amount recommended by the committee.",
            )
            rate = draft.get("proposed_interest_rate_pct")
            decision_bottom[1].metric(
                "Indicative rate",
                f"{float(rate):.2f}%" if isinstance(rate, (int, float)) else "—",
                help="Indicative interest rate in the draft recommendation.",
            )
    analyst_route = _route_label([str(flag) for flag in flags], "analyst")
    critic_route = _route_label([str(flag) for flag in flags], "critic")
    with st.expander("Model execution", expanded=True):
        st.caption("This records which route actually produced each committee result.")
        st.write(f"**Analyst:** {analyst_route}")
        st.write(f"**Critic:** {critic_route}")
    if flags:
        st.warning("Review flags: " + ", ".join(str(flag) for flag in flags))

    interrupt = _extract_interrupt_payload(result)
    if interrupt is not None and thread_id:
        st.warning("Human review required before this decision can be finalized.")
        with st.expander("Why review is required", expanded=True):
            st.json(interrupt)
        reviewer = st.text_input("Reviewer name", value="senior_underwriter", help="Name recorded in the decision audit trail.")
        notes = st.text_area("Review notes", value="Reviewed financials and policy flags.", help="Briefly record the evidence considered by the credit officer.")
        approve, reject = st.columns(2)
        action: dict[str, Any] | None = None
        if approve.button("Approve after review", type="primary", use_container_width=True):
            action = {"approved": True, "override_decision": "approve", "reviewer": reviewer or "credit_officer", "notes": notes}
        if reject.button("Decline after review", use_container_width=True):
            action = {"approved": False, "override_decision": "reject", "reviewer": reviewer or "credit_officer", "notes": notes}
        if action:
            from src.graph import compiled_underwriting_graph

            with st.spinner("Saving the human decision…"):
                try:
                    with compiled_underwriting_graph() as graph:
                        st.session_state["underwriting_result"] = graph.invoke(Command(resume=action), config={"configurable": {"thread_id": thread_id}})
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Checkpoint resume failed: %s", exc)
                    st.error(f"Could not save the review decision: {exc}")
    elif final:
        st.success(f"Decision finalized: {str(final.get('decision', '')).upper()}")

    audit_tab, evidence_tab, details_tab = st.tabs(["Decision trail", "Policy evidence", "Technical details"])
    with audit_tab:
        for entry in result.get("audit_trail") or []:
            st.write(f"• {entry}")
        if result.get("critic_feedback"):
            st.caption("Committee reviewer feedback")
            st.json(result["critic_feedback"])
    with evidence_tab:
        citations = result.get("policy_citations") or []
        if not citations:
            st.info("No policy citations were attached to this assessment.")
        for citation in citations:
            with st.expander(str(citation.get("citation") or citation.get("rule_id") or "Policy rule")):
                st.caption(f"Source: {citation.get('source', 'unknown')}")
                st.write(citation.get("text", ""))
    with details_tab:
        st.json({"dscr_metrics": dscr, "bureau_data": bureau, "bank_data": bank}, expanded=False)


def _render_policy() -> None:
    st.subheader("Ask the credit policy")
    st.caption("Answers use only the indexed policy corpus. If the evidence is insufficient, the assistant abstains.")
    question = st.text_input("Policy question", placeholder="What DSCR floor applies to manufacturing SMEs?", help="Ask about a rule, threshold, exception, or required document.")
    if st.button("Search policy", type="primary", disabled=not question.strip()):
        from src.rag import answer_policy_question

        with st.spinner("Searching policy evidence…"):
            try:
                st.session_state["last_policy_answer"] = answer_policy_question(question.strip())
            except Exception as exc:  # noqa: BLE001
                logger.exception("Policy query failed: %s", exc)
                st.error(f"Policy search failed: {exc}. Build the policy index from the sidebar and try again.")
    answer = st.session_state.get("last_policy_answer")
    if answer is None:
        return
    from src.rag import ABSTENTION_RESPONSE

    if answer.answer == ABSTENTION_RESPONSE:
        st.warning(answer.answer)
    else:
        st.markdown("### Grounded answer")
        st.write(answer.answer)
    st.markdown("### Sources")
    if answer.citations:
        for citation in answer.citations:
            st.write(f"• {citation}")
    else:
        st.caption("No verified policy citation was returned.")
    with st.expander("Retrieved evidence", expanded=False):
        for chunk in answer.retrieved_chunks:
            st.markdown(f"**{chunk.citation}**")
            st.caption(f"{chunk.source_type} · {chunk.file_path}")
            st.write(chunk.text)


def _render_evaluation() -> None:
    st.subheader("Evaluation overview")
    st.caption("Committed artifacts for the golden set, retrieval, routing, and guardrails.")
    artifacts = _load_evaluation_artifacts(_file_signature(DEFAULT_EVALS_DIR))
    if not artifacts:
        st.info(f"No evaluation artifacts found in {DEFAULT_EVALS_DIR}.")
        return
    payloads = [payload for _, payload in artifacts]
    metrics = [
        ("Intake consistency", "intake", "accuracy"),
        ("Corpus source coverage", "retrieval", "corpus_coverage"),
        ("Live route accuracy", "routing", "accuracy"),
        ("Guardrail block rate", "guardrails", "block_rate"),
        ("Matched-pair fixture validity", "matched_pairs", "fixture_accuracy"),
    ]
    columns = st.columns(len(metrics))
    for column, (label, component, key) in zip(columns, metrics, strict=True):
        column.metric(
            label,
            _format_pct_or_score(_find_component_metric(payloads, component, key)),
        )
    with st.expander("Evaluation artifacts", expanded=False):
        for filename, payload in artifacts:
            st.markdown(f"**{filename}**")
            st.json(payload)


def main() -> None:
    st.set_page_config(page_title="WealthPilot · SME underwriting", page_icon="💼", layout="wide")
    st.title("WealthPilot")
    st.caption("SME underwriting workspace: risk signals, grounded policy evidence, and human-controlled decisions.")
    records = _load_sample_applications(_file_signature(DEFAULT_RECORDS_PATH))
    default_sample = _render_sidebar(records)
    underwriting_tab, policy_tab, evaluation_tab = st.tabs(["Underwrite", "Policy search", "Evaluation"])
    with underwriting_tab:
        _render_underwriting(default_sample)
    with policy_tab:
        _render_policy()
    with evaluation_tab:
        _render_evaluation()


if __name__ == "__main__":
    main()
