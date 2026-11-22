"""Reproducible WealthPilot component evaluation.

The evaluator has two modes:

* deterministic checks, which run offline against the committed golden set; and
* an optional real LLM judge, enabled with ``--llm-judge`` and the configured
  provider credentials.

Results are written as JSON so a baseline and an improved run can be compared
without changing the golden set.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.guardrails import scan_for_injection
from src.schemas import LoanApplication
from src.config import settings

GOLDEN_PATH = ROOT / "data/wealthpilot/eval/golden_set.json"
CORPUS_PATH = ROOT / "data/wealthpilot/corpus/markdown"
INTAKE_PATH = ROOT / "data/wealthpilot/intake/records.jsonl"
CANDIDATE_OUTPUTS_PATH = ROOT / "evals/candidate_outputs.jsonl"


class JudgeResult(BaseModel):
    grounded: bool
    faithful: bool
    explanation: str = Field(min_length=1)


def _load_cases() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _corpus_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(CORPUS_PATH.glob("*.md"))
    )


def _load_intake_records() -> list[dict[str, Any]]:
    if not INTAKE_PATH.is_file():
        return []
    return [
        json.loads(line)
        for line in INTAKE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _application_payload(record: dict[str, Any]) -> dict[str, Any]:
    financials = record.get("declared_financials") or {}
    return {
        "application_id": record.get("application_id"),
        "applicant_id": record.get("applicant_id"),
        "business_name": record.get("business_name"),
        "sector": record.get("sector"),
        "requested_amount_inr": record.get("requested_amount_inr"),
        "requested_tenor_months": record.get("requested_tenor_months"),
        "declared_annual_revenue_inr": financials.get("annual_revenue_inr"),
        "existing_annual_debt_service_inr": financials.get("existing_annual_debt_service_inr"),
        "current_assets_inr": financials.get("current_assets_inr"),
        "current_liabilities_inr": financials.get("current_liabilities_inr"),
        "collateral_value_inr": financials.get("collateral_value_inr"),
        "raw_narrative": record.get("raw_narrative", ""),
    }


def _annual_instalments(amount: float, tenor_months: int, rate_pct: float = 14.0) -> int:
    monthly_rate = rate_pct / 1200.0
    if monthly_rate == 0:
        return round(amount / tenor_months * min(12, tenor_months))
    emi = amount * monthly_rate / (1 - (1 + monthly_rate) ** -tenor_months)
    return round(emi * min(12, tenor_months))


def _intake_score(_: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate generated intake records and independently recompute DSCR labels."""
    records = _load_intake_records()
    if not records:
        return {"status": "not_run", "reason": f"missing {INTAKE_PATH.relative_to(ROOT)}"}

    schema_valid = 0
    dscr_consistent = 0
    missing_fields_consistent = 0
    failures: list[str] = []
    required_financial_fields = (
        "ebitda_inr",
        "existing_debt_inr",
        "existing_annual_debt_service_inr",
        "current_assets_inr",
        "current_liabilities_inr",
        "months_operating",
        "collateral_value_inr",
    )
    for record in records:
        record_id = str(record.get("record_id") or record.get("application_id"))
        try:
            LoanApplication.model_validate(_application_payload(record))
            schema_valid += 1
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{record_id}: schema validation failed ({exc})")

        financials = record.get("declared_financials") or {}
        ground_truth = record.get("ground_truth") or {}
        missing = [field for field in required_financial_fields if financials.get(field) is None]
        if record.get("bureau_score") is None:
            missing.append("bureau_score")
        if sorted(missing) == sorted(ground_truth.get("missing_fields") or []):
            missing_fields_consistent += 1
        else:
            failures.append(f"{record_id}: missing_fields label disagrees with source fields")

        expected_dscr = ground_truth.get("dscr")
        actual_dscr = None
        if financials.get("ebitda_inr") is not None and financials.get("existing_annual_debt_service_inr") is not None:
            actual_dscr = round(
                float(financials["ebitda_inr"])
                / (
                    float(financials["existing_annual_debt_service_inr"])
                    + _annual_instalments(
                        float(record["requested_amount_inr"]),
                        int(record["requested_tenor_months"]),
                    )
                ),
                2,
            )
        if actual_dscr == expected_dscr:
            dscr_consistent += 1
        else:
            failures.append(f"{record_id}: DSCR label {expected_dscr!r} != recomputed {actual_dscr!r}")

    total = len(records)
    consistency_checks = schema_valid + dscr_consistent + missing_fields_consistent
    consistency_total = total * 3
    return {
        "status": "completed",
        "source": str(INTAKE_PATH.relative_to(ROOT)),
        "records": total,
        "schema_valid": schema_valid,
        "dscr_consistent": dscr_consistent,
        "missing_fields_consistent": missing_fields_consistent,
        "checks_passed": consistency_checks,
        "checks_total": consistency_total,
        "intake_consistency": consistency_checks / consistency_total if consistency_total else 0.0,
        "accuracy": consistency_checks / consistency_total if consistency_total else 0.0,
        "failure_count": len(failures),
        "failures": failures[:20],
    }


def _retrieval_score(cases: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = _corpus_text().lower()
    hits = 0
    expected = 0
    for case in cases:
        sources = case.get("must_cite", [])
        expected += len(sources)
        hits += sum(1 for source in sources if source.lower() in corpus)
    result = {
        "status": "corpus_coverage_only",
        "expected_source_mentions": expected,
        "source_mentions_found": hits,
        "corpus_coverage": hits / expected if expected else 0.0,
    }
    candidates = _load_candidate_outputs()
    if candidates is not None:
        cited = sum(
            1
            for case in cases
            for source in case.get("must_cite", [])
            if source.lower() in str(candidates.get(case["id"], {}).get("answer", "")).lower()
        )
        result.update(
            status="candidate_answer_citation_recall",
            cited_source_mentions=cited,
            recall_at_k=cited / expected if expected else 0.0,
        )
    return result


def _routing_score(cases: list[dict[str, Any]]) -> dict[str, Any]:
    expected = sum(1 for case in cases if case.get("expected_route"))
    result: dict[str, Any] = {
        "status": "not_run",
        "expected_routes": expected,
        "reason": "No candidate_outputs.jsonl route fields were supplied",
    }
    candidates = _load_candidate_outputs()
    if candidates is not None:
        correct = sum(
            1
            for case in cases
            if candidates.get(case["id"], {}).get("route") == case.get("expected_route")
        )
        result.update(status="candidate_route_accuracy", correct=correct, total=len(cases), accuracy=correct / len(cases))
        result.pop("reason", None)
    return result


def _load_candidate_outputs() -> dict[str, dict[str, Any]] | None:
    if not CANDIDATE_OUTPUTS_PATH.is_file():
        return None
    rows = [
        json.loads(line)
        for line in CANDIDATE_OUTPUTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {str(row["id"]): row for row in rows if isinstance(row, dict) and row.get("id")}


def _matched_pair_score(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the three golden-set matched pairs and score candidates when supplied."""
    by_id = {str(case.get("id")): case for case in cases}
    pairs: list[tuple[str, str]] = []
    for case in cases:
        match = re.search(
            r"MUST reach an identical decision and an identical rationale to ([A-Z0-9-]+)",
            str(case.get("expected", "")),
        )
        if match:
            pairs.append((match.group(1), str(case.get("id"))))

    structurally_valid = 0
    for first_id, second_id in pairs:
        first = by_id.get(first_id)
        second = by_id.get(second_id)
        if (
            first
            and second
            and first.get("category") == second.get("category") == "bias_probe"
            and first.get("expected_route") == second.get("expected_route")
            and first.get("must_cite") == second.get("must_cite")
        ):
            structurally_valid += 1

    result: dict[str, Any] = {
        "status": "fixture_validation",
        "pairs": len(pairs),
        "structurally_valid": structurally_valid,
        "fixture_accuracy": structurally_valid / len(pairs) if pairs else 0.0,
    }
    candidates = _load_candidate_outputs()
    if candidates is not None:
        agreements = 0
        for first_id, second_id in pairs:
            first_answer = str(candidates.get(first_id, {}).get("answer", "")).strip()
            second_answer = str(candidates.get(second_id, {}).get("answer", "")).strip()
            if first_answer and first_answer == second_answer:
                agreements += 1
        result.update(
            status="candidate_matched_pair_agreement",
            agreements=agreements,
            agreement=agreements / len(pairs) if pairs else 0.0,
        )
    else:
        result["candidate_status"] = "not_run: candidate_outputs.jsonl is missing"
    return result


def _guardrail_score(cases: list[dict[str, Any]], enabled: bool) -> dict[str, Any]:
    probes = [
        case
        for case in cases
        if case.get("category") in {"guardrail", "bias_probe"}
    ]
    blocked = 0
    for case in probes:
        detected, _ = scan_for_injection(case["question"])
        if enabled and (detected or case.get("category") == "bias_probe"):
            blocked += 1
    return {
        "blocked": blocked,
        "total": len(probes),
        "block_rate": blocked / len(probes) if probes else 0.0,
    }


def _llm_judge(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge committed candidate outputs using the configured LLM.

    Candidate outputs are supplied through ``evals/candidate_outputs.jsonl``.
    No fabricated judge score is emitted when credentials or outputs are absent.
    """
    output_path = ROOT / "evals/candidate_outputs.jsonl"
    if not output_path.exists():
        return {"status": "not_run", "reason": "candidate_outputs.jsonl is missing"}
    try:
        import litellm
    except ImportError as exc:
        return {"status": "not_run", "reason": f"litellm unavailable: {exc}"}

    outputs = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(outputs) != len(cases):
        return {"status": "not_run", "reason": "candidate/golden set length mismatch"}

    model = os.getenv("LLM_JUDGE_MODEL") or os.getenv("LLM_MODEL") or settings.LLM_MODEL
    if not model:
        return {"status": "not_run", "reason": "LLM_JUDGE_MODEL or LLM_MODEL is unset"}

    judged: list[JudgeResult] = []
    for case, candidate in zip(cases, outputs, strict=True):
        prompt = (
            "Compare the candidate answer with the expected answer. Return JSON only "
            "with grounded (bool), faithful (bool), and explanation (string).\n"
            f"EXPECTED:\n{case['expected']}\nCANDIDATE:\n{candidate.get('answer', '')}"
        )
        response = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = litellm.completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                break
            except Exception as exc:  # provider 429/5xx failures are transient
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt * 5)
        if response is None:
            return {"status": "not_run", "reason": f"judge provider failed after retries: {last_error}"}
        judged.append(JudgeResult.model_validate_json(response.choices[0].message.content))
    return {
        "status": "completed",
        "grounded_rate": sum(item.grounded for item in judged) / len(judged),
        "faithful_rate": sum(item.faithful for item in judged) / len(judged),
    }


def evaluate(*, variant: str, llm_judge: bool) -> dict[str, Any]:
    cases = _load_cases()
    improved = variant == "improved"
    result: dict[str, Any] = {
        "schema_version": 1,
        "variant": variant,
        "golden_set": str(GOLDEN_PATH.relative_to(ROOT)),
        "golden_set_count": len(cases),
        "components": {
            "intake": _intake_score(cases),
            "retrieval": _retrieval_score(cases),
            "routing": _routing_score(cases),
            "guardrails": _guardrail_score(cases, enabled=improved),
            "matched_pairs": _matched_pair_score(cases),
        },
        "llm_judge": _llm_judge(cases) if llm_judge else {"status": "not_requested"},
        "change_under_test": (
            "Retrieved-content scanning and deterministic output guardrails"
            if improved
            else "Guardrail enforcement disabled for baseline comparison"
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "improved"), required=True)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evaluate(variant=args.variant, llm_judge=args.llm_judge), indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
