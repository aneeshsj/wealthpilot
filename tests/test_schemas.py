"""Unit tests for src/schemas.py including hydration and graceful error handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import src.schemas as schemas_mod
from src.schemas import (
    CommitteeVerdict,
    FinancialSnapshot,
    LoanApplication,
    clear_customer_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Ensure customer hydration cache is clean before and after every test."""
    clear_customer_cache()
    yield
    clear_customer_cache()


def _valid_application_payload() -> dict[str, Any]:
    return {
        "application_id": "APP-2026-001",
        "applicant_id": "SME-1001",
        "business_name": "  Kaveri Precision Gears Pvt Ltd  ",
        "sector": "Manufacturing",
        "requested_amount_inr": 2_000_000.0,
        "requested_tenor_months": 36,
        "declared_annual_revenue_inr": 18_500_000.0,
        "existing_annual_debt_service_inr": 450_000.0,
        "current_assets_inr": 6_200_000.0,
        "current_liabilities_inr": 3_100_000.0,
        "collateral_type": "Industrial Machinery",
        "collateral_value_inr": 3_000_000.0,
        "export_receivables_usd": 25_000.0,
        "proprietor_name": "Ananya Rao",
        "city": "Pune",
        "application_language": "en",
        "raw_narrative": "Working capital expansion for CNC tooling line.",
        "unexpected_extra_field": "ignored",
    }


def test_loan_application_valid_and_strips_whitespace() -> None:
    """Validate complete LoanApplication payload and whitespace/extra handling."""
    app = LoanApplication.model_validate(_valid_application_payload())
    assert app.application_id == "APP-2026-001"
    assert app.business_name == "Kaveri Precision Gears Pvt Ltd"
    assert app.requested_amount_inr == 2_000_000.0
    assert app.requested_tenor_months == 36
    assert not hasattr(app, "unexpected_extra_field")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("requested_amount_inr", 0.0),
        ("requested_amount_inr", -100_000.0),
        ("requested_tenor_months", 0),
        ("requested_tenor_months", -12),
        ("declared_annual_revenue_inr", 0.0),
        ("declared_annual_revenue_inr", -50_000.0),
    ],
)
def test_loan_application_rejects_non_positive_amounts_and_tenors(
    field: str,
    bad_value: float | int,
) -> None:
    """Ensure strictly positive constraints reject zero or negative values."""
    payload = _valid_application_payload()
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        LoanApplication.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    [
        "existing_annual_debt_service_inr",
        "current_assets_inr",
        "current_liabilities_inr",
        "collateral_value_inr",
        "export_receivables_usd",
    ],
)
def test_loan_application_allows_zero_but_rejects_negative_balances(field: str) -> None:
    """Allow zero balances (e.g., debt-free SMEs) while rejecting negative values."""
    zero_payload = _valid_application_payload()
    zero_payload[field] = 0.0
    app = LoanApplication.model_validate(zero_payload)
    assert getattr(app, field) == 0.0

    neg_payload = _valid_application_payload()
    neg_payload[field] = -1.0
    with pytest.raises(ValidationError):
        LoanApplication.model_validate(neg_payload)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("requested_amount_inr", float("nan")),
        ("declared_annual_revenue_inr", float("inf")),
        ("application_id", "   "),
        ("business_name", ""),
    ],
)
def test_loan_application_rejects_non_finite_and_blank_identifiers(
    field: str,
    bad_value: Any,
) -> None:
    """Reject NaN, infinity, and empty required string identifiers."""
    payload = _valid_application_payload()
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        LoanApplication.model_validate(payload)


def test_loan_application_hydrates_from_csv_and_prioritizes_incoming_edits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrate omitted customer fields from CSV while prioritizing incoming edits."""
    csv_file = tmp_path / "loan_applications.csv"
    csv_file.write_text(
        "application_id,sme_id,business_name,sector,requested_amount_inr,"
        "requested_tenor_months,declared_annual_revenue_inr,existing_annual_debt_service_inr,"
        "collateral_type,collateral_value_inr,city,application_language\n"
        'APP-HYD-01,SME-9001,Apex Textiles,Textiles,"15,00,000",24,"1,20,00,000",nan,'
        "Property,2200000,Surat,gu\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        schemas_mod,
        "_resolve_customer_csv_paths",
        lambda explicit_path=None: [csv_file],
    )

    partial_payload = {
        "sme_id": "SME-9001",
        "requested_amount_inr": 1_800_000.0,
        "city": "Ahmedabad",
    }
    app = LoanApplication.model_validate(partial_payload)

    assert app.application_id == "APP-HYD-01"
    assert app.applicant_id == "SME-9001"
    assert app.business_name == "Apex Textiles"
    assert app.requested_amount_inr == 1_800_000.0  # Incoming edit overrides CSV
    assert app.declared_annual_revenue_inr == 12_000_000.0  # Currency CSV parsed
    assert app.existing_annual_debt_service_inr == 0.0  # NaN CSV value -> fallback
    assert app.collateral_type == "Property"
    assert app.city == "Ahmedabad"
    assert app.proprietor_name == "Not Provided"


def test_loan_application_applies_safe_fallbacks_when_csv_missing_or_corrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle unreadable/corrupt hydration CSVs gracefully without raising I/O errors."""
    corrupt_csv = tmp_path / "corrupt.csv"
    corrupt_csv.write_bytes(b"\xff\xfe\x00\xbad_utf8_bytes")
    monkeypatch.setattr(
        schemas_mod,
        "_resolve_customer_csv_paths",
        lambda explicit_path=None: [corrupt_csv],
    )

    minimal_new_customer = {
        "application_id": "APP-NEW-01",
        "sme_id": "SME-NEW-01",
        "business_name": "Sunrise Foods",
        "sector": "Food Processing",
        "requested_amount_inr": 1_000_000.0,
        "requested_tenor_months": 12,
        "declared_annual_revenue_inr": 8_000_000.0,
    }
    app = LoanApplication.model_validate(minimal_new_customer)
    assert app.applicant_id == "SME-NEW-01"
    assert app.collateral_type == "Unsecured"
    assert app.city == "Mumbai"
    assert app.application_language == "en"
    assert app.existing_annual_debt_service_inr == 0.0


def test_loan_application_safe_validate_handles_all_errors_gracefully() -> None:
    """Ensure safe_validate returns structured errors without throwing exceptions."""
    ok_app, ok_errors = LoanApplication.safe_validate(_valid_application_payload())
    assert ok_app is not None
    assert ok_errors == []

    bad_json_app, bad_json_errors = LoanApplication.safe_validate("{not valid json")
    assert bad_json_app is None
    assert len(bad_json_errors) == 1
    assert "Invalid JSON" in bad_json_errors[0]

    non_dict_app, non_dict_errors = LoanApplication.safe_validate(["not", "an", "object"])
    assert non_dict_app is None
    assert non_dict_errors == ["Application payload must be a JSON object."]

    unhashable_payload = _valid_application_payload()
    unhashable_payload["applicant_id"] = ["unhashable", "list"]
    bad_id_app, bad_id_errors = LoanApplication.safe_validate(unhashable_payload)
    assert bad_id_app is None
    assert len(bad_id_errors) >= 1


def test_financial_snapshot_valid_and_allows_negative_ebitda() -> None:
    """Validate FinancialSnapshot metrics including optional/negative EBITDA."""
    snapshot = FinancialSnapshot(
        application_id="APP-2026-001",
        declared_annual_revenue_inr=18_500_000.0,
        verified_annual_revenue_inr=17_800_000.0,
        revenue_variance_pct=3.78,
        ebitda_inr=-120_000.0,
        annual_debt_service_inr=850_000.0,
        dscr=1.45,
        current_ratio=2.0,
        collateral_coverage_ratio=1.5,
        ltv_ratio=0.67,
        export_exposure_inr=2_075_000.0,
        revenue_mismatch_flag=False,
        fraud_hold_flag=False,
        notes=["GST filing verified"],
    )
    assert snapshot.ebitda_inr == -120_000.0
    assert snapshot.dscr == 1.45
    assert snapshot.notes == ["GST filing verified"]


def test_financial_snapshot_rejects_negative_ratios_and_nan() -> None:
    """Reject negative DSCR/LTV or non-finite values in FinancialSnapshot."""
    with pytest.raises(ValidationError):
        FinancialSnapshot(
            application_id="APP-2026-001",
            declared_annual_revenue_inr=10_000_000.0,
            verified_annual_revenue_inr=10_000_000.0,
            ebitda_inr=1_200_000.0,
            annual_debt_service_inr=500_000.0,
            dscr=-0.5,
            current_ratio=1.2,
        )

    with pytest.raises(ValidationError):
        FinancialSnapshot(
            application_id="APP-2026-001",
            declared_annual_revenue_inr=10_000_000.0,
            verified_annual_revenue_inr=10_000_000.0,
            ebitda_inr=1_200_000.0,
            annual_debt_service_inr=500_000.0,
            dscr=1.5,
            current_ratio=float("nan"),
        )

    snap, errs = FinancialSnapshot.safe_validate({"application_id": "APP-1", "dscr": -1.0})
    assert snap is None
    assert len(errs) >= 1


@pytest.mark.parametrize(
    "decision",
    ["approve", "borderline", "decline", "human_review"],
)
def test_committee_verdict_valid_decisions_and_list_cleaning(decision: str) -> None:
    """Accept all allowed decision literals and clean reason_codes/citations."""
    verdict = CommitteeVerdict.model_validate(
        {
            "decision": decision,
            "dscr": 1.65,
            "risk_grade": "BBB+",
            "reason_codes": [" DSCR_STRONG ", " ", "COLLATERAL_OK"],
            "citations": [" Policy-4.1 ", ""],
            "rationale": "Strong debt service coverage and verified GST revenue.",
        }
    )
    assert verdict.decision == decision
    assert verdict.reason_codes == ["DSCR_STRONG", "COLLATERAL_OK"]
    assert verdict.citations == ["Policy-4.1"]


def test_committee_verdict_normalizes_null_lists_and_safe_validate() -> None:
    """Gracefully normalize null list fields and return errors via safe_validate."""
    verdict, errs = CommitteeVerdict.safe_validate(
        {
            "decision": "approve",
            "dscr": 1.4,
            "risk_grade": "A",
            "reason_codes": None,
            "citations": None,
            "rationale": "Consistent cashflows.",
        }
    )
    assert verdict is not None
    assert errs == []
    assert verdict.reason_codes == []
    assert verdict.citations == []

    bad_verdict, bad_errs = CommitteeVerdict.safe_validate("invalid_type")
    assert bad_verdict is None
    assert bad_errs == ["CommitteeVerdict payload must be a mapping."]


def test_committee_verdict_rejects_invalid_fields() -> None:
    """Reject invalid decision literal, negative DSCR, or blank rationale/grade."""
    with pytest.raises(ValidationError):
        CommitteeVerdict.model_validate(
            {
                "decision": "manual_override",
                "dscr": 1.2,
                "risk_grade": "BB",
                "rationale": "Invalid decision literal.",
            }
        )

    with pytest.raises(ValidationError):
        CommitteeVerdict(
            decision="decline",
            dscr=-1.0,
            risk_grade="CCC",
            rationale="Negative DSCR.",
        )

    with pytest.raises(ValidationError):
        CommitteeVerdict(
            decision="borderline",
            dscr=1.1,
            risk_grade=" ",
            rationale="Missing risk grade.",
        )

    with pytest.raises(ValidationError):
        CommitteeVerdict(
            decision="human_review",
            dscr=1.1,
            risk_grade="B+",
            rationale="   ",
        )
