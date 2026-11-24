"""Comprehensive unit tests for all WealthPilot underwriting tools in ``src/tools.py``."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import invoke_tool


# ---------------------------------------------------------------------------
# 1. calculate_dscr_and_ratios
# ---------------------------------------------------------------------------


def test_clean_application_meets_all_policy_checks(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify exact DSCR calculation and passing policy checks for APP-1001."""
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-1001")

    assert "error" not in result
    assert result["application_id"] == "APP-1001"
    assert result["applicant_id"] == "APL-101"
    # DSCR = 9_000_000 / (1_500_000 + 3_100_000) = 9_000_000 / 4_600_000 = 1.9565
    assert result["dscr"] == pytest.approx(1.9565, rel=1e-4)
    assert result["meets_dscr_floor"] is True
    assert result["doc_mismatch"] is False
    assert result["fraud_hold"] is False
    assert result["tenor_above_limit"] is False
    assert result["collateral_shortfall"] is False
    assert result["sector_cap_breached"] is False
    assert result["policy_flags"] == []
    assert result["proposed_debt_service_source"] == "loan_applications.proposed_annual_debt_service_inr"


def test_derived_emi_and_multiple_policy_breaches(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Missing application rate does not create a fabricated DSCR basis."""
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="app-1002")

    assert "error" not in result
    assert result["proposed_debt_service_source"] == "missing_application_debt_service_basis"
    assert result["proposed_annual_debt_service_inr"] is None
    assert result["dscr"] is None
    assert result["meets_dscr_floor"] is False
    assert result["doc_mismatch"] is True
    assert result["fraud_hold"] is False
    assert result["max_tenor_months"] == 48
    assert result["tenor_above_limit"] is True
    assert result["collateral_shortfall"] is True
def test_fraud_hold_and_sector_cap_breach(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify >25% computed revenue variance triggers FRAUD_HOLD and SECTOR_CAP triggers when exceeded."""
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-1003")

    assert "error" not in result
    assert result["revenue_variance_pct"] == pytest.approx(30.0)
    assert result["doc_mismatch"] is True
    assert result["fraud_hold"] is True
    assert result["sector_cap_breached"] is True
    assert "FRAUD_HOLD" in result["policy_flags"]
    assert "DOC_MISMATCH" in result["policy_flags"]
    assert "SECTOR_CAP" in result["policy_flags"]


def test_columnar_policy_limits_layout_supported(
    tools_module: ModuleType,
    csv_writer: Callable[[str, list[str], list[dict[str, Any]]], Path],
) -> None:
    """Verify columnar ``policy_limits.csv`` layout is parsed identically to key-value format."""
    csv_writer(
        "policy_limits.csv",
        ["doc_mismatch_threshold_pct", "fraud_hold_threshold_pct", "max_tenor_months"],
        [{"doc_mismatch_threshold_pct": "3.0", "fraud_hold_threshold_pct": "25.0", "max_tenor_months": "36"}],
    )

    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-1001")

    assert "error" not in result
    assert result["doc_mismatch"] is True
    assert result["tenor_above_limit"] is True


def test_zero_debt_service_does_not_divide_by_zero(
    tools_module: ModuleType,
    csv_writer: Callable[[str, list[str], list[dict[str, Any]]], Path],
) -> None:
    """Ensure zero existing and zero proposed debt service returns ``dscr=None`` safely."""
    csv_writer(
        "loan_applications.csv",
        [
            "application_id",
            "applicant_id",
            "sector",
            "risk_grade",
            "requested_amount_inr",
            "requested_tenor_months",
            "ebitda_inr",
            "existing_annual_debt_service_inr",
            "proposed_annual_debt_service_inr",
            "collateral_value_inr",
            "revenue_variance_pct",
        ],
        [
            {
                "application_id": "APP-ZERO",
                "applicant_id": "APL-101",
                "sector": "Manufacturing",
                "risk_grade": "A",
                "requested_amount_inr": "0",
                "requested_tenor_months": "12",
                "ebitda_inr": "5000000",
                "existing_annual_debt_service_inr": "0",
                "proposed_annual_debt_service_inr": "0",
                "collateral_value_inr": "0",
                "revenue_variance_pct": "0",
            }
        ],
    )

    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-ZERO")

    assert "error" not in result
    assert result["dscr"] is None
    assert result["meets_dscr_floor"] is False


@pytest.mark.parametrize("bad_id", ["", "   ", "APP-UNKNOWN"])
def test_calculate_dscr_invalid_application_id_returns_clean_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    bad_id: str,
) -> None:
    """Ensure invalid or empty application IDs return an error dictionary cleanly."""
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id=bad_id)

    assert "error" in result
    assert "not found" in result["error"].lower()


def test_missing_applicant_for_application_returns_applicant_not_found(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Ensure an application referencing a missing applicant returns Applicant ID not found."""
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-ORPHAN")

    assert result["error"] == "Applicant ID not found"
    assert result["applicant_id"] == "APL-9999"


def test_calculate_dscr_missing_csv_returns_error_dict(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Ensure missing CSV files return an error dict instead of raising FileNotFoundError."""
    (mock_api_dir / "loan_applications.csv").unlink()
    result = invoke_tool(tools_module.calculate_dscr_and_ratios, application_id="APP-1001")

    assert "error" in result
    assert "loan_applications.csv" in result["error"]


# ---------------------------------------------------------------------------
# 2. convert_fx_and_interest
# ---------------------------------------------------------------------------


def test_fx_conversion_and_reducing_balance_emi(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify USD-to-INR receivables conversion and exact monthly/annual EMI outputs."""
    result = invoke_tool(
        tools_module.convert_fx_and_interest,
        usd_amount=100000.0,
        principal_inr=10000000.0,
        annual_rate_pct=12.0,
        tenor_months=12,
    )

    assert "error" not in result
    assert result["usd_inr_rate"] == pytest.approx(84.25)
    assert result["rate_as_of"] == "2026-11-01"
    assert result["converted_receivables_inr"] == pytest.approx(8425000.0)
    assert result["monthly_emi_inr"] == pytest.approx(888487.89, abs=0.05)
    assert result["annual_debt_service_inr"] == pytest.approx(result["monthly_emi_inr"] * 12.0, abs=0.2)
    assert result["total_repayment_inr"] == pytest.approx(result["monthly_emi_inr"] * 12.0, abs=0.2)
    assert result["total_interest_inr"] == pytest.approx(result["total_repayment_inr"] - 10000000.0, abs=0.2)


def test_zero_interest_rate_uses_straight_line_principal(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify 0% annual interest divides principal evenly across tenor months."""
    result = invoke_tool(
        tools_module.convert_fx_and_interest,
        usd_amount=0.0,
        principal_inr=1200000.0,
        annual_rate_pct=0.0,
        tenor_months=24,
    )

    assert "error" not in result
    assert result["converted_receivables_inr"] == pytest.approx(0.0)
    assert result["monthly_emi_inr"] == pytest.approx(50000.0)
    assert result["annual_debt_service_inr"] == pytest.approx(600000.0)
    assert result["total_interest_inr"] == pytest.approx(0.0)


def test_columnar_reference_rates_schema_supported(
    tools_module: ModuleType,
    csv_writer: Callable[[str, list[str], list[dict[str, Any]]], Path],
) -> None:
    """Verify fallback parsing for ``usd_inr_rate`` column layout in reference_rates.csv."""
    csv_writer(
        "reference_rates.csv",
        ["usd_inr_rate", "as_of_date"],
        [{"usd_inr_rate": "85.00", "as_of_date": "2026-11-10"}],
    )

    result = invoke_tool(
        tools_module.convert_fx_and_interest,
        usd_amount=1000.0,
        principal_inr=500000.0,
        annual_rate_pct=10.0,
        tenor_months=12,
    )

    assert "error" not in result
    assert result["usd_inr_rate"] == pytest.approx(85.0)
    assert result["converted_receivables_inr"] == pytest.approx(85000.0)


@pytest.mark.parametrize(
    ("usd_amount", "principal_inr", "annual_rate_pct", "tenor_months"),
    [
        (-10.0, 100000.0, 10.0, 12),
        (100.0, -1000.0, 10.0, 12),
        (100.0, 100000.0, -1.0, 12),
        (100.0, 100000.0, 10.0, 0),
        (100.0, 100000.0, 10.0, -6),
    ],
)
def test_convert_fx_invalid_numeric_inputs_return_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    usd_amount: float,
    principal_inr: float,
    annual_rate_pct: float,
    tenor_months: int,
) -> None:
    """Ensure negative amounts/rates or non-positive tenor return an error dictionary."""
    result = invoke_tool(
        tools_module.convert_fx_and_interest,
        usd_amount=usd_amount,
        principal_inr=principal_inr,
        annual_rate_pct=annual_rate_pct,
        tenor_months=tenor_months,
    )

    assert "error" in result


def test_missing_usd_inr_rate_returns_error(
    tools_module: ModuleType,
    csv_writer: Callable[[str, list[str], list[dict[str, Any]]], Path],
) -> None:
    """Ensure missing USD/INR entry in reference_rates.csv returns a structured error."""
    csv_writer(
        "reference_rates.csv",
        ["currency_pair", "rate", "as_of_date"],
        [{"currency_pair": "EUR/INR", "rate": "91.10", "as_of_date": "2026-11-01"}],
    )

    result = invoke_tool(
        tools_module.convert_fx_and_interest,
        usd_amount=5000.0,
        principal_inr=1000000.0,
        annual_rate_pct=11.0,
        tenor_months=24,
    )

    assert "error" in result


# ---------------------------------------------------------------------------
# 3. lookup_credit_bureau
# ---------------------------------------------------------------------------


def test_prime_applicant_bureau_metrics(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify prime applicant bureau scores and clean delinquency flags."""
    result = invoke_tool(tools_module.lookup_credit_bureau, applicant_id="APL-101")

    assert "error" not in result
    assert result["applicant_id"] == "APL-101"
    assert result["bureau_score"] == 785
    assert result["enquiries_last_6m"] == 2
    assert result["dpd_30_last_12m"] == 0
    assert result["dpd_90_ever"] == 0
    assert result["written_off_amount_inr"] == pytest.approx(0.0)
    assert result["has_severe_delinquency"] is False
    assert result["has_recent_delinquency"] is False
    assert result["high_enquiry_velocity"] is False


def test_subprime_applicant_parses_formatted_currency_and_flags(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify formatted currency write-off parsing and severe delinquency risk flags."""
    result = invoke_tool(tools_module.lookup_credit_bureau, applicant_id="apl-103")

    assert "error" not in result
    assert result["bureau_score"] == 610
    assert result["enquiries_last_6m"] == 9
    assert result["dpd_30_last_12m"] == 4
    assert result["dpd_90_ever"] == 1
    assert result["written_off_amount_inr"] == pytest.approx(250000.0)
    assert result["has_severe_delinquency"] is True
    assert result["has_recent_delinquency"] is True
    assert result["high_enquiry_velocity"] is True


@pytest.mark.parametrize("bad_id", ["", "   ", "APL-DOES-NOT-EXIST"])
def test_bureau_invalid_applicant_id_returns_exact_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    bad_id: str,
) -> None:
    """Ensure invalid applicant IDs never crash and return Applicant ID not found."""
    result = invoke_tool(tools_module.lookup_credit_bureau, applicant_id=bad_id)

    assert result == {"error": "Applicant ID not found"}


def test_missing_bureau_csv_returns_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Ensure missing bureau_reports.csv returns a clean error dictionary."""
    (mock_api_dir / "bureau_reports.csv").unlink()
    result = invoke_tool(tools_module.lookup_credit_bureau, applicant_id="APL-101")

    assert "error" in result
    assert "bureau_reports.csv" in result["error"]


# ---------------------------------------------------------------------------
# 4. analyze_bank_statements
# ---------------------------------------------------------------------------


def test_multi_month_bank_statement_aggregation(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify multi-month averages, min balance, net cashflow, and total bounced instruments."""
    result = invoke_tool(tools_module.analyze_bank_statements, applicant_id="APL-101")

    assert "error" not in result
    assert result["applicant_id"] == "APL-101"
    assert result["months_analyzed"] == 3
    assert result["avg_monthly_inflow_inr"] == pytest.approx(11000000.0)
    assert result["avg_monthly_outflow_inr"] == pytest.approx(8500000.0)
    assert result["net_monthly_cashflow_inr"] == pytest.approx(2500000.0)
    assert result["avg_closing_balance_inr"] == pytest.approx(4000000.0)
    assert result["min_closing_balance_inr"] == pytest.approx(3500000.0)
    assert result["total_bounced_instruments"] == 1
    assert result["outflow_to_inflow_ratio"] == pytest.approx(8500000.0 / 11000000.0, rel=1e-4)


def test_zero_inflow_guards_against_division_by_zero(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Ensure zero inflow sets outflow_to_inflow_ratio to None without crashing."""
    result = invoke_tool(tools_module.analyze_bank_statements, applicant_id=" apl-102 ")

    assert "error" not in result
    assert result["applicant_id"] == "APL-102"
    assert result["months_analyzed"] == 1
    assert result["avg_monthly_inflow_inr"] == pytest.approx(0.0)
    assert result["avg_monthly_outflow_inr"] == pytest.approx(500000.0)
    assert result["net_monthly_cashflow_inr"] == pytest.approx(-500000.0)
    assert result["total_bounced_instruments"] == 3
    assert result["outflow_to_inflow_ratio"] is None


@pytest.mark.parametrize("bad_id", ["", "   ", "APL-UNKNOWN"])
def test_bank_statements_invalid_applicant_id_returns_exact_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    bad_id: str,
) -> None:
    """Ensure missing applicant bank statements return Applicant ID not found."""
    result = invoke_tool(tools_module.analyze_bank_statements, applicant_id=bad_id)

    assert result == {"error": "Applicant ID not found"}


def test_missing_bank_statements_csv_returns_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Ensure missing bank_statements.csv returns an error dictionary cleanly."""
    (mock_api_dir / "bank_statements.csv").unlink()
    result = invoke_tool(tools_module.analyze_bank_statements, applicant_id="APL-101")

    assert "error" in result
    assert "bank_statements.csv" in result["error"]


# ---------------------------------------------------------------------------
# 5. escalate_to_human
# ---------------------------------------------------------------------------


def test_escalation_matches_grade_and_amount_signatory(
    tools_module: ModuleType,
    mock_api_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify escalation assigns exact signatory level and emits warning log."""
    with caplog.at_level(logging.WARNING):
        result = invoke_tool(
            tools_module.escalate_to_human,
            application_id="APP-1001",
            reason="Manual covenant exception review",
        )

    assert "error" not in result
    assert result["application_id"] == "APP-1001"
    assert result["applicant_id"] == "APL-101"
    assert result["escalation_status"] == "ESCALATED_FOR_HUMAN_REVIEW"
    assert result["requires_human_review"] is True
    assert result["required_signatory_level"] == "Regional Credit Manager"
    assert result["review_sla_hours"] == 12
    assert result["reason"] == "Manual covenant exception review"
    parsed_ts = datetime.fromisoformat(result["escalated_at"])
    assert parsed_ts.tzinfo is not None
    assert any("APP-1001" in record.getMessage() for record in caplog.records)


def test_high_value_application_routes_to_open_ended_committee_tier(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify applications >= 25M route to Central Credit Committee with 48h SLA."""
    result = invoke_tool(
        tools_module.escalate_to_human,
        application_id="APP-1003",
        reason="FRAUD_HOLD triggered due to 30% revenue variance and SECTOR_CAP breach",
    )

    assert "error" not in result
    assert result["required_signatory_level"] == "Central Credit Committee"
    assert result["review_sla_hours"] == 48


def test_mid_tier_grade_b_routes_to_senior_credit_officer(
    tools_module: ModuleType,
    mock_api_dir: Path,
) -> None:
    """Verify non-grade-A 15M application falls through to Senior Credit Officer."""
    result = invoke_tool(
        tools_module.escalate_to_human,
        application_id="APP-1002",
        reason="DSCR below sector floor",
    )

    assert "error" not in result
    assert result["required_signatory_level"] == "Senior Credit Officer"
    assert result["review_sla_hours"] == 24


@pytest.mark.parametrize("bad_id", ["", "   ", "APP-INVALID"])
def test_escalate_invalid_application_id_returns_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    bad_id: str,
) -> None:
    """Ensure invalid application IDs do not crash escalate_to_human."""
    result = invoke_tool(
        tools_module.escalate_to_human,
        application_id=bad_id,
        reason="Policy exception",
    )

    assert "error" in result
    assert "not found" in result["error"].lower()


@pytest.mark.parametrize("empty_reason", ["", "   "])
def test_empty_reason_returns_error(
    tools_module: ModuleType,
    mock_api_dir: Path,
    empty_reason: str,
) -> None:
    """Ensure blank escalation reasons are rejected cleanly."""
    result = invoke_tool(
        tools_module.escalate_to_human,
        application_id="APP-1001",
        reason=empty_reason,
    )

    assert result == {"error": "Escalation reason must be a non-empty string"}

