"""Deterministic financial, bureau, banking, and escalation tools for WealthPilot.

All quantitative calculations, policy threshold checks, and CSV data lookups
are enforced in Python rather than delegated to LLM reasoning.
"""

from __future__ import annotations

import csv
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

try:
    from src.config import FRAUD_HOLD_VARIANCE_PCT, REVENUE_MISMATCH_PCT, settings
except ImportError:
    try:
        from config import (  # type: ignore[import-not-found,no-redef]
            FRAUD_HOLD_VARIANCE_PCT,
            REVENUE_MISMATCH_PCT,
            settings,
        )
    except ImportError:
        from src.settings import settings  # type: ignore[import-not-found,no-redef]

logger = logging.getLogger(__name__)

_CSV_CACHE_LOCK = threading.Lock()
_CSV_CACHE: dict[tuple[str, int, int], tuple[dict[str, str], ...]] = {}

def _get_mock_api_dir() -> Path:
    """Resolve the mock API CSV directory from application settings."""
    return Path(settings.DATA_DIR) / "mock_api"


def _load_csv_rows(filename: str) -> list[dict[str, str]]:
    """Load and normalize rows from a CSV file inside ``mock_api``."""
    csv_path = _get_mock_api_dir() / filename
    if not csv_path.is_file():
        logger.error("Required mock API CSV file not found: %s", csv_path)
        raise FileNotFoundError(f"CSV file not found: {filename}")

    stat = csv_path.stat()
    cache_key = (str(csv_path.resolve()), stat.st_mtime_ns, stat.st_size)
    with _CSV_CACHE_LOCK:
        cached = _CSV_CACHE.get(cache_key)
        if cached is not None:
            return [dict(row) for row in cached]

    rows: list[dict[str, str]] = []
    with csv_path.open(mode="r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            logger.warning("CSV file has no header row: %s", csv_path)
            return rows
        for raw_row in reader:
            cleaned_row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                cleaned_key = key.strip()
                cleaned_value = value.strip() if isinstance(value, str) else ""
                cleaned_row[cleaned_key] = cleaned_value
            rows.append(cleaned_row)

    frozen_rows = tuple(dict(row) for row in rows)
    with _CSV_CACHE_LOCK:
        path_key = cache_key[0]
        stale_keys = [key for key in _CSV_CACHE if key[0] == path_key and key != cache_key]
        for stale_key in stale_keys:
            del _CSV_CACHE[stale_key]
        _CSV_CACHE[cache_key] = frozen_rows

    logger.debug("Loaded %d rows from %s", len(rows), csv_path.name)
    return [dict(row) for row in frozen_rows]


def _find_row(
    rows: list[dict[str, str]],
    candidate_keys: tuple[str, ...],
    target_value: str,
) -> dict[str, str] | None:
    """Find the first row whose value for any candidate key matches ``target_value``."""
    normalized_target = target_value.strip().lower()
    if not normalized_target:
        return None
    for row in rows:
        for key in candidate_keys:
            cell = row.get(key)
            if cell is not None and cell.strip().lower() == normalized_target:
                return row
    return None


def _filter_rows(
    rows: list[dict[str, str]],
    candidate_keys: tuple[str, ...],
    target_value: str,
) -> list[dict[str, str]]:
    """Filter all rows matching ``target_value`` on any candidate key."""
    normalized_target = target_value.strip().lower()
    if not normalized_target:
        return []
    matched: list[dict[str, str]] = []
    for row in rows:
        for key in candidate_keys:
            cell = row.get(key)
            if cell is not None and cell.strip().lower() == normalized_target:
                matched.append(row)
                break
    return matched


def _get_first_value(
    row: dict[str, str],
    candidate_keys: tuple[str, ...],
    default: str = "",
) -> str:
    """Return the first non-empty string value from ``row`` for ``candidate_keys``."""
    for key in candidate_keys:
        value = row.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _to_float(raw_value: Any, default: float = 0.0) -> float:
    """Safely parse a numeric string (stripping currency symbols/commas) to float."""
    if raw_value is None:
        return default
    if isinstance(raw_value, (int, float)):
        result = float(raw_value)
        return default if (math.isnan(result) or math.isinf(result)) else result
    text = str(raw_value).strip()
    if not text or text.lower() in {"na", "n/a", "null", "none", "nan", "-"}:
        return default
    cleaned = (
        text.replace(",", "")
        .replace("₹", "")
        .replace("$", "")
        .replace("INR", "")
        .replace("USD", "")
        .replace("%", "")
        .strip()
    )
    try:
        parsed = float(cleaned)
    except ValueError:
        logger.warning("Could not parse float from value %r; using default %s", raw_value, default)
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _to_int(raw_value: Any, default: int = 0) -> int:
    """Safely parse an integer value from a string or number."""
    if raw_value is None:
        return default
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    return int(round(_to_float(raw_value, default=float(default))))


def _calculate_monthly_emi(
    principal_inr: float,
    annual_rate_pct: float,
    tenor_months: int,
) -> float:
    """Compute standard reducing-balance monthly EMI in INR."""
    if principal_inr <= 0.0 or tenor_months <= 0:
        return 0.0
    monthly_rate = (annual_rate_pct / 100.0) / 12.0
    if monthly_rate <= 0.0:
        return principal_inr / float(tenor_months)
    factor = (1.0 + monthly_rate) ** tenor_months
    denominator = factor - 1.0
    if denominator <= 0.0:
        return principal_inr / float(tenor_months)
    return principal_inr * monthly_rate * factor / denominator


def _parse_policy_limits(rows: list[dict[str, str]]) -> dict[str, float]:
    """Normalize ``policy_limits.csv`` whether stored as key-value pairs or columns."""
    limits: dict[str, float] = {}
    if not rows:
        return limits

    sample_keys = {k.lower() for k in rows[0].keys()}
    is_kv_layout = bool(
        sample_keys.intersection({"policy_name", "limit_name", "rule_name", "parameter", "metric", "key"})
    ) and bool(sample_keys.intersection({"value", "limit_value", "threshold", "max_value", "min_value"}))

    if is_kv_layout:
        for row in rows:
            name = _get_first_value(
                row,
                ("policy_name", "limit_name", "rule_name", "parameter", "metric", "key"),
            ).lower()
            if not name:
                continue
            val_str = _get_first_value(
                row,
                ("value", "limit_value", "threshold", "max_value", "min_value"),
            )
            if val_str:
                limits[name] = _to_float(val_str)
    else:
        for key, value in rows[0].items():
            if value != "":
                limits[key.lower()] = _to_float(value)

    return limits


def _extract_underwriting_defaults(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Extract reference underwriting parameters/formulas from ``underwriting_reference.csv``."""
    ref: dict[str, Any] = {}
    if not rows:
        return ref

    sample_keys = {k.lower() for k in rows[0].keys()}
    is_kv_layout = bool(
        sample_keys.intersection({"parameter", "metric", "rule", "formula_name", "key"})
    )
    if is_kv_layout:
        for row in rows:
            key = _get_first_value(
                row,
                ("parameter", "metric", "rule", "formula_name", "key"),
            ).lower()
            if not key:
                continue
            val = _get_first_value(row, ("value", "formula", "default_value", "description"))
            if val:
                ref[key] = val
    else:
        for key, value in rows[0].items():
            if value != "":
                ref[key.lower()] = value
    return ref


@tool
def calculate_dscr_and_ratios(application_id: str) -> dict[str, Any]:
    """Calculate exact DSCR, financial ratios, and policy breach flags for an SME loan application.

    Use this tool when:
    - Evaluating a specific loan application's debt service coverage ratio (DSCR),
      revenue variance, tenor eligibility, collateral adequacy, and sector exposure cap.
    - Checking deterministic underwriting policy rules before making a credit decision.

    Do NOT use this tool when:
    - You only have an applicant ID without an application ID (look up bureau or bank
      statements directly with ``lookup_credit_bureau`` or ``analyze_bank_statements``).
    - You need standalone FX conversion or EMI simulation without an application record
      (use ``convert_fx_and_interest`` instead).

    Args:
        application_id: Unique identifier of the loan application (e.g. ``"APP-1001"``).

    Returns:
        Structured dictionary containing computed DSCR, debt service figures,
        revenue variance percentage, boolean policy check results, triggered policy
        breach codes (e.g. ``DOC_MISMATCH``, ``FRAUD_HOLD``, ``TENOR_ABOVE_LIMIT``,
        ``COLLATERAL_SHORTFALL``, ``SECTOR_CAP``), or an ``error`` message if the
        application or applicant ID cannot be found.
    """
    logger.info("Running calculate_dscr_and_ratios for application_id=%s", application_id)
    if not isinstance(application_id, str) or not application_id.strip():
        return {"error": "Application ID not found"}

    try:
        applications = _load_csv_rows("loan_applications.csv")
        applicants = _load_csv_rows("applicants.csv")
        sector_policies = _load_csv_rows("sector_policies.csv")
        policy_limits_rows = _load_csv_rows("policy_limits.csv")
        risk_grades = _load_csv_rows("risk_grades.csv")
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    underwriting_ref: dict[str, Any] = {}
    try:
        underwriting_ref = _extract_underwriting_defaults(_load_csv_rows("underwriting_reference.csv"))
    except FileNotFoundError:
        logger.warning("underwriting_reference.csv not found; using standard EMI derivation")

    app_row = _find_row(applications, ("application_id", "app_id", "id"), application_id)
    if app_row is None:
        logger.warning("Application ID not found: %s", application_id)
        return {"error": "Application ID not found", "application_id": application_id}

    applicant_id = _get_first_value(app_row, ("applicant_id", "borrower_id", "entity_id"))
    applicant_row = _find_row(applicants, ("applicant_id", "borrower_id", "entity_id", "id"), applicant_id)
    if applicant_row is None:
        logger.warning("Applicant ID not found for application %s: %s", application_id, applicant_id)
        return {"error": "Applicant ID not found", "applicant_id": applicant_id}

    sector = _get_first_value(
        app_row,
        ("sector", "industry_sector", "business_sector"),
        default=_get_first_value(applicant_row, ("sector", "industry_sector", "business_sector")),
    )
    risk_grade = _get_first_value(
        app_row,
        ("risk_grade", "internal_risk_grade", "grade"),
        default=_get_first_value(applicant_row, ("risk_grade", "internal_risk_grade", "grade")),
    )

    sector_row = _find_row(sector_policies, ("sector", "industry_sector", "sector_name"), sector)
    grade_row = _find_row(risk_grades, ("risk_grade", "grade", "grade_code"), risk_grade)
    policy_limits = _parse_policy_limits(policy_limits_rows)

    ebitda_inr = _to_float(
        _get_first_value(app_row, ("ebitda_inr", "annual_ebitda_inr", "ebitda"))
        or _get_first_value(applicant_row, ("ebitda_inr", "annual_ebitda_inr", "ebitda"))
    )
    existing_annual_debt_service_inr = _to_float(
        _get_first_value(
            app_row,
            ("existing_annual_debt_service_inr", "existing_debt_service_inr", "existing_annual_obligations_inr"),
        )
        or _get_first_value(
            applicant_row,
            ("existing_annual_debt_service_inr", "existing_debt_service_inr", "existing_annual_obligations_inr"),
        )
    )

    requested_amount_inr = _to_float(
        _get_first_value(app_row, ("requested_amount_inr", "loan_amount_inr", "principal_inr", "amount_inr"))
    )
    requested_tenor_months = _to_int(
        _get_first_value(app_row, ("requested_tenor_months", "tenor_months", "tenure_months"))
    )

    indicative_rate_pct = _to_float(
        _get_first_value(app_row, ("indicative_rate_pct", "interest_rate_pct", "rate_pct"))
        or (
            _get_first_value(
                grade_row,
                ("indicative_rate_pct", "interest_rate_pct", "rate_pct", "base_rate_pct"),
            )
            if grade_row
            else ""
        )
        or str(underwriting_ref.get("default_indicative_rate_pct", underwriting_ref.get("indicative_rate_pct", "0")))
    )

    explicit_proposed_debt_service = _to_float(
        _get_first_value(app_row, ("proposed_annual_debt_service_inr", "proposed_debt_service_inr"))
    )
    derived_monthly_emi = _calculate_monthly_emi(
        requested_amount_inr,
        indicative_rate_pct,
        requested_tenor_months,
    )
    derived_proposed_annual_debt_service = derived_monthly_emi * 12.0

    if explicit_proposed_debt_service > 0.0:
        proposed_annual_debt_service_inr = explicit_proposed_debt_service
        proposed_debt_service_source = "loan_applications.proposed_annual_debt_service_inr"
    else:
        proposed_annual_debt_service_inr = derived_proposed_annual_debt_service
        proposed_debt_service_source = "underwriting_reference_emi_formula"

    total_annual_debt_service_inr = existing_annual_debt_service_inr + proposed_annual_debt_service_inr
    dscr: float | None = (
        round(ebitda_inr / total_annual_debt_service_inr, 4)
        if total_annual_debt_service_inr > 0.0
        else None
    )

    dscr_floor = _to_float(
        _get_first_value(sector_row, ("dscr_floor", "min_dscr")) if sector_row else "",
        default=policy_limits.get("dscr_floor", policy_limits.get("min_dscr", 1.2)),
    )
    meets_dscr_floor = (dscr >= dscr_floor) if dscr is not None else False

    # Revenue variance check
    raw_variance = _get_first_value(
        app_row,
        ("revenue_variance_pct", "gst_revenue_variance_pct", "turnover_variance_pct"),
    ) or _get_first_value(
        applicant_row,
        ("revenue_variance_pct", "gst_revenue_variance_pct", "turnover_variance_pct"),
    )
    if raw_variance != "":
        revenue_variance_pct = round(abs(_to_float(raw_variance)), 2)
    else:
        reported_rev = _to_float(
            _get_first_value(app_row, ("reported_revenue_inr", "declared_revenue_inr", "annual_revenue_inr"))
            or _get_first_value(applicant_row, ("reported_revenue_inr", "declared_revenue_inr", "annual_revenue_inr"))
        )
        verified_rev = _to_float(
            _get_first_value(app_row, ("verified_revenue_inr", "gst_revenue_inr", "audited_revenue_inr"))
            or _get_first_value(applicant_row, ("verified_revenue_inr", "gst_revenue_inr", "audited_revenue_inr"))
        )
        if verified_rev > 0.0:
            revenue_variance_pct = round(abs(reported_rev - verified_rev) / verified_rev * 100.0, 2)
        else:
            revenue_variance_pct = 0.0

    doc_mismatch_threshold = policy_limits.get(
        "doc_mismatch_threshold_pct",
        float(getattr(settings, "REVENUE_MISMATCH_PCT", REVENUE_MISMATCH_PCT)),
    )
    fraud_hold_threshold = policy_limits.get(
        "fraud_hold_threshold_pct",
        float(getattr(settings, "FRAUD_HOLD_VARIANCE_PCT", FRAUD_HOLD_VARIANCE_PCT)),
    )
    doc_mismatch = revenue_variance_pct > doc_mismatch_threshold
    fraud_hold = revenue_variance_pct > fraud_hold_threshold

    # Tenor limit check
    candidate_tenor_limits: list[int] = []
    if sector_row:
        sector_max_tenor = _to_int(_get_first_value(sector_row, ("max_tenor_months", "tenor_limit_months")))
        if sector_max_tenor > 0:
            candidate_tenor_limits.append(sector_max_tenor)
    if grade_row:
        grade_max_tenor = _to_int(_get_first_value(grade_row, ("max_tenor_months", "tenor_limit_months")))
        if grade_max_tenor > 0:
            candidate_tenor_limits.append(grade_max_tenor)
    policy_max_tenor = int(policy_limits.get("max_tenor_months", policy_limits.get("tenor_limit_months", 0.0)))
    if policy_max_tenor > 0:
        candidate_tenor_limits.append(policy_max_tenor)
    max_tenor_months = min(candidate_tenor_limits) if candidate_tenor_limits else 0
    tenor_above_limit = max_tenor_months > 0 and requested_tenor_months > max_tenor_months

    # Collateral shortfall check
    collateral_value_inr = _to_float(
        _get_first_value(app_row, ("collateral_value_inr", "security_value_inr", "collateral_inr"))
    )
    collateral_coverage_ratio = (
        round(collateral_value_inr / requested_amount_inr, 4)
        if requested_amount_inr > 0.0
        else 0.0
    )
    min_collateral_coverage = _to_float(
        _get_first_value(
            sector_row,
            ("min_collateral_coverage_ratio", "min_collateral_coverage", "collateral_coverage_floor"),
        )
        if sector_row
        else "",
        default=_to_float(
            _get_first_value(
                grade_row,
                ("min_collateral_coverage_ratio", "min_collateral_coverage", "collateral_coverage_floor"),
            )
            if grade_row
            else "",
            default=policy_limits.get(
                "min_collateral_coverage_ratio",
                policy_limits.get("min_collateral_coverage", 1.0),
            ),
        ),
    )
    collateral_shortfall = requested_amount_inr > 0.0 and collateral_coverage_ratio < min_collateral_coverage

    # Sector cap check
    sector_cap_inr = _to_float(
        _get_first_value(
            sector_row,
            ("sector_cap_inr", "max_sector_exposure_inr", "max_loan_amount_inr", "exposure_cap_inr"),
        )
        if sector_row
        else "",
        default=policy_limits.get("sector_cap_inr", policy_limits.get("max_sector_exposure_inr", 0.0)),
    )
    existing_sector_exposure_inr = _to_float(
        _get_first_value(
            sector_row,
            ("current_exposure_inr", "existing_sector_exposure_inr", "utilized_exposure_inr"),
        )
        if sector_row
        else ""
    )
    projected_sector_exposure_inr = existing_sector_exposure_inr + requested_amount_inr
    sector_cap_breached = sector_cap_inr > 0.0 and projected_sector_exposure_inr > sector_cap_inr

    policy_flags: list[str] = []
    if not meets_dscr_floor:
        policy_flags.append("DSCR_BELOW_FLOOR")
    if doc_mismatch:
        policy_flags.append("DOC_MISMATCH")
    if fraud_hold:
        policy_flags.append("FRAUD_HOLD")
    if tenor_above_limit:
        policy_flags.append("TENOR_ABOVE_LIMIT")
    if collateral_shortfall:
        policy_flags.append("COLLATERAL_SHORTFALL")
    if sector_cap_breached:
        policy_flags.append("SECTOR_CAP")

    return {
        "application_id": _get_first_value(app_row, ("application_id", "app_id", "id"), application_id),
        "applicant_id": applicant_id,
        "sector": sector,
        "risk_grade": risk_grade,
        "ebitda_inr": round(ebitda_inr, 2),
        "existing_annual_debt_service_inr": round(existing_annual_debt_service_inr, 2),
        "proposed_annual_debt_service_inr": round(proposed_annual_debt_service_inr, 2),
        "total_annual_debt_service_inr": round(total_annual_debt_service_inr, 2),
        "proposed_debt_service_source": proposed_debt_service_source,
        "indicative_rate_pct": round(indicative_rate_pct, 4),
        "dscr": dscr,
        "dscr_floor": round(dscr_floor, 4),
        "meets_dscr_floor": meets_dscr_floor,
        "revenue_variance_pct": revenue_variance_pct,
        "doc_mismatch": doc_mismatch,
        "fraud_hold": fraud_hold,
        "requested_tenor_months": requested_tenor_months,
        "max_tenor_months": max_tenor_months,
        "tenor_above_limit": tenor_above_limit,
        "requested_amount_inr": round(requested_amount_inr, 2),
        "collateral_value_inr": round(collateral_value_inr, 2),
        "collateral_coverage_ratio": collateral_coverage_ratio,
        "min_collateral_coverage_ratio": round(min_collateral_coverage, 4),
        "collateral_shortfall": collateral_shortfall,
        "projected_sector_exposure_inr": round(projected_sector_exposure_inr, 2),
        "sector_cap_inr": round(sector_cap_inr, 2),
        "sector_cap_breached": sector_cap_breached,
        "policy_flags": policy_flags,
        "underwriting_formula_reference": underwriting_ref.get("dscr_formula", "ebitda_inr / (existing + proposed)"),
    }


@tool
def convert_fx_and_interest(
    usd_amount: float,
    principal_inr: float,
    annual_rate_pct: float,
    tenor_months: int,
) -> dict[str, Any]:
    """Convert USD receivables to INR using reference FX rates and compute loan EMI debt service.

    Use this tool when:
    - An SME applicant has export receivables or cash inflows denominated in USD that
      must be converted to INR at the official reference rate.
    - You need exact monthly EMI, annual debt service, and total interest outflow for
      a given INR principal, annual interest rate percentage, and tenor in months.

    Do NOT use this tool when:
    - You want to run full application-level policy checks against sector DSCR floors
      or collateral caps (use ``calculate_dscr_and_ratios`` instead).

    Args:
        usd_amount: Receivables or cashflow amount in USD (must be >= 0).
        principal_inr: Proposed loan principal in INR (must be >= 0).
        annual_rate_pct: Annual nominal interest rate in percent, e.g. ``11.5`` (must be >= 0).
        tenor_months: Repayment tenor in months (must be > 0).

    Returns:
        Structured dictionary with the applied USD-to-INR rate, converted INR
        receivables, monthly EMI, annual debt service, total interest payable, and
        total repayment amount, or an ``error`` field if inputs or rate data are invalid.
    """
    logger.info(
        "Running convert_fx_and_interest usd_amount=%.2f principal_inr=%.2f rate=%.4f tenor=%d",
        usd_amount,
        principal_inr,
        annual_rate_pct,
        tenor_months,
    )
    if usd_amount < 0.0 or principal_inr < 0.0 or annual_rate_pct < 0.0 or tenor_months <= 0:
        return {
            "error": (
                "Invalid numeric parameters: usd_amount, principal_inr, and "
                "annual_rate_pct must be >= 0, and tenor_months must be > 0."
            )
        }

    try:
        rates_rows = _load_csv_rows("reference_rates.csv")
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    usd_inr_rate = 0.0
    rate_as_of = ""
    for row in rates_rows:
        pair = _get_first_value(row, ("currency_pair", "pair", "rate_name", "code")).upper()
        from_ccy = _get_first_value(row, ("from_currency", "base_currency", "currency")).upper()
        to_ccy = _get_first_value(row, ("to_currency", "quote_currency", "target_currency")).upper()
        if pair in {"USD/INR", "USD_INR", "USDINR", "USD-INR"} or (from_ccy == "USD" and to_ccy in {"INR", ""}):
            usd_inr_rate = _to_float(
                _get_first_value(row, ("rate", "exchange_rate", "fx_rate", "spot_rate", "value"))
            )
            rate_as_of = _get_first_value(row, ("as_of_date", "effective_date", "date", "updated_at"))
            if usd_inr_rate > 0.0:
                break
        if "usd_inr_rate" in row or "usd_to_inr" in row:
            usd_inr_rate = _to_float(_get_first_value(row, ("usd_inr_rate", "usd_to_inr")))
            rate_as_of = _get_first_value(row, ("as_of_date", "effective_date", "date"))
            if usd_inr_rate > 0.0:
                break

    if usd_inr_rate <= 0.0:
        logger.error("Valid USD-to-INR exchange rate missing from reference_rates.csv")
        return {"error": "USD-to-INR rate not found in reference_rates.csv"}

    converted_receivables_inr = usd_amount * usd_inr_rate
    monthly_emi_inr = _calculate_monthly_emi(principal_inr, annual_rate_pct, tenor_months)
    annual_debt_service_inr = monthly_emi_inr * 12.0
    total_repayment_inr = monthly_emi_inr * float(tenor_months)
    total_interest_inr = max(0.0, total_repayment_inr - principal_inr)

    return {
        "usd_amount": round(usd_amount, 2),
        "usd_inr_rate": round(usd_inr_rate, 4),
        "rate_as_of": rate_as_of,
        "converted_receivables_inr": round(converted_receivables_inr, 2),
        "principal_inr": round(principal_inr, 2),
        "annual_rate_pct": round(annual_rate_pct, 4),
        "tenor_months": tenor_months,
        "monthly_emi_inr": round(monthly_emi_inr, 2),
        "annual_debt_service_inr": round(annual_debt_service_inr, 2),
        "total_interest_inr": round(total_interest_inr, 2),
        "total_repayment_inr": round(total_repayment_inr, 2),
    }


@tool
def lookup_credit_bureau(applicant_id: str) -> dict[str, Any]:
    """Look up an SME applicant's external credit bureau report metrics.

    Use this tool when:
    - Retrieving ``bureau_score``, recent credit enquiries (``enquiries_last_6m``),
      30+ DPD delinquency count (``dpd_30_last_12m``), 90+ DPD history (``dpd_90_ever``),
      and written-off amounts (``written_off_amount_inr``) for a specific applicant.

    Do NOT use this tool when:
    - Analyzing cash-flow deposits, withdrawals, or bounced cheques from bank accounts
      (use ``analyze_bank_statements`` instead).
    - Computing application DSCR or sector policy limits (use ``calculate_dscr_and_ratios``).

    Args:
        applicant_id: Unique identifier of the SME applicant (e.g. ``"APL-2001"``).

    Returns:
        Structured dictionary of credit bureau metrics and risk indicators, or
        ``{"error": "Applicant ID not found"}`` if the applicant has no bureau record.
    """
    logger.info("Running lookup_credit_bureau for applicant_id=%s", applicant_id)
    if not isinstance(applicant_id, str) or not applicant_id.strip():
        return {"error": "Applicant ID not found"}

    try:
        bureau_rows = _load_csv_rows("bureau_reports.csv")
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    row = _find_row(bureau_rows, ("applicant_id", "borrower_id", "entity_id", "id"), applicant_id)
    if row is None:
        logger.warning("Applicant ID not found in bureau_reports.csv: %s", applicant_id)
        return {"error": "Applicant ID not found"}

    bureau_score = _to_int(_get_first_value(row, ("bureau_score", "credit_score", "cibil_score", "score")))
    enquiries_last_6m = _to_int(_get_first_value(row, ("enquiries_last_6m", "inquiries_last_6m")))
    dpd_30_last_12m = _to_int(_get_first_value(row, ("dpd_30_last_12m", "dpd_30_12m")))
    dpd_90_ever = _to_int(_get_first_value(row, ("dpd_90_ever", "dpd_90_plus_ever")))
    written_off_amount_inr = _to_float(
        _get_first_value(row, ("written_off_amount_inr", "write_off_amount_inr", "written_off_inr"))
    )

    return {
        "applicant_id": _get_first_value(row, ("applicant_id", "borrower_id", "entity_id", "id"), applicant_id),
        "bureau_score": bureau_score,
        "enquiries_last_6m": enquiries_last_6m,
        "dpd_30_last_12m": dpd_30_last_12m,
        "dpd_90_ever": dpd_90_ever,
        "written_off_amount_inr": round(written_off_amount_inr, 2),
        "has_severe_delinquency": dpd_90_ever > 0 or written_off_amount_inr > 0.0,
        "has_recent_delinquency": dpd_30_last_12m > 0,
        "high_enquiry_velocity": enquiries_last_6m > 6,
    }


@tool
def analyze_bank_statements(applicant_id: str) -> dict[str, Any]:
    """Analyze monthly bank statement cashflows and bounced instruments for an applicant.

    Use this tool when:
    - Computing an SME applicant's average monthly inflows, average monthly outflows,
      average closing balance, and total bounced instruments across available bank
      statement periods.

    Do NOT use this tool when:
    - Checking external credit bureau scores or DPD history (use ``lookup_credit_bureau``).
    - Computing application DSCR against sector floors (use ``calculate_dscr_and_ratios``).

    Args:
        applicant_id: Unique identifier of the SME applicant (e.g. ``"APL-2001"``).

    Returns:
        Structured dictionary summarizing monthly inflow/outflow averages, closing
        balances, and total ``bounced_instruments``, or
        ``{"error": "Applicant ID not found"}`` if no statements exist for the applicant.
    """
    logger.info("Running analyze_bank_statements for applicant_id=%s", applicant_id)
    if not isinstance(applicant_id, str) or not applicant_id.strip():
        return {"error": "Applicant ID not found"}

    try:
        statement_rows = _load_csv_rows("bank_statements.csv")
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    matched_rows = _filter_rows(
        statement_rows,
        ("applicant_id", "borrower_id", "entity_id", "id"),
        applicant_id,
    )
    if not matched_rows:
        logger.warning("Applicant ID not found in bank_statements.csv: %s", applicant_id)
        return {"error": "Applicant ID not found"}

    inflows: list[float] = []
    outflows: list[float] = []
    closing_balances: list[float] = []
    total_bounced_instruments = 0

    for row in matched_rows:
        inflow = _to_float(
            _get_first_value(
                row,
                ("monthly_inflow_inr", "inflow_inr", "total_credits_inr", "credits_inr", "inflow"),
            )
        )
        outflow = _to_float(
            _get_first_value(
                row,
                ("monthly_outflow_inr", "outflow_inr", "total_debits_inr", "debits_inr", "outflow"),
            )
        )
        closing_balance = _to_float(
            _get_first_value(
                row,
                ("closing_balance_inr", "ending_balance_inr", "month_end_balance_inr", "closing_balance"),
            )
        )
        bounced = _to_int(
            _get_first_value(
                row,
                ("bounced_instruments", "cheque_bounces", "bounced_count", "iw_ow_bounces"),
            )
        )
        inflows.append(inflow)
        outflows.append(outflow)
        closing_balances.append(closing_balance)
        total_bounced_instruments += bounced

    months_analyzed = len(matched_rows)
    avg_monthly_inflow = sum(inflows) / float(months_analyzed)
    avg_monthly_outflow = sum(outflows) / float(months_analyzed)
    avg_closing_balance = sum(closing_balances) / float(months_analyzed)
    min_closing_balance = min(closing_balances)
    net_monthly_cashflow = avg_monthly_inflow - avg_monthly_outflow
    outflow_to_inflow_ratio = (
        round(avg_monthly_outflow / avg_monthly_inflow, 4)
        if avg_monthly_inflow > 0.0
        else None
    )

    canonical_applicant_id = _get_first_value(
        matched_rows[0],
        ("applicant_id", "borrower_id", "entity_id", "id"),
        applicant_id,
    )

    return {
        "applicant_id": canonical_applicant_id,
        "months_analyzed": months_analyzed,
        "avg_monthly_inflow_inr": round(avg_monthly_inflow, 2),
        "avg_monthly_outflow_inr": round(avg_monthly_outflow, 2),
        "net_monthly_cashflow_inr": round(net_monthly_cashflow, 2),
        "avg_closing_balance_inr": round(avg_closing_balance, 2),
        "min_closing_balance_inr": round(min_closing_balance, 2),
        "total_bounced_instruments": total_bounced_instruments,
        "outflow_to_inflow_ratio": outflow_to_inflow_ratio,
    }


@tool
def escalate_to_human(application_id: str, reason: str) -> dict[str, Any]:
    """Flag a loan application for human credit officer review with required signatory level.

    Use this tool when:
    - Policy breaches (e.g., ``FRAUD_HOLD``, ``DOC_MISMATCH``, ``DSCR_BELOW_FLOOR``,
      ``COLLATERAL_SHORTFALL``, ``SECTOR_CAP``, ``TENOR_ABOVE_LIMIT``), severe bureau
      delinquencies, or high bounced-instrument counts require human credit committee
      approval or manual override.

    Do NOT use this tool when:
    - You have not yet gathered the quantitative metrics needed to justify escalation
      (run ``calculate_dscr_and_ratios``, ``lookup_credit_bureau``, or
      ``analyze_bank_statements`` first).

    Args:
        application_id: Unique identifier of the loan application to escalate.
        reason: Clear, factual explanation of the policy breach or risk trigger.

    Returns:
        Structured dictionary confirming human review escalation, required signatory
        level from ``approval_authorities.csv``, and escalation metadata, or an
        ``error`` dictionary if the application ID is invalid or reason is empty.
    """
    logger.info("Running escalate_to_human for application_id=%s", application_id)
    if not isinstance(application_id, str) or not application_id.strip():
        return {"error": "Application ID not found"}
    if not isinstance(reason, str) or not reason.strip():
        return {"error": "Escalation reason must be a non-empty string"}

    try:
        applications = _load_csv_rows("loan_applications.csv")
        authorities = _load_csv_rows("approval_authorities.csv")
    except FileNotFoundError as exc:
        return {"error": str(exc)}

    app_row = _find_row(applications, ("application_id", "app_id", "id"), application_id)
    if app_row is None:
        logger.warning("Application ID not found for escalation: %s", application_id)
        return {"error": "Application ID not found", "application_id": application_id}

    applicant_id = _get_first_value(app_row, ("applicant_id", "borrower_id", "entity_id"))
    requested_amount_inr = _to_float(
        _get_first_value(app_row, ("requested_amount_inr", "loan_amount_inr", "principal_inr", "amount_inr"))
    )
    risk_grade = _get_first_value(app_row, ("risk_grade", "internal_risk_grade", "grade"))

    matched_authority: dict[str, str] | None = None
    fallback_highest_authority: dict[str, str] | None = None
    highest_min_amount = -1.0

    for auth_row in authorities:
        min_amt = _to_float(
            _get_first_value(auth_row, ("min_amount_inr", "min_loan_inr", "min_exposure_inr")),
            default=0.0,
        )
        max_amt_raw = _get_first_value(
            auth_row,
            ("max_amount_inr", "max_loan_inr", "max_exposure_inr"),
        )
        max_amt = _to_float(max_amt_raw, default=float("inf")) if max_amt_raw != "" else float("inf")
        if max_amt <= 0.0 and max_amt_raw != "":
            max_amt = float("inf")

        if min_amt >= highest_min_amount:
            highest_min_amount = min_amt
            fallback_highest_authority = auth_row

        auth_grade = _get_first_value(auth_row, ("risk_grade", "grade"))
        grade_matches = not auth_grade or not risk_grade or auth_grade.lower() == risk_grade.lower()
        if grade_matches and min_amt <= requested_amount_inr <= max_amt:
            matched_authority = auth_row
            break

    if matched_authority is None:
        matched_authority = fallback_highest_authority

    signatory_level = (
        _get_first_value(
            matched_authority,
            ("signatory_level", "authority_level", "required_signatory", "approver_role", "role"),
            default="Credit Committee",
        )
        if matched_authority is not None
        else "Credit Committee"
    )
    sla_hours = (
        _to_int(_get_first_value(matched_authority, ("sla_hours", "review_sla_hours")), default=24)
        if matched_authority is not None
        else 24
    )

    clean_reason = reason.strip()
    escalated_at = datetime.now(timezone.utc).isoformat()
    logger.warning(
        "Application %s (applicant=%s, amount_inr=%.2f) escalated to %s. Reason: %s",
        application_id,
        applicant_id,
        requested_amount_inr,
        signatory_level,
        clean_reason,
    )

    return {
        "application_id": _get_first_value(app_row, ("application_id", "app_id", "id"), application_id),
        "applicant_id": applicant_id,
        "requested_amount_inr": round(requested_amount_inr, 2),
        "risk_grade": risk_grade,
        "escalation_status": "ESCALATED_FOR_HUMAN_REVIEW",
        "requires_human_review": True,
        "required_signatory_level": signatory_level,
        "review_sla_hours": sla_hours,
        "reason": clean_reason,
        "escalated_at": escalated_at,
    }

