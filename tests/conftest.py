"""Shared pytest fixtures and CSV dataset builders for WealthPilot tool tests."""

from __future__ import annotations

import csv
import dataclasses
import importlib
import importlib.util
import logging
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

logger = logging.getLogger(__name__)


def _ensure_settings_module(default_data_dir: Path) -> None:
    """Provide a fallback ``src.config`` module when running tests standalone."""
    for mod_name in ("src.config", "config", "src.settings"):
        if mod_name in sys.modules:
            return
        try:
            if importlib.util.find_spec(mod_name) is not None:
                return
        except (ModuleNotFoundError, ValueError):
            continue

    if "src" not in sys.modules:
        src_pkg = types.ModuleType("src")
        src_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "src")]
        sys.modules["src"] = src_pkg

    config_mod = types.ModuleType("src.config")
    config_mod.settings = SimpleNamespace(DATA_DIR=default_data_dir)  # type: ignore[attr-defined]
    sys.modules["src.config"] = config_mod
    logger.debug("Registered fallback src.config module for standalone test execution")


def _load_tools_module(default_data_dir: Path) -> ModuleType:
    """Import ``src.tools`` cleanly with fallback settings support if needed."""
    _ensure_settings_module(default_data_dir)
    return importlib.import_module("src.tools")


def invoke_tool(tool_obj: Any, **kwargs: Any) -> dict[str, Any]:
    """Invoke a LangChain ``@tool`` object or underlying callable consistently."""
    if hasattr(tool_obj, "invoke") and callable(tool_obj.invoke):
        result: dict[str, Any] = tool_obj.invoke(kwargs)
        return result
    if hasattr(tool_obj, "func") and callable(tool_obj.func):
        result = tool_obj.func(**kwargs)
        return result
    result = tool_obj(**kwargs)
    return result


def write_csv(
    directory: Path,
    filename: str,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> Path:
    """Write rows to a CSV file inside ``directory`` with UTF-8 encoding."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    with target.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return target


def _with_data_dir(settings_obj: Any, data_dir: Path) -> Any:
    """Return a settings instance with ``DATA_DIR`` set to ``data_dir`` without mutating frozen objects."""
    if dataclasses.is_dataclass(settings_obj) and not isinstance(settings_obj, type):
        return dataclasses.replace(settings_obj, DATA_DIR=data_dir)
    if hasattr(settings_obj, "model_copy") and callable(settings_obj.model_copy):
        return settings_obj.model_copy(update={"DATA_DIR": data_dir})
    return SimpleNamespace(DATA_DIR=data_dir)


@pytest.fixture
def tools_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Return the imported ``src.tools`` module bound to an isolated ``DATA_DIR``."""
    module = _load_tools_module(tmp_path)
    patched_settings = _with_data_dir(module.settings, tmp_path)
    monkeypatch.setattr(module, "settings", patched_settings)
    return module


@pytest.fixture
def mock_api_dir(tmp_path: Path, tools_module: ModuleType) -> Path:
    """Create and return ``tmp_path / 'mock_api'`` populated with baseline CSV fixtures."""
    api_dir = tmp_path / "mock_api"
    api_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        api_dir,
        "applicants.csv",
        ["applicant_id", "company_name", "sector", "risk_grade", "annual_revenue_inr"],
        [
            {
                "applicant_id": "APL-101",
                "company_name": "Apex Precision Components Pvt Ltd",
                "sector": "Manufacturing",
                "risk_grade": "A",
                "annual_revenue_inr": "120000000",
            },
            {
                "applicant_id": "APL-102",
                "company_name": "Horizon Logistics LLP",
                "sector": "Logistics",
                "risk_grade": "B",
                "annual_revenue_inr": "80000000",
            },
            {
                "applicant_id": "APL-103",
                "company_name": "Deccan Textiles Ltd",
                "sector": "Textiles",
                "risk_grade": "C",
                "annual_revenue_inr": "50000000",
            },
        ],
    )

    write_csv(
        api_dir,
        "loan_applications.csv",
        [
            "application_id",
            "applicant_id",
            "sector",
            "risk_grade",
            "requested_amount_inr",
            "requested_tenor_months",
            "indicative_rate_pct",
            "ebitda_inr",
            "existing_annual_debt_service_inr",
            "proposed_annual_debt_service_inr",
            "collateral_value_inr",
            "revenue_variance_pct",
            "reported_revenue_inr",
            "verified_revenue_inr",
        ],
        [
            {
                "application_id": "APP-1001",
                "applicant_id": "APL-101",
                "sector": "Manufacturing",
                "risk_grade": "A",
                "requested_amount_inr": "10000000",
                "requested_tenor_months": "48",
                "indicative_rate_pct": "11.0",
                "ebitda_inr": "9000000",
                "existing_annual_debt_service_inr": "1500000",
                "proposed_annual_debt_service_inr": "3100000",
                "collateral_value_inr": "13000000",
                "revenue_variance_pct": "4.5",
                "reported_revenue_inr": "120000000",
                "verified_revenue_inr": "115000000",
            },
            {
                "application_id": "APP-1002",
                "applicant_id": "APL-102",
                "sector": "Logistics",
                "risk_grade": "B",
                "requested_amount_inr": "15000000",
                "requested_tenor_months": "72",
                "indicative_rate_pct": "",
                "ebitda_inr": "5000000",
                "existing_annual_debt_service_inr": "2000000",
                "proposed_annual_debt_service_inr": "",
                "collateral_value_inr": "10000000",
                "revenue_variance_pct": "15.0",
                "reported_revenue_inr": "80000000",
                "verified_revenue_inr": "69565217",
            },
            {
                "application_id": "APP-1003",
                "applicant_id": "APL-103",
                "sector": "Textiles",
                "risk_grade": "C",
                "requested_amount_inr": "60000000",
                "requested_tenor_months": "36",
                "indicative_rate_pct": "13.5",
                "ebitda_inr": "18000000",
                "existing_annual_debt_service_inr": "4000000",
                "proposed_annual_debt_service_inr": "24000000",
                "collateral_value_inr": "45000000",
                "revenue_variance_pct": "",
                "reported_revenue_inr": "65000000",
                "verified_revenue_inr": "50000000",
            },
            {
                "application_id": "APP-ORPHAN",
                "applicant_id": "APL-9999",
                "sector": "Manufacturing",
                "risk_grade": "A",
                "requested_amount_inr": "5000000",
                "requested_tenor_months": "36",
                "indicative_rate_pct": "11.0",
                "ebitda_inr": "4000000",
                "existing_annual_debt_service_inr": "500000",
                "proposed_annual_debt_service_inr": "1500000",
                "collateral_value_inr": "6000000",
                "revenue_variance_pct": "2.0",
                "reported_revenue_inr": "50000000",
                "verified_revenue_inr": "49000000",
            },
        ],
    )

    write_csv(
        api_dir,
        "sector_policies.csv",
        [
            "sector",
            "dscr_floor",
            "max_tenor_months",
            "min_collateral_coverage_ratio",
            "sector_cap_inr",
            "current_exposure_inr",
        ],
        [
            {
                "sector": "Manufacturing",
                "dscr_floor": "1.25",
                "max_tenor_months": "60",
                "min_collateral_coverage_ratio": "1.20",
                "sector_cap_inr": "500000000",
                "current_exposure_inr": "200000000",
            },
            {
                "sector": "Logistics",
                "dscr_floor": "1.20",
                "max_tenor_months": "60",
                "min_collateral_coverage_ratio": "1.10",
                "sector_cap_inr": "300000000",
                "current_exposure_inr": "150000000",
            },
            {
                "sector": "Textiles",
                "dscr_floor": "1.30",
                "max_tenor_months": "48",
                "min_collateral_coverage_ratio": "1.25",
                "sector_cap_inr": "100000000",
                "current_exposure_inr": "60000000",
            },
        ],
    )

    write_csv(
        api_dir,
        "policy_limits.csv",
        ["policy_name", "value"],
        [
            {"policy_name": "doc_mismatch_threshold_pct", "value": "10.0"},
            {"policy_name": "fraud_hold_threshold_pct", "value": "25.0"},
            {"policy_name": "max_tenor_months", "value": "60"},
            {"policy_name": "min_collateral_coverage_ratio", "value": "1.0"},
        ],
    )

    write_csv(
        api_dir,
        "risk_grades.csv",
        ["risk_grade", "indicative_rate_pct", "max_tenor_months", "min_collateral_coverage_ratio"],
        [
            {
                "risk_grade": "A",
                "indicative_rate_pct": "10.5",
                "max_tenor_months": "60",
                "min_collateral_coverage_ratio": "1.10",
            },
            {
                "risk_grade": "B",
                "indicative_rate_pct": "12.0",
                "max_tenor_months": "48",
                "min_collateral_coverage_ratio": "1.15",
            },
            {
                "risk_grade": "C",
                "indicative_rate_pct": "14.0",
                "max_tenor_months": "36",
                "min_collateral_coverage_ratio": "1.25",
            },
        ],
    )

    write_csv(
        api_dir,
        "underwriting_reference.csv",
        ["parameter", "value"],
        [
            {"parameter": "dscr_formula", "value": "ebitda_inr / (existing_annual_debt_service_inr + proposed_annual_debt_service_inr)"},
            {"parameter": "default_indicative_rate_pct", "value": "12.0"},
        ],
    )

    write_csv(
        api_dir,
        "reference_rates.csv",
        ["currency_pair", "rate", "as_of_date"],
        [
            {"currency_pair": "USD/INR", "rate": "84.25", "as_of_date": "2026-11-01"},
            {"currency_pair": "EUR/INR", "rate": "91.10", "as_of_date": "2026-11-01"},
        ],
    )

    write_csv(
        api_dir,
        "bureau_reports.csv",
        [
            "applicant_id",
            "bureau_score",
            "enquiries_last_6m",
            "dpd_30_last_12m",
            "dpd_90_ever",
            "written_off_amount_inr",
        ],
        [
            {
                "applicant_id": "APL-101",
                "bureau_score": "785",
                "enquiries_last_6m": "2",
                "dpd_30_last_12m": "0",
                "dpd_90_ever": "0",
                "written_off_amount_inr": "0",
            },
            {
                "applicant_id": "APL-102",
                "bureau_score": "690",
                "enquiries_last_6m": "8",
                "dpd_30_last_12m": "2",
                "dpd_90_ever": "0",
                "written_off_amount_inr": "0",
            },
            {
                "applicant_id": "APL-103",
                "bureau_score": "610",
                "enquiries_last_6m": "9",
                "dpd_30_last_12m": "4",
                "dpd_90_ever": "1",
                "written_off_amount_inr": "₹2,50,000",
            },
        ],
    )

    write_csv(
        api_dir,
        "bank_statements.csv",
        [
            "applicant_id",
            "month",
            "monthly_inflow_inr",
            "monthly_outflow_inr",
            "closing_balance_inr",
            "bounced_instruments",
        ],
        [
            {
                "applicant_id": "APL-101",
                "month": "2026-08",
                "monthly_inflow_inr": "10000000",
                "monthly_outflow_inr": "8000000",
                "closing_balance_inr": "3500000",
                "bounced_instruments": "0",
            },
            {
                "applicant_id": "APL-101",
                "month": "2026-09",
                "monthly_inflow_inr": "12000000",
                "monthly_outflow_inr": "9000000",
                "closing_balance_inr": "4500000",
                "bounced_instruments": "1",
            },
            {
                "applicant_id": "APL-101",
                "month": "2026-10",
                "monthly_inflow_inr": "11000000",
                "monthly_outflow_inr": "8500000",
                "closing_balance_inr": "4000000",
                "bounced_instruments": "0",
            },
            {
                "applicant_id": "APL-102",
                "month": "2026-10",
                "monthly_inflow_inr": "0",
                "monthly_outflow_inr": "500000",
                "closing_balance_inr": "100000",
                "bounced_instruments": "3",
            },
        ],
    )

    write_csv(
        api_dir,
        "approval_authorities.csv",
        ["min_amount_inr", "max_amount_inr", "risk_grade", "signatory_level", "sla_hours"],
        [
            {
                "min_amount_inr": "0",
                "max_amount_inr": "10000000",
                "risk_grade": "A",
                "signatory_level": "Regional Credit Manager",
                "sla_hours": "12",
            },
            {
                "min_amount_inr": "0",
                "max_amount_inr": "25000000",
                "risk_grade": "",
                "signatory_level": "Senior Credit Officer",
                "sla_hours": "24",
            },
            {
                "min_amount_inr": "25000001",
                "max_amount_inr": "",
                "risk_grade": "",
                "signatory_level": "Central Credit Committee",
                "sla_hours": "48",
            },
        ],
    )

    return api_dir


@pytest.fixture
def csv_writer(mock_api_dir: Path) -> Callable[[str, list[str], list[dict[str, Any]]], Path]:
    """Return a convenience helper to overwrite any CSV inside ``mock_api_dir``."""

    def _writer(filename: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
        return write_csv(mock_api_dir, filename, fieldnames, rows)

    return _writer
