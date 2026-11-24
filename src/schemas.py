"""Pydantic data contracts and customer hydration for WealthPilot SME underwriting."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

try:
    from src.config import settings
except ImportError:  # pragma: no cover - supports running with src/ on sys.path
    try:
        from config import settings  # type: ignore[no-redef]
    except ImportError:
        settings = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Cache customer records once per resolved CSV path for O(1) SME lookups
_CUSTOMER_CACHE: dict[str, dict[str, Any]] | None = None
_CUSTOMER_CACHE_SOURCE: str | None = None

_MISSING_STRING_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "null", "none", "nan", "-"}
)

_HYDRATION_FALLBACKS: dict[str, Any] = {
    "declared_annual_revenue_inr": None,
    "existing_annual_debt_service_inr": 0.0,
    "current_assets_inr": 0.0,
    "current_liabilities_inr": 0.0,
    "collateral_type": "Unsecured",
    "collateral_value_inr": 0.0,
    "export_receivables_usd": 0.0,
    "proprietor_name": "Not Provided",
    "city": "Mumbai",
    "application_language": "en",
    "raw_narrative": "",
}


def _is_missing_value(value: Any) -> bool:
    """Return True if a value represents null, NaN, or an empty placeholder."""
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    if isinstance(value, str):
        return value.strip().lower() in _MISSING_STRING_TOKENS
    return False


def _ensure_finite(value: float, field_name: str, errors: list[str]) -> None:
    if not math.isfinite(value):
        errors.append(f"{field_name} must be a finite number, got {value!r}")


def _resolve_customer_csv_paths(explicit_path: Path | str | None = None) -> list[Path]:
    """Return candidate CSV paths for SME profile hydration in priority order."""
    if explicit_path is not None:
        return [Path(explicit_path)]

    candidates: list[Path] = []
    data_dir_val = getattr(settings, "DATA_DIR", None) if settings is not None else None
    if isinstance(data_dir_val, (str, Path)) and str(data_dir_val).strip():
        base_dir = Path(data_dir_val)
        candidates.append(base_dir / "mock_api" / "loan_applications.csv")
        candidates.append(base_dir / "loan_applications.csv")

    candidates.extend(
        [
            Path("data/wealthpilot/mock_api/loan_applications.csv"),
            Path("data/loan_applications.csv"),
        ]
    )
    return candidates


def _clean_csv_row(row: dict[str | None, str | None]) -> dict[str, Any]:
    """Strip CSV headers/cells and discard missing placeholder entries."""
    cleaned: dict[str, Any] = {}
    for raw_key, raw_val in row.items():
        if raw_key is None:
            continue
        key = raw_key.strip()
        if not key:
            continue
        if raw_val is None:
            continue
        val_str = raw_val.strip()
        if val_str.lower() in _MISSING_STRING_TOKENS:
            continue
        cleaned[key] = val_str
    return cleaned


def clear_customer_cache() -> None:
    """Clear the cached SME lookup table (intended for tests and reloads)."""
    global _CUSTOMER_CACHE, _CUSTOMER_CACHE_SOURCE
    _CUSTOMER_CACHE = None
    _CUSTOMER_CACHE_SOURCE = None


def _get_customer_records(
    csv_path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and cache SME records indexed by sme_id, applicant_id, and application_id."""
    global _CUSTOMER_CACHE, _CUSTOMER_CACHE_SOURCE

    requested_source = str(Path(csv_path).resolve()) if csv_path is not None else "default"
    if _CUSTOMER_CACHE is not None and _CUSTOMER_CACHE_SOURCE == requested_source:
        return _CUSTOMER_CACHE

    records: dict[str, dict[str, Any]] = {}
    selected_path: Path | None = None
    for candidate in _resolve_customer_csv_paths(csv_path):
        try:
            if candidate.is_file():
                selected_path = candidate
                break
        except OSError as exc:
            logger.warning("Unable to stat candidate CSV %s: %s", candidate, exc)

    if selected_path is None:
        _CUSTOMER_CACHE = records
        _CUSTOMER_CACHE_SOURCE = requested_source
        return records

    try:
        with selected_path.open(mode="r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                logger.warning("Hydration CSV has no header row: %s", selected_path)
            else:
                for raw_row in reader:
                    cleaned = _clean_csv_row(raw_row)
                    if not cleaned:
                        continue
                    for key_col in ("sme_id", "applicant_id", "application_id"):
                        lookup_val = cleaned.get(key_col)
                        if isinstance(lookup_val, str) and lookup_val.strip():
                            records[lookup_val.strip()] = cleaned
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning(
            "Failed reading customer hydration CSV at %s (%s); continuing without cache.",
            selected_path,
            exc,
        )
        records = {}

    _CUSTOMER_CACHE = records
    _CUSTOMER_CACHE_SOURCE = requested_source
    return records


def _coerce_numeric_string(value: Any) -> Any:
    """Normalize currency-formatted numeric strings when safely parseable."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if text.lower() in _MISSING_STRING_TOKENS:
        return value
    cleaned = (
        text.replace(",", "")
        .replace("₹", "")
        .replace("$", "")
        .replace("INR", "")
        .replace("USD", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return value


class LoanApplication(BaseModel):
    """Typed SME loan intake record with graceful CSV hydration and validation."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    application_id: str
    applicant_id: str
    business_name: str
    sector: str
    requested_amount_inr: float
    requested_tenor_months: int
    declared_annual_revenue_inr: float
    dscr: float | None = None
    existing_annual_debt_service_inr: float = 0.0
    current_assets_inr: float = 0.0
    current_liabilities_inr: float = 0.0
    collateral_type: str = "Unsecured"
    collateral_value_inr: float = 0.0
    export_receivables_usd: float = 0.0
    proprietor_name: str = "Not Provided"
    city: str = "Mumbai"
    application_language: str = "en"
    raw_narrative: str = ""

    @model_validator(mode="before")
    @classmethod
    def hydrate_missing_customer_fields(cls, data: Any) -> Any:
        """Hydrate omitted/null fields from cached SME profiles and safe fallbacks."""
        if not isinstance(data, dict):
            return data

        incoming = dict(data)
        if not incoming.get("applicant_id") and isinstance(incoming.get("sme_id"), str):
            sme_val = incoming["sme_id"].strip()
            if sme_val:
                incoming["applicant_id"] = sme_val

        stored_profile: dict[str, Any] = {}
        try:
            records = _get_customer_records()
            for candidate_key in (
                incoming.get("sme_id"),
                incoming.get("applicant_id"),
                incoming.get("application_id"),
            ):
                if isinstance(candidate_key, str) and candidate_key.strip():
                    match = records.get(candidate_key.strip())
                    if match:
                        stored_profile = dict(match)
                        break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Customer profile hydration skipped due to error: %s", exc)

        # Normalize sme_id -> applicant_id from stored profile if needed
        if (
            stored_profile
            and not stored_profile.get("applicant_id")
            and isinstance(stored_profile.get("sme_id"), str)
        ):
            stored_profile["applicant_id"] = stored_profile["sme_id"]

        # Merge stored profile with incoming payload; incoming explicit values take priority,
        # unless incoming value is None/NaN while stored profile has a valid value.
        hydrated: dict[str, Any] = dict(stored_profile)
        for key, value in incoming.items():
            if value is None and key in stored_profile and not _is_missing_value(stored_profile[key]):
                continue
            hydrated[key] = value

        # Apply safe defaults only when missing or null/NaN from CSV/omission;
        # preserve explicit caller invalid values (e.g., negative, 0, NaN, inf) for validation.
        for field_name, default_val in _HYDRATION_FALLBACKS.items():
            if default_val is None:
                continue
            if field_name not in hydrated:
                hydrated[field_name] = default_val
            elif hydrated[field_name] is None:
                hydrated[field_name] = default_val
            elif field_name not in incoming and _is_missing_value(hydrated[field_name]):
                hydrated[field_name] = default_val

        # Coerce currency-formatted strings from CSVs without masking explicit NaN/inf
        for numeric_field in (
            "requested_amount_inr",
            "declared_annual_revenue_inr",
            "existing_annual_debt_service_inr",
            "current_assets_inr",
            "current_liabilities_inr",
            "collateral_value_inr",
            "export_receivables_usd",
        ):
            if numeric_field in hydrated and isinstance(hydrated[numeric_field], str):
                hydrated[numeric_field] = _coerce_numeric_string(hydrated[numeric_field])

        return hydrated

    @model_validator(mode="after")
    def validate_application_constraints(self) -> LoanApplication:
        """Enforce positive loan amounts/tenors and non-negative financial balances."""
        errors: list[str] = []

        for required_str in (
            "application_id",
            "applicant_id",
            "business_name",
            "sector",
        ):
            if not getattr(self, required_str):
                errors.append(f"{required_str} must be a non-empty string")

        numeric_fields = (
            "requested_amount_inr",
            "declared_annual_revenue_inr",
            "existing_annual_debt_service_inr",
            "current_assets_inr",
            "current_liabilities_inr",
            "collateral_value_inr",
            "export_receivables_usd",
        )
        for field_name in numeric_fields:
            _ensure_finite(getattr(self, field_name), field_name, errors)

        if self.dscr is not None:
            _ensure_finite(self.dscr, "dscr", errors)
            if self.dscr < 0:
                errors.append(f"dscr must be non-negative (>= 0), got {self.dscr}")

        if self.requested_amount_inr <= 0:
            errors.append(
                "requested_amount_inr must be strictly positive (> 0), "
                f"got {self.requested_amount_inr}"
            )
        if self.requested_tenor_months <= 0:
            errors.append(
                "requested_tenor_months must be strictly positive (> 0), "
                f"got {self.requested_tenor_months}"
            )
        if self.declared_annual_revenue_inr <= 0:
            errors.append(
                "declared_annual_revenue_inr must be strictly positive (> 0), "
                f"got {self.declared_annual_revenue_inr}"
            )

        non_negative_fields = (
            "existing_annual_debt_service_inr",
            "current_assets_inr",
            "current_liabilities_inr",
            "collateral_value_inr",
            "export_receivables_usd",
        )
        for field_name in non_negative_fields:
            value = getattr(self, field_name)
            if value < 0:
                errors.append(f"{field_name} must be non-negative (>= 0), got {value}")

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @classmethod
    def safe_validate(cls, payload: Any) -> tuple[LoanApplication | None, list[str]]:
        """Validate payload gracefully without raising exceptions to callers."""
        try:
            decoded: Any = payload
            if isinstance(payload, str):
                decoded = json.loads(payload)
            if not isinstance(decoded, dict):
                return None, ["Application payload must be a JSON object."]
            return cls.model_validate(decoded), []
        except json.JSONDecodeError as exc:
            return None, [f"Invalid JSON payload: {exc.msg}"]
        except ValidationError as exc:
            formatted = [
                f"{'.'.join(str(part) for part in err.get('loc', ())) or 'body'}: {err.get('msg', 'invalid')}"
                for err in exc.errors()
            ]
            return None, formatted
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected schema validation failure: %s", exc)
            return None, [str(exc)]


class FinancialSnapshot(BaseModel):
    """Normalized underwriting financial metrics and reconciliation flags for an SME."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    application_id: str
    declared_annual_revenue_inr: float
    verified_annual_revenue_inr: float
    revenue_variance_pct: float = 0.0
    ebitda_inr: float
    annual_debt_service_inr: float
    dscr: float
    current_ratio: float
    collateral_coverage_ratio: float = 0.0
    ltv_ratio: Optional[float] = None
    export_exposure_inr: float = 0.0
    revenue_mismatch_flag: bool = False
    fraud_hold_flag: bool = False
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_snapshot_lists(cls, data: Any) -> Any:
        """Normalize null list fields gracefully before validation."""
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        if cleaned.get("notes") is None:
            cleaned["notes"] = []
        return cleaned

    @model_validator(mode="after")
    def validate_financial_metrics(self) -> FinancialSnapshot:
        """Ensure computed ratios and monetary metrics are finite and valid."""
        errors: list[str] = []

        if not self.application_id:
            errors.append("application_id must be a non-empty string")

        for field_name in (
            "declared_annual_revenue_inr",
            "verified_annual_revenue_inr",
            "revenue_variance_pct",
            "ebitda_inr",
            "annual_debt_service_inr",
            "dscr",
            "current_ratio",
            "collateral_coverage_ratio",
            "export_exposure_inr",
        ):
            _ensure_finite(getattr(self, field_name), field_name, errors)

        if self.ltv_ratio is not None:
            _ensure_finite(self.ltv_ratio, "ltv_ratio", errors)
            if self.ltv_ratio < 0:
                errors.append(f"ltv_ratio must be >= 0, got {self.ltv_ratio}")

        for non_neg in (
            "declared_annual_revenue_inr",
            "verified_annual_revenue_inr",
            "revenue_variance_pct",
            "annual_debt_service_inr",
            "dscr",
            "current_ratio",
            "collateral_coverage_ratio",
            "export_exposure_inr",
        ):
            val = getattr(self, non_neg)
            if val < 0:
                errors.append(f"{non_neg} must be >= 0, got {val}")

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @classmethod
    def safe_validate(cls, payload: Any) -> tuple[FinancialSnapshot | None, list[str]]:
        """Validate FinancialSnapshot gracefully without raising exceptions."""
        try:
            if not isinstance(payload, dict):
                return None, ["FinancialSnapshot payload must be a mapping."]
            return cls.model_validate(payload), []
        except ValidationError as exc:
            formatted = [
                f"{'.'.join(str(part) for part in err.get('loc', ())) or 'body'}: {err.get('msg', 'invalid')}"
                for err in exc.errors()
            ]
            return None, formatted
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected FinancialSnapshot validation failure: %s", exc)
            return None, [str(exc)]


class CommitteeVerdict(BaseModel):
    """Final credit committee underwriting decision with citations and rationale."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    decision: Literal["approve", "borderline", "decline", "human_review"]
    dscr: float
    risk_grade: str
    reason_codes: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="before")
    @classmethod
    def normalize_verdict_lists(cls, data: Any) -> Any:
        """Normalize null list fields gracefully before validation."""
        if not isinstance(data, dict):
            return data
        cleaned = dict(data)
        if cleaned.get("reason_codes") is None:
            cleaned["reason_codes"] = []
        if cleaned.get("citations") is None:
            cleaned["citations"] = []
        return cleaned

    @model_validator(mode="after")
    def validate_verdict(self) -> CommitteeVerdict:
        """Validate verdict completeness and clean list entries."""
        errors: list[str] = []
        _ensure_finite(self.dscr, "dscr", errors)
        if self.dscr < 0:
            errors.append(f"dscr must be >= 0, got {self.dscr}")
        if not self.risk_grade:
            errors.append("risk_grade must be a non-empty string")
        if not self.rationale:
            errors.append("rationale must be a non-empty string")

        self.reason_codes = [code.strip() for code in self.reason_codes if code.strip()]
        self.citations = [cite.strip() for cite in self.citations if cite.strip()]

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @classmethod
    def safe_validate(cls, payload: Any) -> tuple[CommitteeVerdict | None, list[str]]:
        """Validate CommitteeVerdict gracefully without raising exceptions."""
        try:
            if not isinstance(payload, dict):
                return None, ["CommitteeVerdict payload must be a mapping."]
            return cls.model_validate(payload), []
        except ValidationError as exc:
            formatted = [
                f"{'.'.join(str(part) for part in err.get('loc', ())) or 'body'}: {err.get('msg', 'invalid')}"
                for err in exc.errors()
            ]
            return None, formatted
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected CommitteeVerdict validation failure: %s", exc)
            return None, [str(exc)]
