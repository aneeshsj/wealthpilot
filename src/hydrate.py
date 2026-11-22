import pandas as pd
from pathlib import Path
from typing import Any
from pydantic import BaseModel, model_validator

# Cache your customer dataset once so lookups for any SME are instant
_CUSTOMER_CACHE: dict[str, dict[str, Any]] | None = None

def _get_customer_records() -> dict[str, dict[str, Any]]:
    global _CUSTOMER_CACHE
    if _CUSTOMER_CACHE is None:
        # Point to your actual dataset file in data/ (e.g., loan_applications.csv or sme_profiles.csv)
        csv_path = Path("data/loan_applications.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Index by sme_id (or application_id) for O(1) lookup for any customer
            key_col = "sme_id" if "sme_id" in df.columns else "application_id"
            _CUSTOMER_CACHE = df.set_index(key_col, drop=False).to_dict(orient="index")
        else:
            _CUSTOMER_CACHE = {}
    return _CUSTOMER_CACHE


class LoanApplication(BaseModel):
    application_id: str
    sme_id: str
    declared_annual_revenue_inr: float
    existing_annual_debt_service_inr: float
    current_assets_inr: float
    current_liabilities_inr: float
    collateral_type: str
    collateral_value_inr: float
    export_receivables_usd: float
    proprietor_name: str
    city: str
    application_language: str

    @model_validator(mode="before")
    @classmethod
    def hydrate_missing_customer_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        records = _get_customer_records()
        lookup_key = data.get("sme_id") or data.get("application_id")
        stored_profile = records.get(lookup_key, {})

        # Merge stored profile with incoming payload (incoming UI edits take priority)
        hydrated = {**stored_profile, **data}

        # Fallback safe defaults only if a brand-new customer isn't in the dataset yet
        fallbacks = {
            "declared_annual_revenue_inr": 0.0,
            "existing_annual_debt_service_inr": 0.0,
            "current_assets_inr": 0.0,
            "current_liabilities_inr": 0.0,
            "collateral_type": "Unsecured",
            "collateral_value_inr": 0.0,
            "export_receivables_usd": 0.0,
            "proprietor_name": "Not Provided",
            "city": "Mumbai",
            "application_language": "en",
        }
        for field, default_val in fallbacks.items():
            if hydrated.get(field) is None or pd.isna(hydrated.get(field)):
                hydrated[field] = default_val

        return hydrated

