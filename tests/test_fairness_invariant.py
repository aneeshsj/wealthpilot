"""Product-level fairness checks for matched applications."""

from __future__ import annotations

from src.guardrails import sanitize_for_bias


def test_proxy_only_changes_are_removed_from_underwriting_input() -> None:
    common = {
        "sector": "Textiles",
        "requested_amount_inr": 3_000_000,
        "requested_tenor_months": 60,
        "declared_annual_revenue_inr": 24_000_000,
        "raw_narrative": "Working capital expansion. Proprietor: Rajesh Sharma, Bengaluru.",
    }
    first = sanitize_for_bias({**common, "proprietor_name": "Rajesh Sharma", "city": "Bengaluru"})
    second = sanitize_for_bias({**common, "proprietor_name": "Fatima Sheikh", "city": "Malegaon"})
    assert first == second
    assert "Rajesh" not in str(first)
    assert "Fatima" not in str(second)

