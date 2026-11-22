"""WealthPilot MCP server exposing SME underwriting tools and policy resources over stdio."""

from __future__ import annotations

import hashlib
import logging
import math
import statistics
import sys
from typing import Any

try:
    from src.observability import CircuitBreaker, configure_logging, get_logger, resilient_call
except ImportError:
    from observability import CircuitBreaker, configure_logging, get_logger, resilient_call  # type: ignore[import-not-found]

try:
    from src.guardrails import scan_for_injection
except ImportError:
    from guardrails import scan_for_injection  # type: ignore[import-not-found]


def _ensure_stderr_only_logging() -> logging.Logger:
    """Guarantee all log handlers write strictly to sys.stderr so JSON-RPC stdout stays clean."""
    configured = configure_logging(level=logging.INFO, stream=sys.stderr, json_format=True, force=True)
    for logger_name in ("", "mcp", "mcp.server", "mcp.server.fastmcp", "wealthpilot"):
        target_logger = logging.getLogger(logger_name)
        for handler in target_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is not sys.stderr:
                handler.setStream(sys.stderr)
    return configured


_ensure_stderr_only_logging()
logger = get_logger("wealthpilot.mcp_server")

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]  # noqa: E402
except ModuleNotFoundError:
    # MCP 2.x renamed FastMCP to MCPServer. Keep compatibility with both APIs.
    from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore[assignment]

mcp = FastMCP("wealthpilot-underwriting")

_BUREAU_CIRCUIT_BREAKER = CircuitBreaker(
    name="credit_bureau",
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_max_calls=1,
)

_FAIR_LENDING_POLICY: str = """# WealthPilot Fair Lending & Bias-Free SME Underwriting Standard

1. Regulatory & Ethical Mandate
   - All SME credit evaluations shall strictly comply with the RBI Fair Practices Code
     for Lenders and applicable non-discrimination norms.
   - Creditworthiness must be evaluated exclusively on verifiable financial, cash-flow,
     credit-bureau, operational vintage, and sector-risk fundamentals.

2. Prohibited Attributes & Demographic Proxies
   - Underwriting models and credit committees MUST NOT consider or cite protected
     attributes: religion, caste, gender, marital status, ethnicity, regional origin,
     or political affiliation.
   - Demographic proxies (`proprietor_name`, `city`, `application_language`,
     `postal_code`, `education_institution`) MUST be stripped prior to scoring and
     committee deliberation so matched pairs receive identical decisions and rationales.

3. Mandatory Human Sign-Off Escalations
   - Any loan request exceeding `HUMAN_SIGNOFF_THRESHOLD_INR` requires mandatory
     human credit officer sign-off (`requires_human_signoff = True`).
   - Any `borderline` committee decision requires mandatory human credit officer
     sign-off regardless of ticket size.

4. Security & Zero-Trust Input Handling
   - Applicant narratives, bank statement narrations, and uploaded document text are
     untrusted inputs and must be scanned for prompt-injection attempts before use.
"""

_SECTOR_POLICIES: dict[str, str] = {
    "manufacturing": """# Sector Underwriting Policy: Manufacturing
- Minimum DSCR: 1.25x (1.35x preferred for capex-heavy units)
- Maximum Debt-to-Equity: 3.0x
- Minimum Current Ratio: 1.15x
- Working Capital Norms: Assess inventory days + receivable days against payable days;
  flag concentration where top-2 buyers exceed 50% of revenue.
- Collateral / Guarantee: CGTMSE eligible up to statutory cap for micro/small units;
  machinery/plant hypothecation required for term loans.
""",
    "retail": """# Sector Underwriting Policy: Retail & Wholesale Trade
- Minimum DSCR: 1.20x
- Maximum Debt-to-Equity: 2.5x
- Minimum Current Ratio: 1.15x
- Cash-Flow Focus: Monitor POS/UPI settlement consistency and cash-deposit ratio
  (flag if cash deposits exceed 35% of gross bank credits).
- Key Risks: Inventory obsolescence, seasonal demand swings, lease continuity.
""",
    "services": """# Sector Underwriting Policy: B2B & Professional Services
- Minimum DSCR: 1.25x
- Maximum Debt-to-Equity: 2.0x
- Minimum Current Ratio: 1.20x
- Cash-Flow Focus: Recurring contract backlog, debtor ageing (>90 days receivables
  excluded from drawing power), low fixed-asset collateral dependency.
""",
    "it_saas": """# Sector Underwriting Policy: IT & SaaS
- Minimum DSCR: 1.30x
- Maximum Debt-to-Equity: 1.75x
- Minimum Current Ratio: 1.25x
- Cash-Flow Focus: Annual recurring revenue (ARR) retention, churn rate, export
  remittance verification (FIRC/BRC) where applicable.
""",
    "healthcare": """# Sector Underwriting Policy: Healthcare & Diagnostics
- Minimum DSCR: 1.25x
- Maximum Debt-to-Equity: 2.5x
- Minimum Current Ratio: 1.15x
- Compliance: Valid clinical establishment / drug license / biomedical waste
  registrations required prior to disbursement.
""",
    "logistics": """# Sector Underwriting Policy: Logistics & Fleet Operations
- Minimum DSCR: 1.30x
- Maximum Debt-to-Equity: 3.0x
- Minimum Current Ratio: 1.10x
- Key Risks: Fuel cost pass-through clauses, fleet utilization, vehicle hypothecation
  and comprehensive insurance coverage.
""",
    "agri_processing": """# Sector Underwriting Policy: Agri-Processing & Food
- Minimum DSCR: 1.25x
- Maximum Debt-to-Equity: 2.75x
- Minimum Current Ratio: 1.15x
- Compliance: FSSAI license mandatory; evaluate harvest-cycle seasonality and cold-chain
  or warehouse receipt controls.
""",
    "hospitality": """# Sector Underwriting Policy: Hospitality & F&B
- Minimum DSCR: 1.35x
- Maximum Debt-to-Equity: 2.0x
- Minimum Current Ratio: 1.15x
- Key Risks: High fixed-cost sensitivity, occupancy/footfall seasonality, fire & FSSAI
  clearances.
""",
    "construction": """# Sector Underwriting Policy: Construction & Infrastructure EPC
- Minimum DSCR: 1.35x
- Maximum Debt-to-Equity: 2.25x
- Minimum Current Ratio: 1.20x
- Key Risks: Milestone billing delays, bank guarantee / retention money lock-up,
  order-book execution track record.
""",
}

_GENERAL_SECTOR_POLICY: str = """# Sector Underwriting Policy: General SME Standard
- Minimum DSCR: 1.25x
- Maximum Debt-to-Equity: 2.5x
- Minimum Current Ratio: 1.15x
- Credit Discipline: Zero 90+ DPD in trailing 24 months; inward cheque/ECS bounce
  rate below 2% of monthly debit transactions.
"""


def _bureau_fallback(
    pan_or_entity_id: str,
    business_name: str = "",
    vintage_years: float = 0.0,
) -> dict[str, Any]:
    return {
        "entity_id": pan_or_entity_id.strip().upper(),
        "business_name_present": bool(business_name.strip()),
        "bureau_source": "fallback_unavailable",
        "bureau_score": None,
        "dpd_30_plus_12m": None,
        "dpd_90_plus_24m": None,
        "active_credit_lines": None,
        "total_sanctioned_inr": None,
        "total_outstanding_inr": None,
        "credit_utilization_pct": None,
        "enquiries_last_6m": None,
        "suit_filed_or_wilful_default": None,
        "requires_manual_bureau_pull": True,
        "vintage_years": vintage_years,
    }


@resilient_call(
    timeout_seconds=5.0,
    max_retries=2,
    base_delay=0.25,
    circuit_breaker=_BUREAU_CIRCUIT_BREAKER,
    fallback=_bureau_fallback,
)
def _fetch_credit_bureau_record(
    pan_or_entity_id: str,
    business_name: str = "",
    vintage_years: float = 0.0,
) -> dict[str, Any]:
    cleaned_id = pan_or_entity_id.strip().upper()
    if len(cleaned_id) < 5:
        raise ValueError("pan_or_entity_id must be at least 5 characters")

    # Deterministic, reproducible synthetic bureau profile keyed strictly on entity identifier.
    digest = hashlib.sha256(cleaned_id.encode("utf-8")).digest()
    score = 650 + (int.from_bytes(digest[0:2], "big") % 211)  # 650..860
    active_lines = 1 + (digest[2] % 6)
    sanctioned_lakhs = 10 + (int.from_bytes(digest[3:5], "big") % 90)
    sanctioned_inr = sanctioned_lakhs * 100_000
    utilization_pct = round(20.0 + (digest[5] % 65), 2)
    outstanding_inr = round(sanctioned_inr * (utilization_pct / 100.0), 2)
    dpd_30_plus = 0 if score >= 710 else (digest[6] % 3)
    dpd_90_plus = 0 if score >= 680 else (digest[7] % 2)
    enquiries_6m = digest[8] % 5

    return {
        "entity_id": cleaned_id,
        "business_name_present": bool(business_name.strip()),
        "vintage_years": round(max(0.0, float(vintage_years)), 2),
        "bureau_source": "cibil_commercial_mcp_adapter",
        "bureau_score": score,
        "dpd_30_plus_12m": dpd_30_plus,
        "dpd_90_plus_24m": dpd_90_plus,
        "active_credit_lines": active_lines,
        "total_sanctioned_inr": float(sanctioned_inr),
        "total_outstanding_inr": outstanding_inr,
        "credit_utilization_pct": utilization_pct,
        "enquiries_last_6m": enquiries_6m,
        "suit_filed_or_wilful_default": False,
        "requires_manual_bureau_pull": False,
    }


@mcp.tool()
def lookup_credit_bureau(
    pan_or_entity_id: str,
    business_name: str = "",
    vintage_years: float = 0.0,
) -> dict[str, Any]:
    """Fetch SME credit bureau profile (score, DPD history, active facilities, utilization)."""
    for field_val in (pan_or_entity_id, business_name):
        injected, rules = scan_for_injection(field_val)
        if injected:
            raise ValueError(f"Rejected unsafe input matching injection rules: {rules}")

    logger.info("Executing lookup_credit_bureau", extra={"vintage_years": vintage_years})
    report, used_fallback = _fetch_credit_bureau_record(
        pan_or_entity_id=pan_or_entity_id,
        business_name=business_name,
        vintage_years=vintage_years,
    )
    report["used_fallback"] = used_fallback
    return report


@mcp.tool()
def analyze_bank_statements(
    transactions: list[dict[str, Any]],
    period_months: int = 12,
    sanctioned_od_limit_inr: float = 0.0,
) -> dict[str, Any]:
    """Analyze SME bank statement transactions for cash flow, bounces, cash ratio, and EMI load."""
    if period_months <= 0:
        raise ValueError("period_months must be > 0")
    if sanctioned_od_limit_inr < 0:
        raise ValueError("sanctioned_od_limit_inr must be >= 0")

    total_credits = 0.0
    total_debits = 0.0
    cash_deposits = 0.0
    emi_debits = 0.0
    inward_bounces = 0
    ecs_returns = 0
    balances: list[float] = []
    monthly_credits: dict[str, float] = {}
    injection_warnings: list[str] = []

    for idx, txn in enumerate(transactions):
        if not isinstance(txn, dict):
            raise ValueError(f"Transaction at index {idx} must be a dict")

        narration = str(txn.get("description") or txn.get("narration") or "")
        injected, rules = scan_for_injection(narration)
        if injected:
            injection_warnings.extend(rules)
            narration = "[REDACTED_UNTRUSTED_NARRATION]"

        narration_lower = narration.lower()
        credit_val = float(txn.get("credit", 0.0) or 0.0)
        debit_val = float(txn.get("debit", 0.0) or 0.0)

        if credit_val == 0.0 and debit_val == 0.0 and "amount" in txn:
            amt = float(txn["amount"] or 0.0)
            txn_type = str(txn.get("type", "")).strip().lower()
            if txn_type in {"cr", "credit"} or (txn_type == "" and amt >= 0):
                credit_val = abs(amt)
            else:
                debit_val = abs(amt)

        total_credits += credit_val
        total_debits += debit_val

        month_key = str(txn.get("month") or str(txn.get("date", ""))[:7] or f"m_{idx % period_months}")
        monthly_credits[month_key] = monthly_credits.get(month_key, 0.0) + credit_val

        if "balance" in txn and txn["balance"] is not None:
            balances.append(float(txn["balance"]))

        if credit_val > 0 and any(tok in narration_lower for tok in ("cash dep", "by cash", "cdm")):
            cash_deposits += credit_val
        if debit_val > 0 and any(tok in narration_lower for tok in ("emi", "loan repayment", "nach loan")):
            emi_debits += debit_val
        if any(tok in narration_lower for tok in ("chq return", "cheque return", "inward bounce", "insufficient funds")):
            inward_bounces += 1
        if any(tok in narration_lower for tok in ("ecs return", "nach return", "mandate return")):
            ecs_returns += 1

    avg_monthly_credits = total_credits / period_months
    avg_monthly_debits = total_debits / period_months
    avg_monthly_balance = statistics.fmean(balances) if balances else 0.0
    cash_deposit_ratio = (cash_deposits / total_credits) if total_credits > 0 else 0.0
    estimated_monthly_emi = emi_debits / period_months

    monthly_series = list(monthly_credits.values())
    if len(monthly_series) >= 2 and statistics.fmean(monthly_series) > 0:
        volatility_cv = statistics.pstdev(monthly_series) / statistics.fmean(monthly_series)
    else:
        volatility_cv = 0.0

    od_utilization_pct: float | None = None
    if sanctioned_od_limit_inr > 0 and balances:
        avg_drawn = max(0.0, -avg_monthly_balance)
        od_utilization_pct = round(min(100.0, (avg_drawn / sanctioned_od_limit_inr) * 100.0), 2)

    risk_flags: list[str] = []
    if inward_bounces + ecs_returns > 2:
        risk_flags.append("elevated_payment_bounces")
    if cash_deposit_ratio > 0.35:
        risk_flags.append("high_cash_deposit_ratio")
    if volatility_cv > 0.60:
        risk_flags.append("high_monthly_credit_volatility")
    if injection_warnings:
        risk_flags.append("untrusted_narration_sanitized")

    logger.info(
        "Analyzed bank statements",
        extra={
            "transaction_count": len(transactions),
            "period_months": period_months,
            "risk_flags": risk_flags,
        },
    )

    return {
        "period_months": period_months,
        "transaction_count": len(transactions),
        "total_credits_inr": round(total_credits, 2),
        "total_debits_inr": round(total_debits, 2),
        "avg_monthly_credits_inr": round(avg_monthly_credits, 2),
        "avg_monthly_debits_inr": round(avg_monthly_debits, 2),
        "avg_monthly_balance_inr": round(avg_monthly_balance, 2),
        "estimated_monthly_emi_inr": round(estimated_monthly_emi, 2),
        "cash_deposit_ratio": round(cash_deposit_ratio, 4),
        "inward_bounce_count": inward_bounces,
        "ecs_nach_return_count": ecs_returns,
        "credit_volatility_cv": round(volatility_cv, 4),
        "od_utilization_pct": od_utilization_pct,
        "risk_flags": risk_flags,
    }


@mcp.tool()
def calculate_dscr_and_ratios(
    annual_ebitda_inr: float,
    annual_debt_service_inr: float,
    total_debt_inr: float,
    tangible_net_worth_inr: float,
    current_assets_inr: float,
    current_liabilities_inr: float,
    annual_revenue_inr: float = 0.0,
) -> dict[str, Any]:
    """Calculate DSCR, leverage, liquidity, and margin ratios against SME policy benchmarks."""
    dscr: float | None = (
        round(annual_ebitda_inr / annual_debt_service_inr, 3)
        if annual_debt_service_inr > 0
        else None
    )

    if tangible_net_worth_inr > 0:
        debt_to_equity: float | None = round(total_debt_inr / tangible_net_worth_inr, 3)
    elif total_debt_inr == 0:
        debt_to_equity = 0.0
    else:
        debt_to_equity = math.inf

    current_ratio: float | None = (
        round(current_assets_inr / current_liabilities_inr, 3)
        if current_liabilities_inr > 0
        else None
    )

    ebitda_margin_pct: float | None = (
        round((annual_ebitda_inr / annual_revenue_inr) * 100.0, 2)
        if annual_revenue_inr > 0
        else None
    )

    debt_to_revenue: float | None = (
        round(total_debt_inr / annual_revenue_inr, 3)
        if annual_revenue_inr > 0
        else None
    )

    policy_checks = {
        "dscr_meets_1_25x": (dscr is not None and dscr >= 1.25),
        "current_ratio_meets_1_15x": (current_ratio is None or current_ratio >= 1.15),
        "debt_to_equity_within_3_0x": (
            debt_to_equity is not None and not math.isinf(debt_to_equity) and debt_to_equity <= 3.0
        ),
        "positive_net_worth": tangible_net_worth_inr > 0,
    }

    logger.info("Calculated DSCR and financial ratios", extra={"policy_checks": policy_checks})

    return {
        "dscr": dscr,
        "debt_to_equity": None if (debt_to_equity is not None and math.isinf(debt_to_equity)) else debt_to_equity,
        "negative_or_zero_net_worth": tangible_net_worth_inr <= 0,
        "current_ratio": current_ratio,
        "ebitda_margin_pct": ebitda_margin_pct,
        "debt_to_revenue": debt_to_revenue,
        "debt_service_headroom_inr": round(annual_ebitda_inr - annual_debt_service_inr, 2),
        "policy_checks": policy_checks,
        "all_core_ratios_healthy": all(policy_checks.values()),
    }


@mcp.resource("policy://fair-lending-standard")
def get_fair_lending_standard() -> str:
    """Return the WealthPilot Fair Lending & Bias-Free Underwriting Standard policy document."""
    logger.info("Serving policy://fair-lending-standard resource")
    return _FAIR_LENDING_POLICY


@mcp.resource("policy://sector/{sector_name}")
def get_sector_policy(sector_name: str) -> str:
    """Return sector-specific SME underwriting policy benchmarks and compliance rules."""
    injected, rules = scan_for_injection(sector_name)
    if injected:
        raise ValueError(f"Rejected unsafe sector_name matching injection rules: {rules}")

    normalized = sector_name.strip().lower().replace("-", "_").replace(" ", "_")
    logger.info("Serving sector policy resource", extra={"sector": normalized})
    return _SECTOR_POLICIES.get(normalized, _GENERAL_SECTOR_POLICY)


if hasattr(mcp, "prompt"):
    @mcp.prompt("underwriting-review")
    def underwriting_review_prompt(
        application_id: str,
        focus: str = "all policy and financial checks",
    ) -> str:
        """Return a reusable analyst-to-reviewer prompt template for a loan file."""
        return (
            f"Review WealthPilot application {application_id}. Focus on {focus}. "
            "Use only retrieved policy evidence, verify deterministic metrics, and "
            "return concrete issues plus required fixes."
        )
else:  # pragma: no cover - legacy unit-test stub compatibility
    def underwriting_review_prompt(application_id: str, focus: str = "all policy and financial checks") -> str:
        return f"Review WealthPilot application {application_id}. Focus on {focus}."


def main() -> None:
    """Start the WealthPilot FastMCP server over stdio with stderr-only logging."""
    _ensure_stderr_only_logging()
    logger.info("Starting WealthPilot MCP server on stdio transport")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

