"""Deterministic input injection scanning, demographic bias sanitization, and output guardrails."""

from __future__ import annotations

import copy
import os
import re
from typing import Any, Protocol, runtime_checkable

try:
    from src.config import settings  # type: ignore[import-not-found]
except ImportError:
    try:
        from config import settings  # type: ignore[import-not-found]
    except ImportError:

        class _DefaultSettings:
            """Fallback settings when src.config is not present."""

            HUMAN_SIGNOFF_THRESHOLD_INR: int = int(
                os.environ.get("HUMAN_SIGNOFF_THRESHOLD_INR", "5000000")
            )

        settings = _DefaultSettings()  # type: ignore[assignment]

try:
    from src.models import CommitteeVerdict  # type: ignore[import-not-found]
except ImportError:
    try:
        from models import CommitteeVerdict  # type: ignore[import-not-found]
    except ImportError:

        @runtime_checkable
        class CommitteeVerdict(Protocol):  # type: ignore[no-redef]
            """Structural protocol matching WealthPilot's CommitteeVerdict model."""

            decision: str
            rationale: str
            requires_human_signoff: bool


try:
    from src.observability import get_logger
except ImportError:
    from observability import get_logger  # type: ignore[import-not-found]

logger = get_logger("wealthpilot.guardrails")

DEMOGRAPHIC_PROXY_FIELDS: frozenset[str] = frozenset(
    {
        "proprietor_name",
        "city",
        "application_language",
        "postal_code",
        "education_institution",
    }
)

_ZERO_WIDTH_RE: re.Pattern[str] = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")

_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(?:ignore|disregard|forget|bypass|override)\s+"
            r"(?:all\s+|any\s+|your\s+)?(?:previous|prior|above|earlier|system|safety|policy)\s+"
            r"(?:instructions?|prompts?|rules?|directives?|constraints?|guardrails?)\b"
        ),
    ),
    (
        "system_override_directive",
        re.compile(
            r"(?i)\b(?:system\s+override|admin\s+override|developer\s+override|root\s+override|"
            r"override\s+(?:underwriting|policy|guardrails?|decision)|"
            r"new\s+(?:system\s+)?(?:instructions?|directive|persona|role))\b"
        ),
    ),
    (
        "auto_approve_hijack",
        re.compile(
            r"(?i)\b(?:auto[-_\s]?approve(?:\s+immediately|\s+now|\s+this|\s+loan)?|"
            r"approve\s+(?:this\s+)?(?:loan|application|case)\s+(?:immediately|unconditionally|without\s+(?:review|checks?|verification))|"
            r"force\s+approv(?:e|al)|guaranteed\s+approval\s+directive)\b"
        ),
    ),
    (
        "role_impersonation_or_delimiter",
        re.compile(
            r"(?i)(?:^|\n|\s)(?:system|developer|assistant|admin)\s*:\s*(?:you\s+(?:must|will|are)|override|approve|ignore)|"
            r"<\s*/?\s*(?:system|developer|instruction|prompt|admin_override)\s*>|"
            r"\[\s*(?:SYSTEM|INST|ADMIN|OVERRIDE)\s*\]|"
            r"###\s*(?:SYSTEM|INSTRUCTION|OVERRIDE)\b|"
            r"\byou\s+are\s+now\s+(?:an?\s+)?(?:unrestricted|admin|root|auto[-_\s]?approver|system)\b"
        ),
    ),
    (
        "verification_bypass",
        re.compile(
            r"(?i)\b(?:skip|bypass|disable|omit)\s+(?:all\s+)?(?:credit|bureau|cibil|dscr|kyc|aml|compliance|human\s+sign[-_\s]?off)\s+"
            r"(?:checks?|verification|review|rules?|gates?)\b|"
            r"\bdo\s+not\s+(?:flag|reject|escalate|require\s+human)\b"
        ),
    ),
    (
        "verdict_field_injection",
        re.compile(
            r"(?i)\bset\s+(?:the\s+)?(?:decision|verdict|risk[_\s]?score|requires_human_signoff|approval_status)\s*(?:to|=)\s*"
            r"['\"]?(?:approve|approved|low|false|0)['\"]?"
        ),
    ),
    (
        "prompt_or_secret_exfiltration",
        re.compile(
            r"(?i)\b(?:reveal|print|output|leak|dump|repeat|expose)\s+(?:your\s+|the\s+)?"
            r"(?:system\s+prompt|hidden\s+instructions?|internal\s+rules?|api[_\s]?keys?|secrets?)\b"
        ),
    ),
)

_PROTECTED_OUTPUT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "protected_attribute:religion",
        re.compile(
            r"(?i)\b(?:hindu|muslim|islam(?:ic)?|christian|sikh|jain|buddhist|parsi|zoroastrian|"
            r"religion|religious\s+(?:affiliation|background|community|minority)|mosque|temple|church|gurudwara)\b"
        ),
    ),
    (
        "protected_attribute:caste_or_community",
        re.compile(
            r"(?i)\b(?:caste|sub[-_\s]?caste|brahmin|kshatriya|vaishya|shudra|dalit|harijan|"
            r"scheduled\s+caste|scheduled\s+tribe|backward\s+class|upper\s+caste|lower\s+caste|"
            r"obc\s+(?:category|status)|sc/st\s+(?:category|status))\b"
        ),
    ),
    (
        "protected_attribute:gender_or_marital",
        re.compile(
            r"(?i)\b(?:gender|female\s+(?:proprietor|applicant|owner|borrower)|"
            r"male\s+(?:proprietor|applicant|owner|borrower)|"
            r"woman[-_\s]led\s+risk|marital\s+status|unmarried|widow(?:ed)?|divorced|pregnan(?:t|cy)|maternity)\b"
        ),
    ),
    (
        "protected_attribute:ethnicity_or_region",
        re.compile(
            r"(?i)\b(?:ethnicity|ethnic\s+(?:origin|background|group)|race|racial|"
            r"native\s+(?:place|state|language)|mother\s+tongue|vernacular\s+(?:speaker|application)|"
            r"north\s+indian|south\s+indian|hindi[-_\s]speaking|non[-_\s]english\s+speaking)\b"
        ),
    ),
    (
        "demographic_proxy:explicit_proxy_reference",
        re.compile(
            r"(?i)\b(?:proprietor[_\s]name|application[_\s]language|education[_\s]institution|"
            r"postal[_\s]code|pin\s*code|zip\s*code|alma\s+mater|surname|family\s+name|"
            r"hometown|educational\s+pedigree|university\s+tier|elite\s+college)\b"
        ),
    ),
    (
        "demographic_proxy:city_tier_or_locality_bias",
        re.compile(
            r"(?i)\b(?:tier[-_\s]?[123]\s+city\s+(?:bias|risk|profile)|"
            r"located\s+in\s+(?:a\s+)?(?:rural|backward|minority|affluent)\s+(?:pin\s*code|postal\s*code|area|locality))\b"
        ),
    ),
)


def _normalize_for_scan(text: str) -> str:
    cleaned = _ZERO_WIDTH_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def scan_for_injection(text: str) -> tuple[bool, list[str]]:
    """Deterministically detect direct and indirect prompt injection attempts in text."""
    if not text:
        return False, []

    normalized = _normalize_for_scan(text)
    matched_rules: list[str] = []

    for rule_name, pattern in _INJECTION_RULES:
        if pattern.search(text) or pattern.search(normalized):
            if rule_name not in matched_rules:
                matched_rules.append(rule_name)

    detected = len(matched_rules) > 0
    if detected:
        logger.warning(
            "Prompt injection pattern detected",
            extra={"matched_rules": matched_rules, "text_length": len(text)},
        )
    return detected, matched_rules


def _strip_proxies_recursive(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_proxies_recursive(val)
            for key, val in node.items()
            if str(key).strip().lower() not in DEMOGRAPHIC_PROXY_FIELDS
        }
    if isinstance(node, list):
        return [_strip_proxies_recursive(item) for item in node]
    if isinstance(node, tuple):
        return tuple(_strip_proxies_recursive(item) for item in node)
    return node


def _redact_proxy_text(value: str) -> str:
    """Remove free-text demographic context before it reaches underwriting."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", value)
    proxy_terms = (
        "proprietor", "applicant name", "surname", "city", "postcode", "postal",
        "pin code", "language", "mother tongue", "college", "university", "education",
        "marital", "religion", "caste", "gender", "ethnicity",
    )
    kept = [sentence for sentence in sentences if not any(term in sentence.lower() for term in proxy_terms)]
    return " ".join(part.strip() for part in kept if part.strip())


def _redact_proxy_text_recursive(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _redact_proxy_text_recursive(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_redact_proxy_text_recursive(value) for value in node]
    if isinstance(node, str):
        return _redact_proxy_text(node)
    return node


def sanitize_for_bias(application_dict: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of application_dict with demographic proxy fields stripped."""
    if not isinstance(application_dict, dict):
        raise TypeError("application_dict must be a dict")

    cloned = copy.deepcopy(application_dict)
    sanitized = _redact_proxy_text_recursive(_strip_proxies_recursive(cloned))
    assert isinstance(sanitized, dict)
    return sanitized


def _extract_verdict_texts(verdict: Any) -> list[str]:
    texts: list[str] = []
    candidate_fields = (
        "rationale",
        "summary",
        "reasoning",
        "decision_rationale",
        "dissenting_opinion",
        "conditions",
        "key_risks",
        "mitigants",
    )
    for field_name in candidate_fields:
        val = verdict.get(field_name) if isinstance(verdict, dict) else getattr(verdict, field_name, None)
        if isinstance(val, str) and val.strip():
            texts.append(val)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str) and item.strip():
                    texts.append(item)
    return texts


def _get_verdict_attr(verdict: Any, attr: str, default: Any = None) -> Any:
    if isinstance(verdict, dict):
        return verdict.get(attr, default)
    return getattr(verdict, attr, default)


def _set_human_signoff(verdict: Any) -> bool:
    if isinstance(verdict, dict):
        verdict["requires_human_signoff"] = True
        return True
    try:
        setattr(verdict, "requires_human_signoff", True)
        return True
    except (AttributeError, TypeError, ValueError):
        try:
            object.__setattr__(verdict, "requires_human_signoff", True)
            return True
        except Exception:
            return bool(getattr(verdict, "requires_human_signoff", False))


def validate_output_guardrails(
    verdict: CommitteeVerdict,
    requested_amount_inr: int,
) -> tuple[bool, list[str]]:
    """Validate committee verdict rationale for bias/proxy leakage and enforce human sign-off.

    Returns ``(is_valid, violations)``. Blocks (``is_valid=False``) any verdict whose
    rationale references protected attributes or demographic proxies, and forces
    ``verdict.requires_human_signoff = True`` whenever ``requested_amount_inr`` exceeds
    ``settings.HUMAN_SIGNOFF_THRESHOLD_INR`` or ``decision == "borderline"``.
    """
    violations: list[str] = []

    texts_to_check = _extract_verdict_texts(verdict)
    combined_text = "\n".join(texts_to_check)
    normalized_text = _normalize_for_scan(combined_text)

    for rule_name, pattern in _PROTECTED_OUTPUT_RULES:
        if pattern.search(combined_text) or pattern.search(normalized_text):
            if rule_name not in violations:
                violations.append(rule_name)

    decision_raw = _get_verdict_attr(verdict, "decision", "")
    decision_normalized = str(decision_raw).strip().lower()
    threshold_inr = int(getattr(settings, "HUMAN_SIGNOFF_THRESHOLD_INR", 5000000))

    requires_escalation = (
        requested_amount_inr > threshold_inr or decision_normalized == "borderline"
    )
    if requires_escalation:
        previously_flagged = bool(_get_verdict_attr(verdict, "requires_human_signoff", False))
        enforced = _set_human_signoff(verdict)
        if not enforced:
            violations.append("human_signoff_enforcement_failed")
        elif not previously_flagged:
            logger.info(
                "Enforced mandatory human sign-off on committee verdict",
                extra={
                    "requested_amount_inr": requested_amount_inr,
                    "threshold_inr": threshold_inr,
                    "decision": decision_normalized,
                },
            )

    if violations:
        logger.warning(
            "Output guardrail blocked committee verdict",
            extra={"violations": violations, "decision": decision_normalized},
        )
        return False, violations

    return True, []

