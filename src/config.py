"""Centralized configuration and policy constants for WealthPilot."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Named SME underwriting policy constants
AUTO_APPROVE_MAX_INR: int = 2_500_000
HUMAN_SIGNOFF_THRESHOLD_INR: int = 5_000_000
REVENUE_MISMATCH_PCT: float = 10.0
FRAUD_HOLD_VARIANCE_PCT: float = 25.0
MAX_GRAPH_REVISIONS: int = 3
PROTECTED_PROXIES: list[str] = [
    "proprietor_name",
    "city",
    "application_language",
    "postal_code",
    "education_institution",
]


def _get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %d", key, raw, default)
        return default


def _get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", key, raw, default)
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid boolean for %s=%r; using default %s", key, raw, default)
    return default


def _configured_provider() -> str:
    """Return the active provider without coupling the app to one vendor."""
    return (os.getenv("LLM_PROVIDER") or "gemini").strip().lower()


def _configured_model() -> str:
    """Resolve an explicit model or the model belonging to the active provider."""
    explicit = os.getenv("LLM_MODEL")
    if explicit and explicit.strip():
        return explicit.strip()
    if _configured_provider() == "groq":
        return os.getenv("GROQ_API_MODEL", "openai/gpt-oss-120b")
    return os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")


def _configured_api_key() -> str | None:
    """Resolve an explicit key or the key belonging to the active provider."""
    explicit = os.getenv("LLM_API_KEY")
    if explicit and explicit.strip():
        return explicit.strip()
    if _configured_provider() == "groq":
        return os.getenv("GROQ_API_KEY") or None
    return os.getenv("GEMINI_API_KEY") or None


def litellm_model_name(model: str, provider: str | None = None) -> str:
    """Add LiteLLM's provider prefix when a provider-specific model needs one."""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "groq" and not model.startswith("groq/"):
        return f"groq/{model}"
    return model


@dataclass(frozen=True)
class Settings:
    """Application settings and underwriting policy thresholds loaded from environment."""

    LLM_MODEL: str = field(
        default_factory=_configured_model
    )
    LLM_PROVIDER: Optional[str] = field(default_factory=lambda: _configured_provider())
    LLM_API_BASE: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_BASE") or None)
    LLM_API_KEY: Optional[str] = field(
        default_factory=_configured_api_key,
        repr=False,
    )
    LLM_TEMPERATURE: float = field(default_factory=lambda: _get_env_float("LLM_TEMPERATURE", 0.0))
    LLM_MAX_TOKENS: int = field(default_factory=lambda: _get_env_int("LLM_MAX_TOKENS", 1024))
    LLM_TIMEOUT_SECONDS: float = field(
        default_factory=lambda: _get_env_float("LLM_TIMEOUT_SECONDS", 30.0)
    )
    LLM_FALLBACK_ENABLED: bool = field(
        default_factory=lambda: _get_env_bool("LLM_FALLBACK_ENABLED", False)
    )
    LLM_FALLBACK_PROVIDER: Optional[str] = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_PROVIDER") or None
    )
    EMBEDDING_MODEL: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
    )
    GEMINI_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or None,
        repr=False,
    )
    GEMINI_MODEL: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash")
    )
    GROQ_API_KEY: Optional[str] = field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None,
        repr=False,
    )
    GROQ_API_MODEL: str = field(
        default_factory=lambda: os.getenv("GROQ_API_MODEL", "openai/gpt-oss-120b")
    )

    DATA_DIR: str = field(
        default_factory=lambda: os.getenv("DATA_DIR", "data/wealthpilot")
    )
    SQLITE_DB_PATH: str = field(
        default_factory=lambda: os.getenv("SQLITE_DB_PATH", "wealthpilot.db")
    )
    QDRANT_PATH: str = field(
        default_factory=lambda: os.getenv("QDRANT_PATH", "./qdrant_data")
    )
    QDRANT_URL: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_URL") or None
    )
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    QDRANT_COLLECTION_NAME: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_NAME", "wealthpilot_policy_docs")
    )
    MCP_POLICY_RESOURCE_URI: str = field(
        default_factory=lambda: os.getenv(
            "MCP_POLICY_RESOURCE_URI", "policy://fair-lending-standard"
        )
    )
    MCP_ENABLED: bool = field(default_factory=lambda: _get_env_bool("MCP_ENABLED", True))
    GRAPH_RECURSION_LIMIT: int = field(
        default_factory=lambda: _get_env_int("GRAPH_RECURSION_LIMIT", 25)
    )
    DSCR_TOLERANCE: float = field(default_factory=lambda: _get_env_float("DSCR_TOLERANCE", 0.05))

    AUTO_APPROVE_MAX_INR: int = field(
        default_factory=lambda: _get_env_int("AUTO_APPROVE_MAX_INR", AUTO_APPROVE_MAX_INR)
    )
    HUMAN_SIGNOFF_THRESHOLD_INR: int = field(
        default_factory=lambda: _get_env_int(
            "HUMAN_SIGNOFF_THRESHOLD_INR", HUMAN_SIGNOFF_THRESHOLD_INR
        )
    )
    REVENUE_MISMATCH_PCT: float = field(
        default_factory=lambda: _get_env_float(
            "REVENUE_MISMATCH_PCT", REVENUE_MISMATCH_PCT
        )
    )
    FRAUD_HOLD_VARIANCE_PCT: float = field(
        default_factory=lambda: _get_env_float(
            "FRAUD_HOLD_VARIANCE_PCT", FRAUD_HOLD_VARIANCE_PCT
        )
    )
    MAX_GRAPH_REVISIONS: int = field(
        default_factory=lambda: _get_env_int("MAX_GRAPH_REVISIONS", MAX_GRAPH_REVISIONS)
    )
    PROTECTED_PROXIES: list[str] = field(
        default_factory=lambda: list(PROTECTED_PROXIES)
    )

    def __post_init__(self) -> None:
        if self.LLM_API_KEY and not os.getenv("LLM_API_KEY"):
            os.environ["LLM_API_KEY"] = self.LLM_API_KEY
        if self.GEMINI_API_KEY and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = self.GEMINI_API_KEY
        if self.GROQ_API_KEY and not os.getenv("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = self.GROQ_API_KEY

    @property
    def data_dir_path(self) -> Path:
        return Path(self.DATA_DIR)

    @property
    def sqlite_db_path(self) -> Path:
        return Path(self.SQLITE_DB_PATH)

    @property
    def qdrant_dir_path(self) -> Path:
        return Path(self.QDRANT_PATH)

    @classmethod
    def from_env(cls, env_file: Optional[str | Path] = None) -> Settings:
        """Reload environment variables (optionally from a specific file) and build Settings."""
        load_dotenv(dotenv_path=env_file, override=False)
        return cls()


def configure_logging(level: Optional[str | int] = None) -> None:
    """Configure root logging cleanly without duplicate handlers."""
    target_level = level if level is not None else settings.LOG_LEVEL
    if isinstance(target_level, str):
        resolved_level = getattr(logging, target_level.upper(), logging.INFO)
    else:
        resolved_level = target_level

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=resolved_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(resolved_level)


settings: Settings = Settings.from_env()

