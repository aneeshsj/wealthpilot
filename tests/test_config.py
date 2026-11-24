import dataclasses
import logging
from pathlib import Path
import pytest

import config


@pytest.fixture
def clean_env(monkeypatch):
    keys = [
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "LLM_FALLBACK_PROVIDER",
        "GEMINI_MODEL",
        "EMBEDDING_MODEL",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "GROQ_API_MODEL",
        "DATA_DIR",
        "SQLITE_DB_PATH",
        "QDRANT_PATH",
        "LOG_LEVEL",
        "AUTO_APPROVE_MAX_INR",
        "HUMAN_SIGNOFF_THRESHOLD_INR",
        "REVENUE_MISMATCH_PCT",
        "FRAUD_HOLD_VARIANCE_PCT",
        "MAX_GRAPH_REVISIONS",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)


def test_default_settings(clean_env):
    s = config.Settings()
    assert s.LLM_MODEL == "gemini/gemini-flash-lite-latest"
    assert s.EMBEDDING_MODEL == "models/text-embedding-004"
    assert s.GEMINI_API_KEY is None
    assert s.DATA_DIR == "data/wealthpilot"
    assert s.AUTO_APPROVE_MAX_INR == 2_500_000
    assert s.HUMAN_SIGNOFF_THRESHOLD_INR == 5_000_000
    assert s.REVENUE_MISMATCH_PCT == 10.0
    assert s.FRAUD_HOLD_VARIANCE_PCT == 25.0
    assert s.MAX_GRAPH_REVISIONS == 3
    assert s.PROTECTED_PROXIES == config.PROTECTED_PROXIES
    assert s.PROTECTED_PROXIES is not config.PROTECTED_PROXIES


def test_env_overrides(clean_env, monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-pro")
    monkeypatch.setenv("AUTO_APPROVE_MAX_INR", "1000000")
    monkeypatch.setenv("REVENUE_MISMATCH_PCT", "12.5")
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key")

    s = config.Settings()
    assert s.LLM_MODEL == "gemini/gemini-2.5-pro"
    assert s.AUTO_APPROVE_MAX_INR == 1_000_000
    assert s.REVENUE_MISMATCH_PCT == 12.5
    assert s.GEMINI_API_KEY == "test-secret-key"


@pytest.mark.parametrize("bad_value", ["not_a_number", "   ", "12.34"])
def test_invalid_int_env_falls_back_with_warning(clean_env, monkeypatch, caplog, bad_value):
    monkeypatch.setenv("MAX_GRAPH_REVISIONS", bad_value)
    with caplog.at_level(logging.WARNING):
        s = config.Settings()

    assert s.MAX_GRAPH_REVISIONS == config.MAX_GRAPH_REVISIONS
    if bad_value.strip():
        assert "Invalid integer for MAX_GRAPH_REVISIONS" in caplog.text


@pytest.mark.parametrize("bad_value", ["invalid_float", "   "])
def test_invalid_float_env_falls_back_with_warning(clean_env, monkeypatch, caplog, bad_value):
    monkeypatch.setenv("FRAUD_HOLD_VARIANCE_PCT", bad_value)
    with caplog.at_level(logging.WARNING):
        s = config.Settings()

    assert s.FRAUD_HOLD_VARIANCE_PCT == config.FRAUD_HOLD_VARIANCE_PCT
    if bad_value.strip():
        assert "Invalid float for FRAUD_HOLD_VARIANCE_PCT" in caplog.text


def test_immutability_and_secret_masking(clean_env):
    s = config.Settings(GEMINI_API_KEY="super-secret")
    assert "super-secret" not in repr(s)

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.LOG_LEVEL = "DEBUG"


def test_post_init_exports_gemini_api_key(clean_env, monkeypatch):
    assert "GEMINI_API_KEY" not in config.os.environ
    config.Settings(GEMINI_API_KEY="injected-key")
    assert config.os.environ["GEMINI_API_KEY"] == "injected-key"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def test_path_properties(clean_env):
    s = config.Settings(
        DATA_DIR="/tmp/data",
        SQLITE_DB_PATH="/tmp/test.db",
        QDRANT_PATH="/tmp/qdrant",
    )
    assert s.data_dir_path == Path("/tmp/data")
    assert s.sqlite_db_path == Path("/tmp/test.db")
    assert s.qdrant_dir_path == Path("/tmp/qdrant")


def test_from_env_loads_file_without_overriding_existing(clean_env, monkeypatch, tmp_path):
    env_file = tmp_path / ".env.test"
    env_file.write_text("LOG_LEVEL=DEBUG\nAUTO_APPROVE_MAX_INR=750000\n")

    monkeypatch.setenv("LOG_LEVEL", "ERROR")

    s = config.Settings.from_env(env_file=env_file)
    assert s.LOG_LEVEL == "ERROR"
    assert s.AUTO_APPROVE_MAX_INR == 750_000


def test_configure_logging():
    config.configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING

    config.configure_logging(logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG
