"""Unit and persistence tests for WealthPilot's SQLite memory subsystem (src/memory.py).

Compatible with both ``python3 -m unittest test_memory.py`` and ``pytest test_memory.py``.
"""

from __future__ import annotations

import ast
import csv
import logging
import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from unittest.mock import patch

import src.memory as memory_module
from src.memory import (
    DEFAULT_WEALTHPILOT_SOPS,
    EpisodicMemory,
    ProceduralMemory,
    SemanticMemory,
    WealthPilotMemoryManager,
    WorkingMemory,
    WorkingMessage,
)

logger = logging.getLogger(__name__)


def _write_mock_past_decisions(csv_path: Path) -> None:
    """Create a realistic mock past_decisions.csv fixture."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "applicant_id",
        "thread_id",
        "timestamp",
        "summary",
        "outcome",
        "sector",
    ]
    rows = [
        {
            "event_id": "evt-100-1",
            "applicant_id": "SME-100",
            "thread_id": "hist-100",
            "timestamp": "2025-04-10T09:00:00+00:00",
            "summary": "Approved equipment term loan with verified DSCR 1.48x",
            "outcome": "APPROVE",
            "sector": "logistics",
        },
        {
            "event_id": "evt-100-2",
            "applicant_id": "SME-100",
            "thread_id": "hist-100b",
            "timestamp": "2025-08-22T14:30:00+00:00",
            "summary": "Referred seasonal working capital line pending AR aging schedule",
            "outcome": "REFER",
            "sector": "logistics",
        },
        {
            "event_id": "evt-200-1",
            "applicant_id": "SME-200",
            "thread_id": "hist-200",
            "timestamp": "2025-09-01T11:15:00+00:00",
            "summary": "Declined expansion loan due to DSCR 0.89x below hospitality floor",
            "outcome": "DECLINE",
            "sector": "hospitality",
        },
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestMemoryHygiene(unittest.TestCase):
    """Static verification of coding standards in src/memory.py."""

    def test_no_print_calls_in_memory_module(self) -> None:
        source_path = Path(__file__).resolve().parent.parent / "src" / "memory.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                self.fail(f"Forbidden print() call found at line {node.lineno} in src/memory.py")


class TestWorkingMemory(unittest.TestCase):
    """Tests for WorkingMemory rolling window, summarization, and applicant isolation."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp_dir.name) / "working.db"
        self.memory = WorkingMemory(db_path=self.db_path)

    def tearDown(self) -> None:
        self.memory.close()
        self._tmp_dir.cleanup()

    def test_rolling_window_keeps_last_eight_and_summarizes_earlier(self) -> None:
        for idx in range(1, 13):
            role = "user" if idx % 2 == 1 else "assistant"
            self.memory.save_message(
                "SME-100",
                "thread-alpha",
                role,
                f"Turn {idx}: reviewing Q{((idx - 1) % 4) + 1} cashflow and DSCR",
            )

        ctx = self.memory.retrieve_context("SME-100", "thread-alpha")
        self.assertEqual(len(ctx.messages), 8)
        self.assertEqual(ctx.summarized_message_count, 4)
        self.assertEqual(ctx.messages[0].content.split(":")[0], "Turn 5")
        self.assertEqual(ctx.messages[-1].content.split(":")[0], "Turn 12")
        self.assertIn("Turn 1:", ctx.summary)
        self.assertIn("Turn 4:", ctx.summary)

    def test_injected_summarizer_callback_is_used(self) -> None:
        def custom_summarizer(evicted: Sequence[WorkingMessage], prior: str) -> str:
            joined = "|".join(m.content for m in evicted)
            return f"{prior}::{joined}".strip(":")

        custom_mem = WorkingMemory(
            db_path=Path(self._tmp_dir.name) / "custom_working.db",
            window_size=2,
            summarizer=custom_summarizer,
        )
        try:
            custom_mem.save_message("SME-100", "t1", "user", "first")
            custom_mem.save_message("SME-100", "t1", "assistant", "second")
            custom_mem.save_message("SME-100", "t1", "user", "third")
            ctx = custom_mem.retrieve_context("SME-100", "t1")
            self.assertEqual(ctx.summary, "first")
            self.assertEqual([m.content for m in ctx.messages], ["second", "third"])
        finally:
            custom_mem.close()

    def test_strict_applicant_isolation_on_shared_thread_id(self) -> None:
        self.memory.save_message("SME-100", "shared-thread", "user", "SME-100 revenue is $2.4M")
        self.memory.save_message("SME-200", "shared-thread", "user", "SME-200 collateral is $500k")

        ctx_100 = self.memory.retrieve_context("SME-100", "shared-thread")
        ctx_200 = self.memory.retrieve_context("SME-200", "shared-thread")

        self.assertEqual(len(ctx_100.messages), 1)
        self.assertEqual(ctx_100.messages[0].content, "SME-100 revenue is $2.4M")
        self.assertEqual(len(ctx_200.messages), 1)
        self.assertEqual(ctx_200.messages[0].content, "SME-200 collateral is $500k")

        deleted = self.memory.clear_thread("SME-100", "shared-thread")
        self.assertEqual(deleted, 1)
        self.assertEqual(len(self.memory.retrieve_context("SME-100", "shared-thread").messages), 0)
        self.assertEqual(len(self.memory.retrieve_context("SME-200", "shared-thread").messages), 1)

    def test_validation_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.memory.save_message("", "t1", "user", "hello")
        with self.assertRaises(ValueError):
            self.memory.save_message("SME-100", "t1", "invalid_role", "hello")
        with self.assertRaises(ValueError):
            self.memory.save_message("SME-100", "t1", "user", "   ")


class TestEpisodicMemory(unittest.TestCase):
    """Tests for EpisodicMemory CSV seeding, logging, relevance, and isolation."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp_dir.name) / "episodic.db"
        self.csv_path = Path(self._tmp_dir.name) / "past_decisions.csv"
        _write_mock_past_decisions(self.csv_path)
        self.memory = EpisodicMemory(db_path=self.db_path, seed_csv_path=self.csv_path)

    def tearDown(self) -> None:
        self.memory.close()
        self._tmp_dir.cleanup()

    def test_seeds_from_csv_once_and_isolates_by_applicant(self) -> None:
        events_100 = self.memory.retrieve_events("SME-100", limit=10)
        events_200 = self.memory.retrieve_events("SME-200", limit=10)

        self.assertEqual(len(events_100), 2)
        self.assertTrue(all(ev.applicant_id == "SME-100" for ev in events_100))
        self.assertEqual(len(events_200), 1)
        self.assertEqual(events_200[0].outcome, "DECLINE")

        # Re-instantiating against the same SQLite DB must not duplicate seeded rows.
        reopened = EpisodicMemory(db_path=self.db_path, seed_csv_path=self.csv_path)
        try:
            self.assertEqual(len(reopened.retrieve_events("SME-100", limit=10)), 2)
        finally:
            reopened.close()

    def test_save_event_and_filtered_retrieval(self) -> None:
        saved = self.memory.save_event(
            applicant_id="SME-100",
            thread_id="thread-new",
            summary="Verified bank statements and confirmed DSCR 1.52x with warehouse lien",
            outcome="APPROVE",
            metadata={"dscr": 1.52},
        )
        self.assertEqual(saved.applicant_id, "SME-100")

        approved = self.memory.retrieve_events("SME-100", outcome="APPROVE", limit=5)
        self.assertEqual(len(approved), 2)
        self.assertTrue(all(ev.outcome == "APPROVE" for ev in approved))

        relevant = self.memory.retrieve_events("SME-100", query="warehouse lien", limit=1)
        self.assertEqual(len(relevant), 1)
        self.assertEqual(relevant[0].event_id, saved.event_id)

        # SME-200 must never see SME-100's new event.
        self.assertEqual(len(self.memory.retrieve_events("SME-200", limit=10)), 1)

    def test_missing_seed_csv_is_handled_gracefully(self) -> None:
        with self.assertLogs("src.memory", level="WARNING"):
            mem = EpisodicMemory(
                db_path=Path(self._tmp_dir.name) / "no_csv.db",
                seed_csv_path=Path(self._tmp_dir.name) / "missing.csv",
            )
        try:
            self.assertEqual(mem.retrieve_events("SME-999"), [])
        finally:
            mem.close()


class TestSemanticMemory(unittest.TestCase):
    """Tests for SemanticMemory upsert overwrites, relevance, and applicant isolation."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.memory = SemanticMemory(db_path=Path(self._tmp_dir.name) / "semantic.db")

    def tearDown(self) -> None:
        self.memory.close()
        self._tmp_dir.cleanup()

    def test_save_fact_overwrites_cleanly_and_isolates_applicants(self) -> None:
        self.memory.save_fact("SME-100", "Sector", "Manufacturing")
        self.memory.save_fact("SME-100", "sector", "Precision Manufacturing")
        self.memory.save_fact("SME-200", "sector", "Retail")

        facts_100 = self.memory.retrieve_facts("SME-100", limit=10)
        self.assertEqual(len(facts_100), 1)
        self.assertEqual(facts_100[0].fact_key, "sector")
        self.assertEqual(facts_100[0].fact_value, "Precision Manufacturing")

        self.assertEqual(self.memory.retrieve_fact_map("SME-200"), {"sector": "Retail"})

    def test_batch_save_query_keys_and_delete(self) -> None:
        self.memory.save_facts(
            "SME-100",
            {
                "annual_revenue_usd": 3200000,
                "verified_dscr": 1.44,
                "years_in_operation": 7,
            },
        )

        subset = self.memory.retrieve_facts("SME-100", keys=["verified_dscr"], limit=5)
        self.assertEqual(len(subset), 1)
        self.assertEqual(subset[0].fact_value, "1.44")

        ranked = self.memory.retrieve_facts("SME-100", query="dscr ratio", limit=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].fact_key, "verified_dscr")

        # Deleting SME-100's fact does not touch SME-200.
        self.memory.save_fact("SME-200", "verified_dscr", 1.15)
        self.assertTrue(self.memory.delete_fact("SME-100", "verified_dscr"))
        self.assertFalse(self.memory.delete_fact("SME-100", "verified_dscr"))
        self.assertEqual(self.memory.retrieve_fact_map("SME-200"), {"verified_dscr": "1.15"})


class TestProceduralMemory(unittest.TestCase):
    """Tests for ProceduralMemory default WealthPilot SOPs and rule versioning."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.memory = ProceduralMemory(db_path=Path(self._tmp_dir.name) / "procedural.db")

    def tearDown(self) -> None:
        self.memory.close()
        self._tmp_dir.cleanup()

    def test_initializes_default_wealthpilot_sops(self) -> None:
        rules = self.memory.retrieve_rules(limit=10)
        self.assertEqual(len(rules), len(DEFAULT_WEALTHPILOT_SOPS))
        self.assertEqual([r.step_order for r in rules], [1, 2, 3, 4, 5])

        joined_instructions = " ".join(r.instruction.lower() for r in rules)
        self.assertIn("demographic proxies", joined_instructions)
        self.assertIn("sector dscr floor", joined_instructions)

    def test_rule_versioning_and_filtering(self) -> None:
        v2 = self.memory.save_rule(
            rule_id="WP-SOP-003",
            step_order=3,
            task_type="dscr_compliance",
            instruction="Enforce sector DSCR floor >= 1.25x and stress-test +200bps rate shock.",
        )
        self.assertEqual(v2.version, 2)
        self.assertTrue(v2.is_active)

        active_dscr = self.memory.retrieve_rules(task_type="dscr_compliance", limit=5)
        self.assertEqual(len(active_dscr), 1)
        self.assertEqual(active_dscr[0].version, 2)

        history = self.memory.retrieve_rule_versions("WP-SOP-003")
        self.assertEqual([r.version for r in history], [2, 1])
        self.assertTrue(history[0].is_active)
        self.assertFalse(history[1].is_active)

        query_hits = self.memory.retrieve_rules(query="demographic proxies", limit=3)
        self.assertGreaterEqual(len(query_hits), 1)
        self.assertEqual(query_hits[0].rule_id, "WP-SOP-001")


class TestPersistenceAndManager(unittest.TestCase):
    """End-to-end tests on the real SQLite DB with per-run isolation and sandbox fallback."""

    def test_resolve_db_path_precedence(self) -> None:
        fake_settings = SimpleNamespace(SQLITE_DB_PATH="/tmp/settings_wealthpilot.db")
        with patch.object(memory_module, "_app_settings", fake_settings):
            with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/env_wealthpilot.db"}, clear=False):
                self.assertEqual(
                    memory_module._resolve_db_path("/tmp/explicit.db"),
                    Path("/tmp/explicit.db"),
                )
                self.assertEqual(
                    memory_module._resolve_db_path(None),
                    Path("/tmp/settings_wealthpilot.db"),
                )

        with patch.object(memory_module, "_app_settings", None):
            with patch.dict(os.environ, {"SQLITE_DB_PATH": "/tmp/env_wealthpilot.db"}, clear=False):
                self.assertEqual(
                    memory_module._resolve_db_path(None),
                    Path("/tmp/env_wealthpilot.db"),
                )

    def _run_restart_and_isolation_assertions(
        self,
        db_path: str | Path | None,
        *,
        seed_csv_path: str | Path | None = None,
    ) -> None:
        run_token = uuid.uuid4().hex[:10]
        applicant_1 = f"TEST-SME-100-{run_token}"
        applicant_2 = f"TEST-SME-200-{run_token}"
        thread_id = f"test-underwrite-{run_token}"

        mgr_kwargs: dict[str, Path | str] = {}
        if seed_csv_path is not None:
            mgr_kwargs["seed_csv_path"] = seed_csv_path

        mgr = WealthPilotMemoryManager(db_path=db_path, **mgr_kwargs)
        try:
            for idx in range(1, 11):
                mgr.working.save_message(
                    applicant_1,
                    thread_id,
                    "user" if idx % 2 else "assistant",
                    f"Check {idx}: DSCR and collateral coverage",
                )
            mgr.working.save_message(applicant_2, thread_id, "user", "Other SME message")
            mgr.semantic.save_fact(applicant_1, "dscr", "1.50")
            mgr.semantic.save_fact(applicant_2, "dscr", "0.85")
            logged_event = mgr.episodic.save_event(
                applicant_1,
                thread_id,
                "Credit committee cleared facility",
                "APPROVE",
            )
        finally:
            mgr.close()

        # Simulate fresh process startup reopening the same persistent SQLite database.
        restarted = WealthPilotMemoryManager(db_path=db_path, **mgr_kwargs)
        try:
            bundle_1 = restarted.retrieve_applicant_bundle(
                applicant_1,
                thread_id,
                query="DSCR",
                episodic_limit=5,
                semantic_limit=5,
                procedural_limit=5,
            )
            self.assertEqual(len(bundle_1["working"].messages), 8)
            self.assertEqual(bundle_1["working"].summarized_message_count, 2)
            self.assertEqual(len(bundle_1["episodic"]), 1)
            self.assertEqual(bundle_1["episodic"][0].event_id, logged_event.event_id)
            self.assertEqual(bundle_1["episodic"][0].applicant_id, applicant_1)
            self.assertEqual(
                {f.fact_key: f.fact_value for f in bundle_1["semantic"]},
                {"dscr": "1.50"},
            )
            self.assertGreaterEqual(len(bundle_1["procedural"]), 1)

            bundle_2 = restarted.retrieve_applicant_bundle(applicant_2, thread_id)
            self.assertEqual(len(bundle_2["working"].messages), 1)
            self.assertEqual(bundle_2["working"].messages[0].content, "Other SME message")
            self.assertEqual(len(bundle_2["episodic"]), 0)
            self.assertEqual(
                {f.fact_key: f.fact_value for f in bundle_2["semantic"]},
                {"dscr": "0.85"},
            )
        finally:
            # Clean up synthetic test rows so the real database remains unpolluted.
            try:
                restarted.working.clear_thread(applicant_1, thread_id)
                restarted.working.clear_thread(applicant_2, thread_id)
                restarted.semantic.delete_fact(applicant_1, "dscr")
                restarted.semantic.delete_fact(applicant_2, "dscr")
                with restarted._backend.lock:
                    conn = restarted._backend.connection
                    with conn:
                        conn.execute(
                            "DELETE FROM episodic_events WHERE applicant_id IN (?, ?)",
                            (applicant_1, applicant_2),
                        )
            finally:
                restarted.close()

    def test_survives_process_restart_and_isolates_bundle(self) -> None:
        try:
            self._run_restart_and_isolation_assertions(db_path=None)
        except (OSError, sqlite3.OperationalError) as exc:
            logger.warning(
                "Real SQLite database unavailable (%s); falling back to sandboxed DB",
                exc,
            )
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                sandbox_db = tmp_dir / "nested" / "wealthpilot_sandbox.db"
                sandbox_csv = tmp_dir / "past_decisions.csv"
                _write_mock_past_decisions(sandbox_csv)
                self._run_restart_and_isolation_assertions(
                    db_path=sandbox_db,
                    seed_csv_path=sandbox_csv,
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main()
