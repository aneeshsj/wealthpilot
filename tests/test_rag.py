"""Unit and integration-style tests for WealthPilot hybrid RAG pipeline (src/rag.py)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import src.rag as rag
from src.rag import (
    ABSTENTION_RESPONSE,
    PolicyChunk,
    RetrievedChunk,
    answer_policy_question,
    build_index,
    close_qdrant_clients,
    retrieve_policy,
)

logger = logging.getLogger(__name__)


class _InMemoryQdrantClient:
    """Lightweight in-memory stand-in for QdrantClient local disk mode in unit tests."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.collections: dict[str, dict[str, Any]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def delete_collection(self, collection_name: str) -> None:
        self.collections.pop(collection_name, None)

    def create_collection(self, collection_name: str, vectors_config: Any) -> None:
        self.collections[collection_name] = {"vectors_config": vectors_config, "points": []}

    def upsert(
        self, collection_name: str, points: list[Any], wait: bool = True
    ) -> None:
        del wait
        stored = self.collections[collection_name]["points"]
        by_id = {p.id: p for p in stored}
        for point in points:
            by_id[point.id] = point
        self.collections[collection_name]["points"] = list(by_id.values())

    def count(self, collection_name: str, exact: bool = True) -> SimpleNamespace:
        del exact
        return SimpleNamespace(count=len(self.collections[collection_name]["points"]))

    def scroll(
        self,
        collection_name: str,
        limit: int = 256,
        offset: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> tuple[list[SimpleNamespace], int | None]:
        del with_payload, with_vectors
        points = self.collections[collection_name]["points"]
        start = int(offset or 0)
        batch = points[start : start + limit]
        records = [
            SimpleNamespace(id=p.id, payload=dict(p.payload)) for p in batch
        ]
        next_offset = start + limit if start + limit < len(points) else None
        return records, next_offset

    def query_points(
        self,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> SimpleNamespace:
        del with_payload
        points = self.collections[collection_name]["points"]

        def _dot(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True))

        scored = sorted(
            points,
            key=lambda p: _dot(list(p.vector), query),
            reverse=True,
        )[:limit]
        hits = [
            SimpleNamespace(id=p.id, score=_dot(list(p.vector), query), payload=dict(p.payload))
            for p in scored
        ]
        return SimpleNamespace(points=hits)

    def close(self) -> None:
        return None


def _fake_embedding_vector(text: str) -> list[float]:
    """Generate deterministic 4D embeddings reflecting key SME credit topics."""
    lowered = text.lower()
    return [
        1.0 if "dscr" in lowered or "coverage" in lowered else 0.1,
        1.0 if "gst" in lowered or "mismatch" in lowered else 0.1,
        1.0 if "collateral" in lowered or "ltv" in lowered else 0.1,
        1.0 if "fraud" in lowered or "signoff" in lowered else 0.1,
    ]


@pytest.fixture(autouse=True)
def _reset_rag_caches() -> Any:
    """Ensure cached Qdrant/BM25/reranker state does not leak across tests."""
    close_qdrant_clients()
    rag._BM25_CACHE.clear()
    rag._RERANKER_CACHE.clear()
    rag._FASTEMBED_CACHE.clear()
    yield
    close_qdrant_clients()
    rag._BM25_CACHE.clear()


def _create_sample_corpus(root: Path, md_count: int = 8, pdf_count: int = 5) -> None:
    """Populate a temporary DATA_DIR with 13 Markdown + PDF policy files."""
    md_dir = root / "corpus" / "markdown"
    pdf_dir = root / "corpus" / "pdf"
    md_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    topics = [
        ("01_dscr_policy", "DSCR Policy", "Minimum DSCR", "Borrowers must maintain DSCR >= 1.25x."),
        ("02_gst_reconciliation", "GST Policy", "Revenue Mismatch", "GST mismatch over 10% triggers review."),
        ("03_collateral_ltv", "Collateral Policy", "Maximum LTV", "Commercial property LTV cap is 65%."),
        ("04_fraud_controls", "Fraud Policy", "Variance Hold", "Revenue variance over 25% requires fraud hold."),
        ("05_auto_approval", "Approval Matrix", "Auto Approval Limit", "Loans up to INR 2,500,000 may auto-approve."),
        ("06_human_signoff", "Approval Matrix", "Credit Committee", "Loans above INR 5,000,000 require signoff."),
        ("07_vintage_rules", "Eligibility Policy", "Business Vintage", "Minimum operating vintage is 24 months."),
        ("08_sector_caps", "Concentration Policy", "Sector Limits", "Single sector exposure capped at 15%."),
    ]

    for idx in range(md_count):
        stem, title, heading, body = topics[idx % len(topics)]
        long_body = f"{body}\n\n" + ("Additional SME credit underwriting guidelines. " * 25)
        (md_dir / f"{stem}_{idx}.md").write_text(
            f"# {title}\n\n## {heading}\n{long_body}\n",
            encoding="utf-8",
        )

    for idx in range(pdf_count):
        (pdf_dir / f"policy_doc_{idx + 1}.pdf").write_bytes(b"%PDF-1.4\n%fake\n")


def test_markdown_chunking_preserves_headings_and_overlap(tmp_path: Path) -> None:
    md_dir = tmp_path / "corpus" / "markdown"
    md_dir.mkdir(parents=True)
    md_file = md_dir / "sme_credit_policy.md"
    repeated_para = "Cash flow underwriting requires trailing 12-month bank statements. " * 22
    md_file.write_text(
        "# SME Credit Policy\n\n"
        "## Eligibility\n"
        "### Minimum DSCR\n"
        f"{repeated_para}\n",
        encoding="utf-8",
    )

    chunks = rag._chunk_markdown_document(
        md_path=md_file,
        base_dir=tmp_path,
        chunk_size=700,
        overlap=100,
    )

    assert len(chunks) >= 2
    assert all(isinstance(c, PolicyChunk) for c in chunks)
    assert all(c.doc_title == "SME Credit Policy" for c in chunks)
    assert all(c.section == "Eligibility > Minimum DSCR" for c in chunks)
    assert all(c.source_type == "markdown" for c in chunks)
    assert all(c.file_path == "corpus/markdown/sme_credit_policy.md" for c in chunks)
    assert all(len(c.text) <= 800 for c in chunks)


def test_pdf_extraction_falls_back_to_pdfplumber_when_pypdf_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_dir = tmp_path / "corpus" / "pdf"
    pdf_dir.mkdir(parents=True)
    pdf_file = pdf_dir / "collateral_rules.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%stub\n")

    def _failing_pypdf(_path: Path) -> tuple[str | None, list[str]]:
        raise RuntimeError("corrupt xref table")

    def _working_pdfplumber(_path: Path) -> tuple[str | None, list[str]]:
        return (
            "Collateral Valuation Standards",
            [
                "1. Property LTV Caps\n"
                "Commercial real estate collateral is capped at 65% LTV.\n"
                "2. Machinery Valuation\n"
                "Plant and machinery collateral is capped at 50% LTV."
            ],
        )

    monkeypatch.setattr(rag, "_extract_pdf_pages_pypdf", _failing_pypdf)
    monkeypatch.setattr(rag, "_extract_pdf_pages_pdfplumber", _working_pdfplumber)

    chunks = rag._chunk_pdf_document(
        pdf_path=pdf_file,
        base_dir=tmp_path,
        chunk_size=700,
        overlap=100,
    )

    assert len(chunks) == 2
    assert chunks[0].doc_title == "Collateral Valuation Standards"
    assert chunks[0].section == "1. Property LTV Caps"
    assert chunks[0].source_type == "pdf"
    assert chunks[0].file_path == "corpus/pdf/collateral_rules.pdf"
    assert chunks[1].section == "2. Machinery Valuation"


def test_build_index_all_13_docs_with_litellm_and_qdrant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    qdrant_dir = tmp_path / "qdrant"
    _create_sample_corpus(data_dir, md_count=8, pdf_count=5)

    monkeypatch.setattr(
        rag,
        "_extract_pdf_content",
        lambda p: (
            f"PDF {p.stem}",
            "1. Underwriting Rules\nGST mismatch threshold is 10% for SME loans.",
        ),
    )
    monkeypatch.setattr(rag, "QdrantClient", _InMemoryQdrantClient)

    captured_models: list[str] = []

    def _mock_litellm_embedding(**kwargs: Any) -> SimpleNamespace:
        captured_models.append(str(kwargs["model"]))
        batch = kwargs["input"]
        return SimpleNamespace(
            data=[
                {"index": idx, "embedding": _fake_embedding_vector(text)}
                for idx, text in enumerate(batch)
            ]
        )

    monkeypatch.setattr(rag.litellm, "embedding", _mock_litellm_embedding)
    monkeypatch.setattr(
        rag,
        "settings",
        SimpleNamespace(
            DATA_DIR=str(data_dir),
            QDRANT_PATH=str(qdrant_dir),
            EMBEDDING_MODEL="models/text-embedding-004",
            LLM_MODEL="gemini/gemini-2.5-flash",
            GEMINI_API_KEY="test-gemini-key",
        ),
    )

    stats = build_index(data_dir=data_dir, qdrant_path=qdrant_dir)

    assert stats.document_count == 13
    assert stats.markdown_count == 8
    assert stats.pdf_count == 5
    assert stats.chunk_count >= 13
    assert stats.vector_dim == 4
    assert stats.embedding_backend == "litellm"
    assert all(m == "gemini/text-embedding-004" for m in captured_models)


def test_build_index_falls_back_to_fastembed_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    qdrant_dir = tmp_path / "qdrant"
    _create_sample_corpus(data_dir, md_count=1, pdf_count=0)

    monkeypatch.setattr(rag, "QdrantClient", _InMemoryQdrantClient)
    monkeypatch.setattr(
        rag,
        "settings",
        SimpleNamespace(
            DATA_DIR=str(data_dir),
            QDRANT_PATH=str(qdrant_dir),
            EMBEDDING_MODEL="models/text-embedding-004",
            LLM_MODEL="gemini/gemini-2.5-flash",
            GEMINI_API_KEY=None,
        ),
    )
    monkeypatch.setattr(
        rag,
        "_embed_via_fastembed",
        lambda texts, _model: [_fake_embedding_vector(t) for t in texts],
    )

    stats = build_index(data_dir=data_dir, qdrant_path=qdrant_dir)
    assert stats.embedding_backend == "fastembed"
    assert stats.chunk_count >= 1


def test_retrieve_policy_fuses_dense_and_bm25_and_returns_reranked_top_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    qdrant_dir = tmp_path / "qdrant"
    md_dir = data_dir / "corpus" / "markdown"
    md_dir.mkdir(parents=True)

    # Create 12 distinct policy files so top-10 RRF + top-4 reranker are both exercised
    for idx in range(12):
        topic = "DSCR coverage ratio minimum 1.25x" if idx == 5 else f"General policy rule {idx}"
        (md_dir / f"doc_{idx:02d}.md").write_text(
            f"# Policy Doc {idx}\n\n## Section {idx}\n{topic} underwriting standard.\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(rag, "QdrantClient", _InMemoryQdrantClient)
    monkeypatch.setattr(
        rag,
        "settings",
        SimpleNamespace(
            DATA_DIR=str(data_dir),
            QDRANT_PATH=str(qdrant_dir),
            EMBEDDING_MODEL="models/text-embedding-004",
            LLM_MODEL="gemini/gemini-2.5-flash",
            GEMINI_API_KEY=None,
        ),
    )
    monkeypatch.setattr(
        rag,
        "_embed_via_fastembed",
        lambda texts, _model: [_fake_embedding_vector(t) for t in texts],
    )

    build_index(data_dir=data_dir, qdrant_path=qdrant_dir)

    # Cross-encoder intentionally boosts doc_09 above doc_05 to prove rerank ordering wins
    class _FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
            assert len(pairs) == 10  # Top 10 fused candidates sent to reranker
            scores: list[float] = []
            for _query, doc_text in pairs:
                if "Policy Doc 9" in doc_text:
                    scores.append(99.0)
                elif "Policy Doc 5" in doc_text:
                    scores.append(50.0)
                else:
                    scores.append(1.0)
            return scores

    monkeypatch.setattr(rag, "_get_reranker", lambda _model: _FakeCrossEncoder())

    results = retrieve_policy("What is the minimum DSCR coverage ratio?", qdrant_path=qdrant_dir)

    assert len(results) == 4
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].doc_title == "Policy Doc 9"
    assert results[0].rerank_score == 99.0
    assert results[1].doc_title == "Policy Doc 5"
    assert results[1].rerank_score == 50.0
    assert results == sorted(
        results, key=lambda r: (-r.rerank_score, -r.rrf_score, r.chunk_id)
    )


def test_retrieve_policy_raises_when_collection_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag, "QdrantClient", _InMemoryQdrantClient)
    with pytest.raises(RuntimeError, match="does not exist"):
        retrieve_policy("minimum DSCR", qdrant_path=tmp_path / "empty_qdrant")


def test_answer_policy_question_wraps_untrusted_context_and_cites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = RetrievedChunk(
        chunk_id="id-1",
        text="Minimum DSCR is 1.25x for term loans.",
        doc_title="DSCR Policy",
        section="2.1 Minimum DSCR",
        source_type="markdown",
        file_path="corpus/markdown/01_dscr.md",
        chunk_index=0,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.032,
        rerank_score=9.2,
    )
    monkeypatch.setattr(rag, "retrieve_policy", lambda **_kwargs: [chunk])

    captured_kwargs: dict[str, Any] = {}

    def _mock_completion(**kwargs: Any) -> SimpleNamespace:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="The minimum DSCR is 1.25x [DSCR Policy | 2.1 Minimum DSCR]."
                    )
                )
            ]
        )

    monkeypatch.setattr(rag.litellm, "completion", _mock_completion)
    monkeypatch.setattr(
        rag,
        "settings",
        SimpleNamespace(
            LLM_MODEL="gemini/gemini-2.5-flash",
            GEMINI_API_KEY="test-key",
        ),
    )

    result = answer_policy_question("What is the minimum DSCR?")

    system_msg = captured_kwargs["messages"][0]["content"]
    assert "<UNTRUSTED_POLICY_CONTEXT>" in system_msg
    assert "</UNTRUSTED_POLICY_CONTEXT>" in system_msg
    assert "[DSCR Policy | 2.1 Minimum DSCR]" in system_msg
    assert ABSTENTION_RESPONSE in system_msg
    assert captured_kwargs["model"] == "gemini/gemini-2.5-flash"
    assert result.citations == ["[DSCR Policy | 2.1 Minimum DSCR]"]
    assert "1.25x" in result.answer


def test_answer_policy_question_appends_missing_retrieved_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = RetrievedChunk(
        chunk_id="id-3",
        text="Loans above five million require human sign-off.",
        doc_title="Approval Matrix",
        section="Human Sign-Off",
        source_type="markdown",
        file_path="corpus/markdown/06_human_signoff.md",
        chunk_index=0,
        dense_rank=1,
        sparse_rank=1,
        rrf_score=0.032,
        rerank_score=9.2,
    )
    monkeypatch.setattr(rag, "retrieve_policy", lambda **_kwargs: [chunk])
    monkeypatch.setattr(
        rag.litellm,
        "completion",
        lambda **_kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Human sign-off is required."))]
        ),
    )
    monkeypatch.setattr(rag, "settings", SimpleNamespace(LLM_MODEL="test-model", GEMINI_API_KEY="test-key"))

    result = answer_policy_question("When is human sign-off required?")

    assert result.citations == ["[Approval Matrix | Human Sign-Off]"]
    assert "Sources: [Approval Matrix | Human Sign-Off]" in result.answer


def test_answer_policy_question_abstains_when_no_chunks_or_absent_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Case 1: Empty retrieval -> immediate abstention without calling LLM
    llm_spy = MagicMock()
    monkeypatch.setattr(rag, "retrieve_policy", lambda **_kwargs: [])
    monkeypatch.setattr(rag.litellm, "completion", llm_spy)

    empty_res = answer_policy_question("What is the crypto collateral haircut?")
    assert empty_res.answer == ABSTENTION_RESPONSE
    assert empty_res.citations == []
    llm_spy.assert_not_called()

    # Case 2: Retrieved chunks lack the answer -> LLM returns strict abstention
    chunk = RetrievedChunk(
        chunk_id="id-2",
        text="Commercial property LTV cap is 65%.",
        doc_title="Collateral Policy",
        section="Maximum LTV",
        source_type="pdf",
        file_path="corpus/pdf/collateral.pdf",
        chunk_index=0,
        dense_rank=1,
        sparse_rank=None,
        rrf_score=0.016,
        rerank_score=0.4,
    )
    monkeypatch.setattr(rag, "retrieve_policy", lambda **_kwargs: [chunk])
    with patch.object(
        rag.litellm,
        "completion",
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=ABSTENTION_RESPONSE))]
        ),
    ):
        absent_res = answer_policy_question("What is the yacht financing limit?")

    assert absent_res.answer == ABSTENTION_RESPONSE
    assert absent_res.citations == []

