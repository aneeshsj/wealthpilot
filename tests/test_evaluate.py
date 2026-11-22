from __future__ import annotations

from evals.evaluate import _intake_score, _load_cases, _matched_pair_score, _retrieval_score, _routing_score


def test_intake_evaluation_recomputes_real_records() -> None:
    result = _intake_score(_load_cases())

    assert result["status"] == "completed"
    assert result["records"] == 142
    assert result["schema_valid"] == 142
    assert result["dscr_consistent"] == 142
    assert result["missing_fields_consistent"] == 142
    assert result["accuracy"] == 1.0


def test_matched_pairs_are_explicitly_validated() -> None:
    result = _matched_pair_score(_load_cases())

    assert result["pairs"] == 3
    assert result["structurally_valid"] == 3
    assert result["fixture_accuracy"] == 1.0
    if "candidate_status" in result:
        assert result["candidate_status"].startswith("not_run")
    else:
        assert result["status"] == "candidate_matched_pair_agreement"
        assert result["agreements"] <= result["pairs"]


def test_evaluator_does_not_claim_live_retrieval_or_routing_without_candidates() -> None:
    cases = _load_cases()

    retrieval = _retrieval_score(cases)
    routing = _routing_score(cases)

    if "recall_at_k" in retrieval:
        assert retrieval["status"] == "candidate_answer_citation_recall"
        assert 0.0 <= retrieval["recall_at_k"] <= 1.0
        assert routing["status"] == "candidate_route_accuracy"
        assert 0.0 <= routing["accuracy"] <= 1.0
    else:
        assert retrieval["status"] == "corpus_coverage_only"
        assert routing["status"] == "not_run"
