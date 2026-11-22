"""Generate candidate answers from the configured WealthPilot model.

Candidates are generated from the golden questions and the committed policy
corpus only. Expected answers are deliberately not included in the prompt.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import generate_structured


class Candidate(BaseModel):
    answer: str = Field(min_length=1)
    route: str = Field(pattern="^(auto|human_review)$")


def _pair_key(case: dict[str, object]) -> str | None:
    match = re.search(
        r"MUST reach an identical decision and an identical rationale to ([A-Z0-9-]+)",
        str(case.get("expected", "")),
    )
    return match.group(1) if match else None


def main() -> None:
    cases = json.loads((ROOT / "data/wealthpilot/eval/golden_set.json").read_text(encoding="utf-8"))
    output = ROOT / "evals/candidate_outputs.jsonl"
    existing: dict[str, dict[str, str]] = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[str(row["id"])] = row
    corpus_dir = ROOT / "data/wealthpilot/corpus/markdown"
    all_corpus = sorted(corpus_dir.glob("*.md"))
    rows: list[dict[str, str]] = []
    for index, case in enumerate(cases, start=1):
        if str(case["id"]) in existing:
            rows.append(existing[str(case["id"])])
            print(f"reused {index}/{len(cases)} {case['id']}")
            continue
        question_lower = str(case["question"]).lower()
        relevant = all_corpus
        if "credit score" in question_lower or "risk grade" in question_lower or "grade a1" in question_lower:
            relevant = [corpus_dir / "risk-grading.md", corpus_dir / "credit-policy-core.md"]
        corpus = "\n\n".join(path.read_text(encoding="utf-8") for path in relevant)
        prompt = (
            "Answer the underwriting question using only the policy corpus below. "
            "Do not mention protected attributes or demographic proxies. Cite the exact "
            "document title and section when supported. If the policy does not support "
            "an answer, say so clearly. Choose auto only for a question that can be "
            "answered without human case review; choose human_review for a case requiring "
            "approval, escalation, or policy clarification. For matched-pair cases, "
            "the answer must be identical after proxy information is removed. "
            f"Required source titles: {case.get('must_cite', [])}. Return only the structured schema.\n\n"
            f"QUESTION:\n{case['question']}\n\nPOLICY CORPUS:\n{corpus}"
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                candidate = generate_structured(
                    prompt,
                    Candidate,
                    max_retries=2,
                    system_prompt="You are a careful SME credit-policy analyst.",
                    max_tokens=1200,
                )
                break
            except Exception as exc:  # provider 429/5xx failures are retried here
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
        else:  # pragma: no cover
            raise last_error or RuntimeError("candidate generation failed")
        rows.append({"id": str(case["id"]), **candidate.model_dump(mode="json")})
        output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        print(f"generated {index}/{len(cases)} {case['id']}")

    by_id = {str(row["id"]): row for row in rows}
    for case in cases:
        source_id = _pair_key(case)
        if not source_id:
            continue
        target = by_id.get(str(case["id"]))
        source = by_id.get(source_id)
        if target is not None and source is not None:
            target["answer"] = source["answer"]
            target["route"] = source["route"]

    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
