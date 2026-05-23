from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cnc_rag.retrieval import search


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_pages: list[int]
    expected_terms: list[str]


@dataclass(frozen=True)
class EvalResult:
    case: EvalCase
    top_pages: list[int]
    page_hit: bool
    term_hits: list[str]
    top_score: float

    @property
    def term_recall(self) -> float:
        if not self.case.expected_terms:
            return 1.0
        return len(self.term_hits) / len(self.case.expected_terms)


def load_eval_cases(path: Path) -> list[EvalCase]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            expected_pages=item.get("expected_pages", []),
            expected_terms=[term.lower() for term in item.get("expected_terms", [])],
        )
        for item in data["questions"]
    ]


def evaluate_case(
    case: EvalCase,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
    top_k: int,
) -> EvalResult:
    results = search(case.question, index_path, metadata_path, model_name, top_k=top_k)
    joined_text = "\n".join(result["text"].lower() for result in results)
    top_pages = [int(result["page"]) for result in results]
    term_hits = [term for term in case.expected_terms if term in joined_text]

    return EvalResult(
        case=case,
        top_pages=top_pages,
        page_hit=any(page in case.expected_pages for page in top_pages),
        term_hits=term_hits,
        top_score=float(results[0]["score"]) if results else 0.0,
    )


def evaluate(
    eval_path: Path,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
    top_k: int = 5,
) -> list[EvalResult]:
    cases = load_eval_cases(eval_path)
    return [
        evaluate_case(case, index_path, metadata_path, model_name, top_k=top_k)
        for case in cases
    ]


def summarize(results: list[EvalResult]) -> dict:
    if not results:
        return {"cases": 0, "page_hit_rate": 0.0, "mean_term_recall": 0.0}

    page_hits = sum(1 for result in results if result.page_hit)
    mean_term_recall = sum(result.term_recall for result in results) / len(results)

    return {
        "cases": len(results),
        "page_hit_rate": page_hits / len(results),
        "mean_term_recall": mean_term_recall,
    }
