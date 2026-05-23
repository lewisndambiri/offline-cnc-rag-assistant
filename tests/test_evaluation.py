from cnc_rag.evaluation import EvalCase, EvalResult, summarize


def test_summarize_computes_page_hit_rate_and_term_recall():
    results = [
        EvalResult(
            case=EvalCase(
                id="a",
                question="A?",
                expected_pages=[1],
                expected_terms=["alpha", "beta"],
            ),
            top_pages=[1, 2],
            page_hit=True,
            term_hits=["alpha"],
            top_score=0.9,
        ),
        EvalResult(
            case=EvalCase(
                id="b",
                question="B?",
                expected_pages=[3],
                expected_terms=["gamma"],
            ),
            top_pages=[4, 5],
            page_hit=False,
            term_hits=["gamma"],
            top_score=0.7,
        ),
    ]

    summary = summarize(results)

    assert summary["cases"] == 2
    assert summary["page_hit_rate"] == 0.5
    assert summary["mean_term_recall"] == 0.75
