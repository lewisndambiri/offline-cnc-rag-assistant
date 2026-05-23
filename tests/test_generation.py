from cnc_rag.generation.ollama import (
    build_prompt,
    ensure_cited_lines,
    format_source_list,
    stream_answer_with_ollama,
)


def test_prompt_uses_source_citation_markers():
    contexts = [
        {
            "citation": "S1",
            "document_title": "Manual",
            "page": 12,
            "score": 0.9,
            "text": "Use MDI for unsaved blocks of code.",
        }
    ]

    prompt = build_prompt("How do I use MDI?", contexts)

    assert "[S1: Manual, page 12]" in prompt
    assert "Every factual sentence or numbered step must end" in prompt


def test_format_source_list_prints_page_and_score():
    contexts = [
        {
            "citation": "S1",
            "document_title": "Manual",
            "page": 12,
            "score": 0.9,
        }
    ]

    assert format_source_list(contexts) == "[S1] Manual, page 12 (score 0.900)"


def test_ensure_cited_lines_appends_missing_trailing_citation():
    contexts = [{"citation": "S2"}]
    answer = "1. Press [ALTER] [S2]. Only the O number is required."

    assert ensure_cited_lines(answer, contexts).endswith("[S2]")


def test_stream_answer_with_ollama_yields_tokens(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield b'{"response": "Press ", "done": false}'
            yield b'{"response": "[MDI]. [S1]", "done": true}'

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("cnc_rag.generation.ollama.requests.post", fake_post)

    tokens = list(stream_answer_with_ollama("How?", [], "mistral"))

    assert tokens == ["Press ", "[MDI]. [S1]"]
