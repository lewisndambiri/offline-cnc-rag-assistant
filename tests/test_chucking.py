from cnc_rag.ingestion.pdf import chunk_text, clean_text


def test_clean_text_collapses_extra_spacing():
    assert clean_text("A   B\n\n\nC") == "A B\n\nC"


def test_chunk_text_splits_long_text():
    text = "\n\n".join(["alpha " * 80, "beta " * 80, "gamma " * 80])
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)

