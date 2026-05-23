from cnc_rag.retrieval.index import lexical_score, procedural_score, tokenize


def test_tokenize_removes_common_question_words():
    assert tokenize("How do I set a tool offset?") == ["set", "tool", "offset"]


def test_lexical_score_rewards_matching_terms():
    score = lexical_score(
        "How do I set a tool offset?",
        "Press OFFSET to view the tool offset values and set tool length geometry.",
    )

    assert score > 0.6


def test_procedural_score_rewards_step_cues():
    score = procedural_score(
        "How do I set a tool offset?",
        "Press OFFSET. Highlight the TOOL OFFSET field. Use the jog handle.",
    )

    assert score > 0.5


def test_procedural_score_penalizes_settings_lists():
    settings_score = procedural_score(
        "How do I set a tool offset?",
        "List of Settings. This setting changes the way PART ZERO SET works.",
    )

    assert settings_score == 0.0
