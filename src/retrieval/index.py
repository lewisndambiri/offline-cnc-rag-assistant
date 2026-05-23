from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from requests.exceptions import RequestException
from sentence_transformers import SentenceTransformer


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the local embedding model cannot be loaded."""


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "with",
}


def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@lru_cache(maxsize=2)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name, device="cpu", local_files_only=True)
    except (OSError, RuntimeError, RequestException) as exc:
        raise EmbeddingModelLoadError(
            "Could not load the embedding model from the local cache. If this is the first "
            "run, connect to the internet once and run `cnc-rag ingest` so "
            "sentence-transformers can download and cache the model. For a fully offline "
            "setup, pass a local model path with `--embedding-model /path/to/model`."
        ) from exc


def embed_texts(model_name: str, texts: list[str]) -> np.ndarray:
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return np.asarray(embeddings, dtype="float32")


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def lexical_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0

    text_terms = set(tokenize(text))
    hits = query_terms & text_terms
    coverage = len(hits) / len(query_terms)

    query_phrase = " ".join(tokenize(query))
    text_normalized = " ".join(tokenize(text))
    phrase_bonus = 0.15 if query_phrase and query_phrase in text_normalized else 0.0

    return min(1.0, coverage + phrase_bonus)


def procedural_score(query: str, text: str) -> float:
    query_normalized = " ".join(tokenize(query))
    is_action_question = query_normalized.startswith(("how", "set", "select", "use", "edit"))
    if not is_action_question and "how" not in query.lower():
        return 0.0

    text_lower = text.lower()
    cues = [
        "press ",
        "select ",
        "highlight ",
        "type ",
        "use ",
        "jog ",
        "step",
        "1.",
        "2.",
        "3.",
        "a.",
        "b.",
    ]
    cue_hits = sum(1 for cue in cues if cue in text_lower)
    score = min(1.0, cue_hits / 4)

    if "list of settings" in text_lower:
        score -= 0.35

    return max(0.0, score)


def citation_ref(index: int) -> str:
    return f"S{index}"


def build_index(
    chunks_path: Path,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
) -> int:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError(f"No chunks found in {chunks_path}")

    embeddings = embed_texts(model_name, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump({"embedding_model": model_name, "chunks": chunks}, handle, indent=2)

    return len(chunks)


def search(
    query: str,
    index_path: Path,
    metadata_path: Path,
    model_name: str,
    top_k: int = 5,
    candidate_k: int = 30,
) -> list[dict]:
    index = faiss.read_index(str(index_path))
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    model = load_embedding_model(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)
    candidate_k = max(top_k, min(candidate_k, index.ntotal))
    scores, indices = index.search(np.asarray(query_embedding, dtype="float32"), candidate_k)

    candidates = []
    for vector_score, index_id in zip(scores[0], indices[0]):
        if index_id < 0:
            continue
        chunk = metadata["chunks"][int(index_id)]
        keyword_score = lexical_score(query, chunk["text"])
        instruction_score = procedural_score(query, chunk["text"])
        hybrid_score = (
            (float(vector_score) * 0.65)
            + (keyword_score * 0.2)
            + (instruction_score * 0.15)
        )
        candidates.append(
            {
                **chunk,
                "score": hybrid_score,
                "vector_score": float(vector_score),
                "keyword_score": keyword_score,
                "procedural_score": instruction_score,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)

    selected: list[dict] = []
    page_counts: dict[tuple[str, int], int] = {}
    for candidate in candidates:
        page_key = (candidate["source_path"], int(candidate["page"]))
        if page_counts.get(page_key, 0) >= 1 and len(selected) < max(2, top_k // 2):
            continue
        selected.append(candidate)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if len(selected) == top_k:
            break

    if len(selected) < top_k:
        selected_ids = {result["id"] for result in selected}
        selected.extend(
            candidate for candidate in candidates if candidate["id"] not in selected_ids
        )

    results = selected[:top_k]
    for index_id, result in enumerate(results, start=1):
        result["citation"] = citation_ref(index_id)

    return results
