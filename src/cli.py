from __future__ import annotations

import argparse
from pathlib import Path

from requests.exceptions import RequestException

from cnc_rag.config import (
    CHUNKS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    EVAL_QUESTIONS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RAW_MANUALS_DIR,
)
from cnc_rag.evaluation import evaluate, summarize
from cnc_rag.generation import answer_with_ollama
from cnc_rag.generation.ollama import format_source_list
from cnc_rag.ingestion import ingest_manuals
from cnc_rag.retrieval import build_index, search


def ingest_command(args: argparse.Namespace) -> None:
    chunks = ingest_manuals(RAW_MANUALS_DIR, CHUNKS_PATH)
    count = build_index(CHUNKS_PATH, FAISS_INDEX_PATH, METADATA_PATH, args.embedding_model)
    print(f"Ingested {len(chunks)} chunks and indexed {count} vectors.")


def search_command(args: argparse.Namespace) -> None:
    results = search(
        args.query,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        args.embedding_model,
        top_k=args.top_k,
    )
    for index, result in enumerate(results, start=1):
        print(f"\n[{result['citation']}] score={result['score']:.3f}")
        print(
            f"vector={result['vector_score']:.3f} keyword={result['keyword_score']:.3f} "
            f"procedure={result['procedural_score']:.3f}"
        )
        print(f"{result['document_title']} page {result['page']}")
        print(result["text"][:700])


def ask_command(args: argparse.Namespace) -> None:
    contexts = search(
        args.query,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        args.embedding_model,
        top_k=args.top_k,
    )
    try:
        answer = answer_with_ollama(args.query, contexts, args.ollama_model)
        print(answer)
    except RequestException as exc:
        print(f"Could not reach Ollama for answer generation: {exc}")
        print("Start Ollama with `ollama serve`, or use `cnc-rag search` for retrieval only.")
    print("\nSources:")
    print(format_source_list(contexts))


def chat_command(args: argparse.Namespace) -> None:
    print("Offline CNC assistant. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nQuestion> ").strip()
        except EOFError:
            print()
            return

        if query.lower() in {"exit", "quit"}:
            return
        if not query:
            continue

        contexts = search(
            query,
            FAISS_INDEX_PATH,
            METADATA_PATH,
            args.embedding_model,
            top_k=args.top_k,
        )
        try:
            answer = answer_with_ollama(query, contexts, args.ollama_model)
            print(f"\n{answer}")
        except RequestException as exc:
            print(f"\nCould not reach Ollama for answer generation: {exc}")
            print("Start Ollama with `ollama serve`, or use `cnc-rag search` for retrieval only.")
        print("\nSources:")
        print(format_source_list(contexts))


def eval_command(args: argparse.Namespace) -> None:
    results = evaluate(
        args.eval_file,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        args.embedding_model,
        top_k=args.top_k,
    )
    summary = summarize(results)

    print(f"Cases: {summary['cases']}")
    print(f"Page hit rate@{args.top_k}: {summary['page_hit_rate']:.0%}")
    print(f"Mean keyword recall@{args.top_k}: {summary['mean_term_recall']:.0%}")

    for result in results:
        status = "PASS" if result.page_hit else "MISS"
        print(f"\n[{status}] {result.case.id}")
        print(f"Question: {result.case.question}")
        print(f"Expected pages: {result.case.expected_pages}")
        print(f"Top pages: {result.top_pages}")
        print(f"Keyword hits: {result.term_hits}")
        print(f"Top score: {result.top_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CNC RAG assistant")
    subparsers = parser.add_subparsers(required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Extract manual text and build index")
    ingest_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ingest_parser.set_defaults(func=ingest_command)

    search_parser = subparsers.add_parser("search", help="Retrieve relevant manual chunks")
    search_parser.add_argument("query")
    search_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.set_defaults(func=search_command)

    ask_parser = subparsers.add_parser("ask", help="Retrieve chunks and ask local Ollama")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask_parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.set_defaults(func=ask_command)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive local chatbot")
    chat_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    chat_parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.set_defaults(func=chat_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate retrieval against sample questions")
    eval_parser.add_argument("--eval-file", type=Path, default=EVAL_QUESTIONS_PATH)
    eval_parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.set_defaults(func=eval_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
