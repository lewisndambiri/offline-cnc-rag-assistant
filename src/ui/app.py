from __future__ import annotations

import time

import requests
import streamlit as st

from cnc_rag.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from cnc_rag.generation import stream_answer_with_ollama
from cnc_rag.generation.ollama import ensure_cited_lines
from cnc_rag.retrieval import search


SAMPLE_QUESTIONS = [
    "What safety precautions should I follow before operating the machine?",
    "How do I use MDI mode?",
    "How do I set a tool offset?",
    "How do I select or edit the active program?",
    "Where are G codes and M codes listed?",
]


def configure_page() -> None:
    st.set_page_config(
        page_title="CNC RAG Assistant",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
          --cnc-border: #d8dee6;
          --cnc-muted: #5f6b7a;
          --cnc-panel: #f7f9fc;
        }
        .block-container {
          padding-top: 1.5rem;
          padding-bottom: 1.5rem;
          max-width: 1440px;
        }
        [data-testid="stSidebar"] {
          border-right: 1px solid var(--cnc-border);
        }
        [data-testid="stMetricValue"] {
          font-size: 1.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def retrieve(question: str, top_k: int, embedding_model: str) -> list[dict]:
    return search(
        question,
        FAISS_INDEX_PATH,
        METADATA_PATH,
        embedding_model,
        top_k=top_k,
    )


def render_source(source: dict) -> None:
    citation = source.get("citation", "S?")
    with st.container(border=True):
        st.markdown(f"**[{citation}] {source['document_title']}**")
        st.caption(f"Page {source['page']} | chunk `{source['id']}`")

        score_cols = st.columns(4)
        score_cols[0].metric("Hybrid", f"{source['score']:.3f}")
        score_cols[1].metric("Vector", f"{source['vector_score']:.3f}")
        score_cols[2].metric("Keyword", f"{source['keyword_score']:.3f}")
        score_cols[3].metric("Procedure", f"{source['procedural_score']:.3f}")

        st.text_area(
            "Excerpt",
            value=source["text"][:1400],
            height=220,
            disabled=True,
            label_visibility="collapsed",
            key=f"source-{source['id']}",
        )


def render_sidebar() -> tuple[int, str, str, bool, int]:
    st.sidebar.title("Controls")
    top_k = st.sidebar.slider("Sources", min_value=2, max_value=8, value=3)
    generate_answer = st.sidebar.toggle("Generate answer", value=True)
    num_predict = st.sidebar.slider(
        "Answer length",
        min_value=180,
        max_value=900,
        value=450,
        step=90,
    )
    ollama_model = st.sidebar.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
    embedding_model = st.sidebar.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)

    st.sidebar.divider()
    st.sidebar.caption("Sample questions")
    for question in SAMPLE_QUESTIONS:
        if st.sidebar.button(question, use_container_width=True):
            st.session_state.pending_question = question

    return top_k, embedding_model, ollama_model, generate_answer, num_predict


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def stream_ui_answer(
    question: str,
    sources: list[dict],
    ollama_model: str,
    num_predict: int,
) -> str:
    placeholder = st.empty()
    status = st.caption("Streaming local answer from Ollama...")
    answer_parts = []

    for token in stream_answer_with_ollama(
        question,
        sources,
        ollama_model,
        num_predict=num_predict,
    ):
        answer_parts.append(token)
        placeholder.markdown("".join(answer_parts))

    answer = ensure_cited_lines("".join(answer_parts).strip(), sources)
    placeholder.markdown(answer)
    status.empty()
    return answer


def main() -> None:
    configure_page()
    top_k, embedding_model, ollama_model, generate_answer, num_predict = render_sidebar()

    st.title("CNC RAG Assistant")
    st.caption("Haas Mill NGC manual | local FAISS retrieval | Ollama/Mistral generation")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Corpus", "839 chunks")
    metric_cols[1].metric("Retrieval", "Hybrid")
    metric_cols[2].metric("Mode", "Offline")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []

    render_history()

    prompt = st.chat_input("Ask a CNC operation or maintenance question")
    if not prompt and st.session_state.get("pending_question"):
        prompt = st.session_state.pop("pending_question")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        started_at = time.perf_counter()
        with st.status("Retrieving manual evidence...", expanded=False) as status:
            sources = retrieve(prompt, top_k, embedding_model)
            status.update(
                label=f"Retrieved {len(sources)} source chunks",
                state="complete",
                expanded=False,
            )
        st.session_state.last_sources = sources

        with st.chat_message("assistant"):
            if generate_answer:
                try:
                    answer = stream_ui_answer(prompt, sources, ollama_model, num_predict)
                except requests.RequestException as exc:
                    st.warning(str(exc))
                    answer = "Ollama generation is unavailable. Review the ranked sources below."
                    st.markdown(answer)
            else:
                answer = "Retrieval-only mode is active. Review the ranked sources below."
                st.markdown(answer)
            st.caption(f"Completed in {time.perf_counter() - started_at:.1f}s")

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.last_sources:
        st.divider()
        left, right = st.columns([0.72, 0.28])
        with left:
            st.subheader("Retrieved Sources")
            for source in st.session_state.last_sources:
                render_source(source)
        with right:
            st.subheader("Source Scores")
            for source in st.session_state.last_sources:
                st.metric(
                    f"[{source.get('citation', 'S?')}] Page {source['page']}",
                    f"{source['score']:.3f}",
                    help="Hybrid score combines semantic, keyword, and procedural signals.",
                )


if __name__ == "__main__":
    main()
