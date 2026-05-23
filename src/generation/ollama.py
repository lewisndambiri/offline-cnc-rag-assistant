from __future__ import annotations

import json
import re
from collections.abc import Iterator

import requests


CITATION_PATTERN = re.compile(r"\[S\d+\]")
TRAILING_CITATION_PATTERN = re.compile(r"\[S\d+\]\.?$")


def ensure_cited_lines(answer: str, contexts: list[dict]) -> str:
    default_citation = f"[{contexts[0].get('citation', 'S1')}]" if contexts else "[S1]"
    cited_lines = []

    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("sources:"):
            cited_lines.append(line)
            continue

        citations = CITATION_PATTERN.findall(line)
        if not citations:
            cited_lines.append(f"{line} {default_citation}")
            continue

        if not TRAILING_CITATION_PATTERN.search(stripped):
            cited_lines.append(f"{line} {citations[-1]}")
            continue

        cited_lines.append(line)

    return "\n".join(cited_lines)


def format_source_list(contexts: list[dict]) -> str:
    lines = []
    for index, context in enumerate(contexts, start=1):
        citation = context.get("citation", f"S{index}")
        lines.append(
            f"[{citation}] {context['document_title']}, page {context['page']} "
            f"(score {context['score']:.3f})"
        )
    return "\n".join(lines)


def build_prompt(question: str, contexts: list[dict]) -> str:
    source_blocks = []
    for index, context in enumerate(contexts, start=1):
        citation = context.get("citation", f"S{index}")
        source_blocks.append(
            f"[{citation}: {context['document_title']}, page {context['page']}]\n"
            f"{context['text']}"
        )

    sources = "\n\n".join(source_blocks)
    return f"""You are an offline CNC documentation assistant.
Answer only from the provided sources.
If the sources do not contain enough information, say that clearly.
Every factual sentence or numbered step must end with one or more citation markers,
such as [S1] or [S2].
Do not put all citations only at the end of the answer.
Do not cite sources that do not support the sentence.
Do not provide unsafe machine-operation instructions beyond what the sources support.
Keep the answer practical and concise.

Use this style:
1. Press the relevant control key. [S1]
2. Save the program only if the cited source describes that action. [S2]

Question:
{question}

Sources:
{sources}
"""


def answer_with_ollama(
    question: str,
    contexts: list[dict],
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    num_predict: int = 450,
) -> str:
    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": build_prompt(question, contexts),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=120,
    )
    response.raise_for_status()
    return ensure_cited_lines(response.json()["response"].strip(), contexts)


def stream_answer_with_ollama(
    question: str,
    contexts: list[dict],
    model: str,
    base_url: str = "http://localhost:11434",
    temperature: float = 0.1,
    num_predict: int = 450,
) -> Iterator[str]:
    with requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": build_prompt(question, contexts),
            "stream": True,
            "options": {"temperature": temperature, "num_predict": num_predict},
        },
        timeout=120,
        stream=True,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            payload = json.loads(line)
            token = payload.get("response", "")
            if token:
                yield token
            if payload.get("done"):
                break
