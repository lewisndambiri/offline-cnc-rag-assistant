from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Chunk:
    id: str
    source_path: str
    document_title: str
    page: int
    text: str


def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> list[str]:
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)
        current = paragraph

        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars - overlap :].strip()

    if current:
        chunks.append(current)

    return chunks


def ingest_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(str(path))
    title = reader.metadata.title if reader.metadata and reader.metadata.title else path.stem
    chunks: list[Chunk] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        for chunk_index, chunk in enumerate(chunk_text(text), start=1):
            chunks.append(
                Chunk(
                    id=f"{path.stem}:p{page_index}:c{chunk_index}",
                    source_path=str(path),
                    document_title=title,
                    page=page_index,
                    text=chunk,
                )
            )

    return chunks


def ingest_manuals(input_dir: Path, output_path: Path) -> list[Chunk]:
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    chunks: list[Chunk] = []

    for pdf_path in pdf_paths:
        chunks.extend(ingest_pdf(pdf_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    return chunks

