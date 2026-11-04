"""Document loading and structure-aware chunking for Chinese/English text."""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;])|(?<=\.)\s+|\n+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack sentences into chunks of ~size chars with sentence overlap."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sent in sentences:
        if current and length + len(sent) > size:
            chunks.append("".join(current))
            # carry trailing sentences as overlap
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev)
            current, length = carry, carry_len
        current.append(sent)
        length += len(sent)
    if current:
        chunks.append("".join(current))
    return chunks


def chunk_text(text: str, source: str, size: int = 360, overlap: int = 60) -> list[Document]:
    """Split by markdown headings first, then pack sentences within each section.

    Each chunk keeps its document title and section heading in metadata and is
    prefixed with the section path, which noticeably helps both BM25 and the
    embedding model on short policy clauses.
    """
    title = ""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in text.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            level, heading = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = heading
                continue
            sections.append((heading, []))
        elif line.strip() and not line.lstrip().startswith(">"):
            sections[-1][1].append(line.strip())

    docs: list[Document] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        for piece in _pack(split_sentences(body), size, overlap):
            prefix = " / ".join(p for p in (title, heading) if p)
            content = f"【{prefix}】{piece}" if prefix else piece
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "title": title or source,
                        "section": heading,
                        "chunk_id": f"{source}#{len(docs)}",
                    },
                )
            )
    return docs


def load_corpus(docs_dir: Path, size: int = 360, overlap: int = 60) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            docs.extend(chunk_text(read_text(path), path.name, size, overlap))
    return docs
