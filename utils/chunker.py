"""Page-aware chunking: chunks never span pages, so every chunk has an exact page number."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.pdf_processor import PageText

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(int(data["chunk_id"]), int(data["page"]), str(data["text"]))


def chunk_pages(
    pages: Sequence[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.lstrip(".;, ").strip()  # drop separator residue kept by the splitter
            if piece:
                chunks.append(Chunk(len(chunks), page.page, piece))
    return chunks
