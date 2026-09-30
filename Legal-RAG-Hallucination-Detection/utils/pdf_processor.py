"""PDF text extraction with page numbers (PyMuPDF). Everything is done in memory."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pymupdf  # NOT the deprecated `fitz` alias

MIN_TOTAL_CHARS = 50  # fewer extractable characters than this => scanned / image-only PDF


class PDFProcessingError(Exception):
    """Base class for user-facing PDF problems."""


class InvalidPDFError(PDFProcessingError):
    pass


class EncryptedPDFError(PDFProcessingError):
    pass


class EmptyPDFError(PDFProcessingError):
    pass


class ScannedPDFError(PDFProcessingError):
    pass


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based, as printed in a PDF viewer
    text: str


def _clean_block(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\s*\n(?=[a-z])", "", text)  # re-join words hyphenated at line ends
    text = re.sub(r"\s*\n\s*", " ", text)  # soft line wraps -> spaces
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def _open(source: Union[bytes, bytearray, str, Path]):
    try:
        if isinstance(source, (bytes, bytearray)):
            if not source:
                raise InvalidPDFError("The uploaded file is empty.")
            doc = pymupdf.open(stream=bytes(source), filetype="pdf")
        else:
            path = Path(source)
            if not path.is_file():
                raise InvalidPDFError(f"File not found: {path}")
            doc = pymupdf.open(str(path))
    except PDFProcessingError:
        raise
    except Exception as exc:  # pymupdf raises several distinct error types
        raise InvalidPDFError("This file is not a valid or readable PDF.") from exc
    if not doc.is_pdf:
        doc.close()
        raise InvalidPDFError("This file is not a PDF document.")
    return doc


def extract_pages(source: Union[bytes, bytearray, str, Path]) -> list[PageText]:
    """Return one PageText per PDF page (blank pages included, page numbers preserved)."""
    doc = _open(source)
    try:
        if doc.needs_pass:
            raise EncryptedPDFError("This PDF is password-protected. Remove the password and retry.")
        if doc.page_count == 0:
            raise EmptyPDFError("This PDF has no pages.")
        pages: list[PageText] = []
        for number, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks", sort=True)
            parts = [_clean_block(b[4]) for b in blocks if b[6] == 0]  # b[6]==0 -> text block
            pages.append(PageText(number, "\n\n".join(p for p in parts if p)))
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise InvalidPDFError("The PDF could not be read (it may be corrupted).") from exc
    finally:
        doc.close()

    if sum(len(p.text) for p in pages) < MIN_TOTAL_CHARS:
        raise ScannedPDFError(
            "No extractable text found. This looks like a scanned/image-only PDF - "
            "run OCR on it first (e.g. OCRmyPDF) and upload the searchable version."
        )
    return pages


def get_pdf_stats(pages: list[PageText]) -> dict:
    return {
        "pages": len(pages),
        "pages_with_text": sum(1 for p in pages if p.text.strip()),
        "characters": sum(len(p.text) for p in pages),
    }
