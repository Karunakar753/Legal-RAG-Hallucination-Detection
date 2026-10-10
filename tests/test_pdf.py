import pytest

from utils.pdf_processor import (
    InvalidPDFError,
    PDFProcessingError,
    ScannedPDFError,
    extract_pages,
    get_pdf_stats,
)


def test_extracts_text_with_page_numbers(sample_pdf_bytes):
    pages = extract_pages(sample_pdf_bytes)
    assert [p.page for p in pages] == [1, 2, 3]
    assert "Section 302" in pages[0].text
    assert "eyewitnesses" in pages[1].text
    assert "appeal is dismissed" in pages[2].text


def test_accepts_file_path(sample_pdf_bytes, tmp_path):
    path = tmp_path / "judgment.pdf"
    path.write_bytes(sample_pdf_bytes)
    assert len(extract_pages(path)) == 3


def test_stats(sample_pdf_bytes):
    stats = get_pdf_stats(extract_pages(sample_pdf_bytes))
    assert stats["pages"] == 3 and stats["pages_with_text"] == 3 and stats["characters"] > 100


@pytest.mark.parametrize("bad", [b"", b"this is definitely not a pdf", b"%PDF-1.7 garbage"])
def test_invalid_pdf_raises_friendly_error(bad):
    with pytest.raises(InvalidPDFError):
        extract_pages(bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(InvalidPDFError):
        extract_pages(tmp_path / "nope.pdf")


def test_scanned_pdf_raises(blank_pdf_bytes):
    with pytest.raises(ScannedPDFError):
        extract_pages(blank_pdf_bytes)


def test_errors_share_base_class(blank_pdf_bytes):
    with pytest.raises(PDFProcessingError):
        extract_pages(blank_pdf_bytes)
