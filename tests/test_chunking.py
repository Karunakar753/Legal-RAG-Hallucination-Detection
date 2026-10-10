import pytest

from utils.chunker import Chunk, chunk_pages
from utils.pdf_processor import PageText, extract_pages


def test_every_chunk_keeps_its_page_number(sample_pdf_bytes):
    pages = extract_pages(sample_pdf_bytes)
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > len(pages)
    by_page = {p.page: p.text for p in pages}
    for chunk in chunks:
        assert chunk.page in by_page
        assert chunk.text in by_page[chunk.page]  # chunk really came from that page
    assert {c.page for c in chunks} == {1, 2, 3}


def test_chunk_ids_are_sequential_and_size_bounded():
    pages = [PageText(1, "word " * 500), PageText(2, "other " * 300)]
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=50)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    assert all(len(c.text) <= 300 for c in chunks)


def test_blank_pages_are_skipped():
    chunks = chunk_pages([PageText(1, "   "), PageText(2, "Real content here.")])
    assert len(chunks) == 1 and chunks[0].page == 2


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages([PageText(1, "text")], size, overlap)


def test_chunk_roundtrip():
    c = Chunk(3, 7, "hello")
    assert Chunk.from_dict(c.to_dict()) == c
