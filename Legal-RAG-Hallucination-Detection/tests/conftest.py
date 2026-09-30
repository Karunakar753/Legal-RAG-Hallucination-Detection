import re
import zlib

import numpy as np
import pymupdf
import pytest

PAGE_TEXTS = [
    "IN THE SUPREME COURT OF INDIA. State of Kerala v. Ramesh Kumar. Criminal Appeal No. 1234 of 2019. "
    "The appellant challenges the conviction recorded by the High Court under Section 302 of the Indian Penal Code. "
    "The trial court had sentenced the accused to life imprisonment on 12 March 2018.",
    "The Court examined the testimony of the eyewitnesses and found the prosecution evidence reliable. "
    "The medical report confirmed that the deceased died of multiple stab wounds. "
    "The defence argued that the recovery of the weapon was doubtful, but the Court rejected this contention.",
    "For these reasons the appeal is dismissed and the conviction under Section 302 is upheld. "
    "The sentence of life imprisonment is confirmed. No order as to costs is made. Pronounced on 5 August 2020.",
]


class FakeModel:
    """Deterministic bag-of-words hashing embedder - mimics SentenceTransformer.encode (no downloads)."""

    DIM = 512

    def get_sentence_embedding_dimension(self):
        return self.DIM

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[i, zlib.crc32(word.encode()) % self.DIM] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9) if normalize_embeddings else out


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def sample_pdf_bytes():
    return make_pdf(PAGE_TEXTS)


@pytest.fixture
def blank_pdf_bytes():
    return make_pdf(["", ""])  # no text layer, like a scan
