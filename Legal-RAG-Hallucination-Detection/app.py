"""Legal RAG + Hallucination Detection - Streamlit app.

Run:  python -m streamlit run app.py
Uploaded PDFs are processed in memory only and are never written to disk.
"""
from __future__ import annotations

import hashlib

import streamlit as st

from utils.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from utils.embeddings import embed_texts, load_embedding_model
from utils.hallucination import check_answer
from utils.llm import LLMError, MissingAPIKeyError, generate_answer, get_model_name, resolve_api_key
from utils.pdf_processor import PDFProcessingError, extract_pages, get_pdf_stats
from utils.prompts import DISCLAIMER, INSUFFICIENT_EVIDENCE_MESSAGE
from utils.retriever import MAX_TOP_K, MIN_TOP_K, Retriever
from utils.vector_store import VectorStore


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def get_embedding_model():
    return load_embedding_model()


def build_index(pdf_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int) -> dict:
    """PDF bytes -> pages -> chunks -> embeddings -> FAISS. Nothing is saved to disk."""
    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    if not chunks:
        raise PDFProcessingError("No text chunks could be created from this PDF.")
    embeddings = embed_texts(get_embedding_model(), [c.text for c in chunks])
    return {
        "key": (hashlib.sha256(pdf_bytes).hexdigest(), chunk_size, chunk_overlap),
        "filename": filename,
        "stats": {**get_pdf_stats(pages), "chunks": len(chunks)},
        "store": VectorStore.from_chunks(embeddings, chunks),
    }


def run_query(doc: dict, question: str, top_k: int, api_key: str | None, model_name: str | None) -> dict:
    """Retrieve -> generate (evidence only) -> hallucination check."""
    model = get_embedding_model()
    retriever = Retriever(model, doc["store"])
    evidence = retriever.retrieve(question, top_k)
    result = {"question": question, "evidence": evidence, "answer": None,
              "report": None, "error": None, "notice": None}

    if not retriever.is_sufficient(evidence):
        best = max((e.score for e in evidence), default=0.0)
        result["answer"] = INSUFFICIENT_EVIDENCE_MESSAGE
        result["notice"] = (f"No passage was relevant enough (best similarity {best:.2f}); "
                            "the LLM was not called, to avoid an ungrounded answer.")
        return result
    try:
        answer = generate_answer(question, evidence, api_key=api_key, model=model_name)
    except (MissingAPIKeyError, LLMError) as exc:
        result["error"] = str(exc)
        return result
    result["answer"] = answer
    result["report"] = check_answer(answer, evidence, lambda texts: embed_texts(model, texts))
    return result


def render_doc_info(doc: dict) -> None:
    s = doc["stats"]
    st.subheader("Document")
    cols = st.columns(5)
    cols[0].metric("File", doc["filename"][:22] + ("..." if len(doc["filename"]) > 22 else ""))
    cols[1].metric("Pages", s["pages"])
    cols[2].metric("Pages with text", s["pages_with_text"])
    cols[3].metric("Chunks", s["chunks"])
    cols[4].metric("Characters", f"{s['characters']:,}")
    if s["pages_with_text"] < s["pages"]:
        st.caption(f"{s['pages'] - s['pages_with_text']} page(s) had no extractable text (blank or scanned).")


def render_result(result: dict) -> None:
    evidence, report = result["evidence"], result["report"]
    if result["notice"]:
        st.info(result["notice"])
    if result["error"]:
        st.error(result["error"])
        st.caption("Retrieved evidence is still shown below.")
    if result["answer"]:
        st.subheader("Answer")
        st.markdown(result["answer"].replace("$", "\\$"))

    retrieved_pages = sorted({e.page for e in evidence})
    if retrieved_pages:
        st.markdown("**Pages retrieved:** " + ", ".join(map(str, retrieved_pages)))

    if report is not None:
        st.subheader("Support / hallucination check")
        if report.cited_pages:
            st.markdown("**Pages cited in answer:** " + ", ".join(map(str, report.cited_pages)))
        if report.invalid_citations:
            st.error("Cited page(s) not in retrieved evidence: " + ", ".join(map(str, report.invalid_citations)))
        if report.overall_score is None:
            st.info(report.note)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Support score", f"{report.overall_score:.0%}")
            c2.metric("Claims checked", len(report.claims))
            c3.metric("Potentially unsupported", len(report.unsupported_claims))
            st.progress(min(max(report.overall_score, 0.0), 1.0))
            st.caption(f"{report.verdict}. {report.note}")

            st.subheader("Potentially unsupported claims")
            if report.unsupported_claims:
                for c in report.unsupported_claims:
                    with st.expander(c.claim[:110] + ("..." if len(c.claim) > 110 else ""), expanded=True):
                        st.markdown(f"**Claim:** {c.claim}".replace("$", "\\$"))
                        for reason in c.reasons:
                            st.markdown(f"- {reason}")
                        st.caption(f"Support {c.score:.0%} | closest evidence: page {c.best_page}")
            else:
                st.success("No potentially unsupported claims detected. Still verify against the evidence below.")
            with st.expander("Claim-by-claim details"):
                st.dataframe(
                    [{"claim": c.claim, "support": round(c.score, 2), "semantic": round(c.semantic_score, 2),
                      "lexical": round(c.lexical_score, 2), "closest page": c.best_page,
                      "flagged": not c.supported} for c in report.claims],
                )

    st.subheader("Retrieved evidence")
    for e in evidence:
        with st.expander(f"Source {e.rank} | Page {e.page} | similarity {e.score:.2f}", expanded=e.rank == 1):
            st.text(e.text)


def main() -> None:
    st.set_page_config(page_title="Legal RAG - Hallucination Detection", page_icon="⚖️", layout="wide")
    st.title("⚖️ Legal RAG with Hallucination Detection")
    st.warning(DISCLAIMER)

    with st.sidebar:
        st.header("Settings")
        typed_key = st.text_input("API key (OpenAI or Groq)", type="password",
                                  help="Used for this session only. Otherwise read from .env or Streamlit Secrets.")
        active_key = resolve_api_key(typed_key)
        model_name = st.text_input("LLM Model", value=get_model_name(typed_key))
        chunk_size = st.slider("Chunk size (characters)", 500, 2000, DEFAULT_CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 400, DEFAULT_CHUNK_OVERLAP, 50)
        if active_key:
            provider = "Groq" if active_key.startswith("gsk_") else "OpenAI"
            st.success(f"{provider} API key configured.")
        else:
            st.error("No API key configured - retrieval works, answer generation will not.")
        st.caption("PDFs are processed in memory and never stored permanently.")

    uploaded = st.file_uploader("Upload a legal judgment (PDF)", type=["pdf"])
    if uploaded is None:
        st.session_state.pop("doc", None)
        st.session_state.pop("result", None)
        st.info("Upload a text-based judgment PDF to begin.")
        return

    data = uploaded.getvalue()
    key = (hashlib.sha256(data).hexdigest(), chunk_size, min(chunk_overlap, chunk_size - 1))
    doc = st.session_state.get("doc")
    if doc is None or doc["key"] != key:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Extracting text, chunking and embedding..."):
                doc = build_index(data, uploaded.name, key[1], key[2])
        except PDFProcessingError as exc:
            st.session_state.pop("doc", None)
            st.error(str(exc))
            return
        except Exception as exc:  # e.g. embedding model download failure
            st.session_state.pop("doc", None)
            st.error(f"Could not process the document: {exc}")
            return
        st.session_state["doc"] = doc

    render_doc_info(doc)
    st.divider()
    question = st.text_area("Ask a question about the judgment", height=90,
                            placeholder="e.g. What did the court decide, and on what grounds?")
    top_k = st.slider("Top-K evidence passages", MIN_TOP_K, MAX_TOP_K, 5)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                st.session_state["result"] = run_query(doc, question.strip(), top_k, typed_key, model_name)
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Something went wrong: {exc}")
    if st.session_state.get("result"):
        render_result(st.session_state["result"])


if __name__ == "__main__":
    main()
