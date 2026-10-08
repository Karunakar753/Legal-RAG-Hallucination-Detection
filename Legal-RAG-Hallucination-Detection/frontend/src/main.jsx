import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import ReactMarkdown from "react-markdown";
import "./styles.css";

const API_URL =
  import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, "") ||
  (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

function getApiUrl(path) {
  if (!API_URL) {
    throw new Error(
      "The analysis service is not configured. Connect a secured backend by setting VITE_API_URL.",
    );
  }
  return `${API_URL}${path}`;
}

const STARTER_QUESTIONS = [
  "Summarize the facts, issues, applicable provisions, arguments, reasoning, and final order.",
  "Which Acts or legal provisions does the judgment expressly mention, and how does the court apply them?",
  "What were the court's reasons for its decision, and which pages support each reason?",
  "What was the dispute before the court, and how did the court resolve it?",
  "Which facts in the judgment support the final conclusion?",
];

async function parseResponse(response) {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "The request could not be completed.");
  }
  return body;
}

function App() {
  const [file, setFile] = useState(null);
  const [documentInfo, setDocumentInfo] = useState(null);
  const [question, setQuestion] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const apiConfigured = Boolean(API_URL);
  const showSuggestions = Boolean(documentInfo) && !result && !error;

  async function uploadJudgment(selectedFile) {
    if (!selectedFile) return;
    setError("");
    setResult(null);
    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", selectedFile);
      const response = await fetch(getApiUrl("/api/documents"), {
        method: "POST",
        body: form,
      });
      const data = await parseResponse(response);
      setDocumentInfo(data);
      setFile(selectedFile);
    } catch (requestError) {
      setDocumentInfo(null);
      setFile(null);
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  async function removeJudgment() {
    if (documentInfo) {
      try {
        await fetch(
          getApiUrl(`/api/documents/${documentInfo.document_id}`),
          { method: "DELETE" },
        );
      } catch {
        // The server clears in-memory documents on restart or cache eviction.
      }
    }
    setDocumentInfo(null);
    setFile(null);
    setResult(null);
    setQuestion("");
    setError("");
  }

  async function askQuestion(event) {
    event.preventDefault();
    if (!documentInfo || !question.trim()) return;
    setError("");
    setResult(null);
    setBusy(true);
    try {
      const response = await fetch(getApiUrl("/api/query"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: documentInfo.document_id,
          question: question.trim(),
          top_k: 5,
          api_key: apiKey.trim() || null,
          model: model.trim() || null,
        }),
      });
      setResult(await parseResponse(response));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  }

  function acceptFile(candidate) {
    if (!candidate || !apiConfigured) return;
    if (!candidate.name.toLowerCase().endsWith(".pdf")) {
      setError("Choose a PDF judgment file.");
      return;
    }
    setError("");
    setFile(candidate);
    uploadJudgment(candidate);
  }

  const report = result?.report;
  const stats = documentInfo?.stats;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Casebrief home">
          <span className="brand-mark" aria-hidden="true">
            C
          </span>
          <span>
            casebrief<span className="brand-dot">.</span>
          </span>
        </a>
        <div className="topbar-note">
          <span className="status-dot" /> Evidence-first legal analysis
        </div>
      </header>

      <main className="page">
        <section className="intro">
          <div className="eyebrow">
            <span className="eyebrow-line" /> LEGAL DOCUMENT WORKSPACE
          </div>
          <h1>
            Understand the judgment.
            <br />
            <span>Stay grounded in the record.</span>
          </h1>
          <p>
            Upload a PDF, ask a question, and get an answer tied to the pages
            the court actually relied on. The tool helps you check facts,
            reasoning, and legal provisions without leaving the source document.
          </p>
        </section>

        {!apiConfigured && (
          <p className="notice-banner" role="status">
            The analysis service is not connected to this deployment. PDF
            uploads and questions will be enabled after a secured backend URL is
            configured.
          </p>
        )}

        <section className="workspace">
          <aside className="document-column">
            <div className="section-heading">
              <div>
                <span className="step-label">01 / SOURCE</span>
                <h2>Your judgment</h2>
              </div>
              {documentInfo && (
                <button
                  className="text-button"
                  type="button"
                  onClick={removeJudgment}
                >
                  Remove
                </button>
              )}
            </div>

            {!documentInfo ? (
              <label
                className={`upload-card ${dragging ? "is-dragging" : ""} ${busy ? "is-busy" : ""} ${!apiConfigured ? "is-disabled" : ""}`}
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  acceptFile(event.dataTransfer.files[0]);
                }}
              >
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(event) => acceptFile(event.target.files[0])}
                  disabled={!apiConfigured || busy}
                />
                <span className="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none">
                    <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14.5v4A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-4" />
                  </svg>
                </span>
                <strong>
                  {busy
                    ? "Reading your judgment…"
                    : apiConfigured
                      ? "Drop your PDF here"
                      : "Analysis service unavailable"}
                </strong>
                <span className="upload-subtitle">
                  {busy
                    ? "Extracting text and building page references"
                    : apiConfigured
                      ? "or click to browse · up to 25 MB"
                      : "A secured backend has not been connected"}
                </span>
                <span className="upload-type">PDF ONLY</span>
              </label>
            ) : (
              <div className="document-card">
                <div className="document-file-icon">PDF</div>
                <div className="document-file-info">
                  <strong title={file?.name}>{documentInfo.filename}</strong>
                  <span>
                    {stats.pages} pages <i /> {stats.chunks} evidence passages
                  </span>
                </div>
                <span className="ready-check" aria-label="Ready">
                  ✓
                </span>
                <div className="document-stats">
                  <div>
                    <strong>{stats.pages}</strong>
                    <span>Pages</span>
                  </div>
                  <div>
                    <strong>{stats.pages_with_text}</strong>
                    <span>With text</span>
                  </div>
                  <div>
                    <strong>{stats.chunks}</strong>
                    <span>Passages</span>
                  </div>
                </div>
                {stats.pages_with_text < stats.pages && (
                  <p className="scan-note">
                    {stats.pages - stats.pages_with_text} page(s) have no
                    extractable text and may be scanned.
                  </p>
                )}
              </div>
            )}

            <div className="privacy-note">
              <span className="lock-icon" aria-hidden="true">
                ⌑
              </span>
              <div>
                <strong>Processed in memory</strong>
                <span>
                  Your PDF is not saved to disk. It is removed when you remove
                  it or the server restarts.
                </span>
              </div>
            </div>

            <details className="settings">
              <summary>
                <span>Answer settings</span>
                <span className="settings-summary">
                  API key &amp; model <span aria-hidden="true">＋</span>
                </span>
              </summary>
              <div className="settings-fields">
                <label>
                  API key <span className="optional">OPTIONAL</span>
                  <input
                    type="password"
                    value={apiKey}
                    onChange={(event) => setApiKey(event.target.value)}
                    placeholder="OpenAI or Groq key"
                    autoComplete="off"
                  />
                  <small>
                    Used for this session only; or configure the server
                    environment.
                  </small>
                </label>
                <label>
                  Model <span className="optional">OPTIONAL</span>
                  <input
                    value={model}
                    onChange={(event) => setModel(event.target.value)}
                    placeholder="Use the server default"
                  />
                </label>
              </div>
            </details>
          </aside>

          <section className="analysis-column">
            <div className="section-heading">
              <div>
                <span className="step-label">02 / ANALYSIS</span>
                <h2>Ask the record</h2>
              </div>
              <span className="grounded-pill">✳ EVIDENCE-BASED</span>
            </div>

            <form className="question-form" onSubmit={askQuestion}>
              <label className="visually-hidden" htmlFor="question">
                Ask a question about this judgment
              </label>
              <textarea
                id="question"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="What did the court decide, and what reasons and provisions did it rely on?"
                disabled={!documentInfo || busy}
                rows="3"
              />
              <div className="question-footer">
                <span>
                  Only passages from your uploaded judgment are used to answer.
                </span>
                <button
                  className="ask-button"
                  type="submit"
                  disabled={!documentInfo || !question.trim() || busy}
                >
                  {busy ? (
                    <>
                      <span className="spinner" /> Working
                    </>
                  ) : (
                    <>
                      Analyze <span aria-hidden="true">↗</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {showSuggestions && (
              <div className="suggestions" aria-live="polite">
                <span className="suggestion-label">START WITH</span>
                {STARTER_QUESTIONS.map((starter) => (
                  <button
                    key={starter}
                    type="button"
                    disabled={!documentInfo || busy}
                    onClick={() => {
                      setQuestion(starter);
                    }}
                  >
                    <span aria-hidden="true">↳</span>
                    {starter}
                  </button>
                ))}
              </div>
            )}

            {error && (
              <div className="error-banner" role="alert">
                <strong>Something needs attention</strong>
                <span>{error}</span>
              </div>
            )}

            {result && (
              <div className="results">
                {result.notice && (
                  <div className="notice-banner">{result.notice}</div>
                )}
                {result.error && (
                  <div className="error-banner" role="alert">
                    <strong>Answer not generated</strong>
                    <span>
                      {result.error} Retrieved passages are still available
                      below.
                    </span>
                  </div>
                )}
                {result.answer && (
                  <article className="answer-card">
                    <div className="result-heading">
                      <div>
                        <span className="step-label">ANALYSIS</span>
                        <h3>Answer from the judgment</h3>
                      </div>
                      <span className="source-count">
                        {result.evidence.length} SOURCES
                      </span>
                    </div>
                    <div className="answer-markdown">
                      <ReactMarkdown>{result.answer}</ReactMarkdown>
                    </div>
                  </article>
                )}

                {report && (
                  <section className="support-card">
                    <div className="result-heading">
                      <div>
                        <span className="step-label">CHECK</span>
                        <h3>Support check</h3>
                      </div>
                      {report.overall_score !== null && (
                        <span
                          className={`support-score ${report.overall_score >= 0.5 ? "is-supported" : "is-weak"}`}
                        >
                          {Math.round(report.overall_score * 100)}% heuristic
                          support
                        </span>
                      )}
                    </div>
                    <p className="support-note">
                      {report.note || report.verdict}. This heuristic is a
                      review aid, not proof of legal accuracy.
                    </p>
                    {report.invalid_citations.length > 0 && (
                      <p className="invalid-citations">
                        Citations outside retrieved pages:{" "}
                        {report.invalid_citations.join(", ")}
                      </p>
                    )}
                    {report.unsupported_claims.length > 0 && (
                      <details className="claim-details">
                        <summary>
                          {report.unsupported_claims.length} claim(s) need
                          manual checking
                        </summary>
                        <ul>
                          {report.unsupported_claims.map((claim, index) => (
                            <li key={`${claim.claim}-${index}`}>
                              <strong>{claim.claim}</strong>
                              <span>
                                {claim.reasons.join("; ") ||
                                  "Weak match to retrieved evidence."}
                              </span>
                            </li>
                          ))}
                        </ul>
                      </details>
                    )}
                  </section>
                )}

                <section className="evidence-section">
                  <div className="result-heading">
                    <div>
                      <span className="step-label">SOURCE MATERIAL</span>
                      <h3>Evidence used</h3>
                    </div>
                  </div>
                  {result.evidence.map((item) => (
                    <details
                      className="evidence-item"
                      key={`${item.rank}-${item.page}`}
                    >
                      <summary>
                        <span className="evidence-rank">
                          {String(item.rank).padStart(2, "0")}
                        </span>
                        <strong>Page {item.page}</strong>
                        <span className="similarity">
                          Similarity {item.score.toFixed(2)}
                        </span>
                        <span className="chevron">⌄</span>
                      </summary>
                      <p>{item.text}</p>
                    </details>
                  ))}
                </section>
              </div>
            )}
          </section>
        </section>

        <footer className="disclaimer">
          <span className="disclaimer-mark">!</span>
          <p>
            <strong>Research tool, not legal advice.</strong> Machine-generated
            analysis can be incomplete or wrong. Verify every statement against
            the cited judgment and consult a qualified lawyer for legal matters.
          </p>
        </footer>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
