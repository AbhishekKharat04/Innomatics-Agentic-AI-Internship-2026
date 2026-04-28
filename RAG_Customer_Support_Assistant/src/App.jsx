import { useEffect, useState } from "react";

const starterPrompts = [
  "What is this project about?",
  "Which deliverables are mandatory in the assignment?",
  "I want to speak to a manager about a refund",
];

function StatusPill({ online }) {
  return (
    <div className={`status-pill ${online ? "is-online" : "is-offline"}`}>
      <span className="status-dot" />
      {online ? "API Connected" : "Checking Backend"}
    </div>
  );
}

function SourceList({ sources }) {
  if (!sources?.length) {
    return <p className="muted">No source references were returned.</p>;
  }

  return (
    <ul className="source-list">
      {sources.map((source) => (
        <li key={source}>{source}</li>
      ))}
    </ul>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState(null);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const res = await fetch("/api/health");
        if (!res.ok) {
          throw new Error("Backend health check failed.");
        }
        if (active) {
          setHealth(true);
        }
      } catch {
        if (active) {
          setHealth(false);
        }
      }
    }

    checkHealth();
    return () => {
      active = false;
    };
  }, []);

  async function askQuestion(prompt = question) {
    const normalized = prompt.trim();
    if (!normalized) {
      setError("Ask a question to query the knowledge base.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: normalized }),
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to get an answer.");
      }

      setResponse(data);
      setQuestion(normalized);
    } catch (runtimeError) {
      setError(runtimeError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="app-grid">
        <section className="hero-panel">
          <div className="hero-topline">RAG + LangGraph + HITL</div>
          <h1>Customer Support Assistant</h1>
          <p className="hero-copy">
            A browser-based interface for the assignment project. It queries the
            PDF knowledge base, shows retrieved evidence, and escalates risky
            cases to a human-review path.
          </p>

          <div className="hero-metrics">
            <div className="metric-card">
              <span className="metric-label">Frontend</span>
              <strong>React</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">Backend</span>
              <strong>Node + Express</strong>
            </div>
            <div className="metric-card">
              <span className="metric-label">RAG Core</span>
              <strong>Python + LangGraph</strong>
            </div>
          </div>

          <div className="security-card">
            <h2>Security Defaults</h2>
            <ul>
              <li>Input validation and payload size limits</li>
              <li>Rate limiting on the chat endpoint</li>
              <li>Restricted localhost CORS policy</li>
              <li>Safe Python spawning without shell interpolation</li>
              <li>Helmet-based hardening headers</li>
            </ul>
          </div>
        </section>

        <section className="chat-panel">
          <div className="chat-header">
            <div>
              <p className="eyebrow">Live Demo</p>
              <h2>Ask the assistant</h2>
            </div>
            <StatusPill online={health} />
          </div>

          <div className="prompt-list">
            {starterPrompts.map((prompt) => (
              <button
                key={prompt}
                type="button"
                className="prompt-chip"
                onClick={() => {
                  setQuestion(prompt);
                  void askQuestion(prompt);
                }}
              >
                {prompt}
              </button>
            ))}
          </div>

          <label className="composer">
            <span className="composer-label">Question</span>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about the assignment, deliverables, workflow, or escalation path..."
              maxLength={500}
              rows={5}
            />
          </label>

          <div className="composer-actions">
            <button
              type="button"
              className="primary-button"
              onClick={() => void askQuestion()}
              disabled={loading}
            >
              {loading ? "Processing..." : "Send Question"}
            </button>
            <span className="character-count">{question.length}/500</span>
          </div>

          {error ? <div className="error-banner">{error}</div> : null}

          <div className="response-panel">
            <div className="response-heading">
              <h3>Assistant Response</h3>
              {response ? (
                <span
                  className={`escalation-badge ${
                    response.escalated ? "is-escalated" : "is-resolved"
                  }`}
                >
                  {response.escalated ? "Escalated to Human" : "Answered by RAG"}
                </span>
              ) : null}
            </div>

            {response ? (
              <>
                <p className="answer-text">{response.answer}</p>

                <div className="details-grid">
                  <div className="detail-card">
                    <p className="detail-label">Source Pages</p>
                    <SourceList sources={response.sources} />
                  </div>

                  <div className="detail-card">
                    <p className="detail-label">Relevance Scores</p>
                    {response.relevanceScores?.length ? (
                      <div className="score-row">
                        {response.relevanceScores.map((score, index) => (
                          <span key={`${score}-${index}`} className="score-pill">
                            {score}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <p className="muted">No retrieval scores available.</p>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="empty-state">
                <p>
                  The interface is ready. Ask a question to see retrieval,
                  grounding, and escalation behavior.
                </p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
