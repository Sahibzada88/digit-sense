import { useCallback, useRef, useState } from "react";
import DrawingCanvas from "./components/DrawingCanvas";
import ProbabilityBars from "./components/ProbabilityBars";
import FeedbackBar from "./components/FeedbackBar";
import AnalyticsPanel from "./components/AnalyticsPanel";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const canvasRef = useRef(null);
  const [tab, setTab] = useState("predict"); // predict | analytics
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [errorMsg, setErrorMsg] = useState("");
  const [analyticsRefreshKey, setAnalyticsRefreshKey] = useState(0);

  const resetResult = useCallback(() => {
    setResult(null);
    setStatus("idle");
  }, []);

  const handleClear = () => {
    canvasRef.current?.clear();
    resetResult();
  };

  const handlePredict = async () => {
    if (canvasRef.current?.isEmpty()) return;

    setStatus("loading");
    setErrorMsg("");
    try {
      const dataUrl = canvasRef.current.exportBase64();
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setErrorMsg(
        err.message === "Failed to fetch"
          ? "Can't reach the DigitSense API. Is the backend running on " + API_BASE + "?"
          : err.message
      );
    }
  };

  return (
    <div className="page">
      <header className="header">
        <div className="header-mark">DigitSense</div>
        <p className="header-tagline">
          A handwritten digit classifier built end-to-end with PyTorch and FastAPI —
          trained on full MNIST with a residual CNN, served live behind this React interface.
        </p>
        <div className="tabs">
          <button
            className={`tab${tab === "predict" ? " tab--active" : ""}`}
            onClick={() => setTab("predict")}
            type="button"
          >
            Predict
          </button>
          <button
            className={`tab${tab === "analytics" ? " tab--active" : ""}`}
            onClick={() => setTab("analytics")}
            type="button"
          >
            History &amp; Analytics
          </button>
        </div>
      </header>

      {tab === "predict" && (
        <main className="workspace">
          <section className="panel panel--input">
            <h2 className="panel-title">Draw a digit</h2>
            <p className="panel-hint">Use your mouse or finger — one digit, centered, fills most of the box.</p>

            <div className="canvas-frame">
              <DrawingCanvas ref={canvasRef} onStrokeStart={resetResult} disabled={status === "loading"} />
            </div>

            <div className="controls">
              <button className="btn btn--ghost" onClick={handleClear} type="button">
                Clear
              </button>
              <button
                className="btn btn--primary"
                onClick={handlePredict}
                type="button"
                disabled={status === "loading"}
              >
                {status === "loading" ? "Reading digit…" : "Predict"}
              </button>
            </div>

            {status === "error" && <p className="error-text">{errorMsg}</p>}
          </section>

          <section className="panel panel--result">
            <h2 className="panel-title">Prediction</h2>

            {!result && status !== "loading" && (
              <div className="empty-state">
                <p>Draw a digit and press Predict to see the model's read.</p>
              </div>
            )}

            {status === "loading" && (
              <div className="empty-state">
                <p>Running inference…</p>
              </div>
            )}

            {result && status !== "loading" && (
              <>
                <div className="result-hero">
                  <span className="result-digit">{result.predicted_digit}</span>
                  <div className="result-meta">
                    <span className="result-confidence">
                      {(result.confidence * 100).toFixed(1)}% confidence
                    </span>
                    <span className="result-latency">
                      {result.inference_time_ms.toFixed(2)} ms inference
                    </span>
                  </div>
                </div>

                <FeedbackBar
                  key={result.prediction_id}
                  predictionId={result.prediction_id}
                  predictedDigit={result.predicted_digit}
                  apiBase={API_BASE}
                  onSubmitted={() => setAnalyticsRefreshKey((k) => k + 1)}
                />

                <h3 className="prob-title">Probability by digit</h3>
                <ProbabilityBars
                  probabilities={result.probabilities}
                  predicted={result.predicted_digit}
                />
              </>
            )}
          </section>
        </main>
      )}

      {tab === "analytics" && (
        <main className="workspace workspace--single">
          <section className="panel panel--analytics">
            <AnalyticsPanel refreshKey={analyticsRefreshKey} />
          </section>
        </main>
      )}

      <footer className="footer">
        <span>Model: residual CNN, PyTorch, full MNIST · Backend: FastAPI + SQLite · Frontend: React</span>
        <span className="footer-sep">·</span>
      </footer>
    </div>
  );
}
