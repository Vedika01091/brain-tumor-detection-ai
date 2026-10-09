import { useEffect, useState } from "react";
import { CLASS_META } from "../classInfo";
import ProbabilityBars from "./ProbabilityBars";

export default function ResultPanel({ result }) {
  const [view, setView] = useState("heatmap");

  useEffect(() => {
    setView("heatmap");
  }, [result]);

  if (!result) {
    return (
      <div className="card">
        <div className="result-placeholder">
          <div className="icon" aria-hidden="true">
            🧠
          </div>
          <p>Upload an MRI scan and click Analyze to see results here.</p>
        </div>
      </div>
    );
  }

  const meta = CLASS_META[result.prediction];
  const confidencePct = (result.confidence * 100).toFixed(1);
  const isTumor = result.tumor_detected;

  return (
    <div className="card">
      <h2>Analysis Result</h2>

      <div className={`diagnosis-banner ${isTumor ? "status-critical" : "status-good"}`}>
        <span className="icon" aria-hidden="true">
          {isTumor ? "⚠" : "✅"}
        </span>
        <div>
          <div className="diagnosis-title">
            {isTumor ? `Possible ${meta.label} detected` : "No tumour detected"}
          </div>
          <div className="diagnosis-sub">
            Confidence {confidencePct}% &mdash; {meta.description}
          </div>
        </div>
      </div>

      <div className="image-toggle" role="group" aria-label="Toggle image view">
        <button
          type="button"
          className={view === "original" ? "active" : ""}
          onClick={() => setView("original")}
        >
          Original
        </button>
        <button
          type="button"
          className={view === "heatmap" ? "active" : ""}
          onClick={() => setView("heatmap")}
        >
          Grad-CAM Heatmap
        </button>
      </div>

      <div className="result-image-frame">
        {view === "original" ? (
          <img src={result.original_image} alt="Uploaded MRI scan" />
        ) : (
          <img
            src={result.heatmap_overlay}
            alt={`MRI scan with Grad-CAM tumor attention overlay, predicted ${meta.label}`}
          />
        )}
      </div>

      <h2 style={{ marginTop: 22 }}>Class Probabilities</h2>
      <ProbabilityBars probabilities={result.probabilities} />
    </div>
  );
}
