import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import "./App.css";
import Hero from "./components/Hero";
import UploadPanel from "./components/UploadPanel";
import ResultPanel from "./components/ResultPanel";
import HistoryPanel from "./components/HistoryPanel";
import { checkHealth, getModelInfo, predictImage } from "./api";

let nextId = 1;

function App() {
  const [history, setHistory] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    checkHealth()
      .then((res) => setBackendStatus(res.model_loaded ? "ready" : "model-not-ready"))
      .catch(() => setBackendStatus("offline"));
    getModelInfo()
      .then(setModelInfo)
      .catch(() => {});
  }, []);

  const activeResult = history.find((item) => item.id === activeId) ?? null;

  async function handleAnalyze(file) {
    setLoading(true);
    setError(null);
    try {
      const res = await predictImage(file);
      const entry = { id: nextId++, ...res };
      setHistory((prev) => [entry, ...prev]);
      setActiveId(entry.id);
    } catch (err) {
      setError(err.message || "Something went wrong while analyzing the scan.");
    } finally {
      setLoading(false);
    }
  }

  const overallAccuracy = modelInfo?.metrics?.accuracy;

  return (
    <div className="app-root">
      <Hero accuracy={overallAccuracy} />

      <section id="workspace" className="workspace container">
        <div className="section-heading">
          <span className="eyebrow">Live Demo</span>
          <h2>Upload a scan, get an explained diagnosis</h2>
          <p>Results are rendered client-side straight from the Flask inference API &mdash; nothing leaves your machine.</p>
        </div>

        <main className="app-main">
          <div className="left-column">
            <UploadPanel onAnalyze={handleAnalyze} loading={loading} error={error} />
            <HistoryPanel history={history} activeId={activeId} onSelect={setActiveId} />
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeResult?.id ?? "empty"}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              <ResultPanel result={activeResult} />
            </motion.div>
          </AnimatePresence>
        </main>
      </section>

      <footer className="app-footer container">
        <span>
          Backend:{" "}
          {backendStatus === "ready" && "connected, model loaded"}
          {backendStatus === "checking" && "connecting…"}
          {backendStatus === "model-not-ready" && "connected, model not trained yet"}
          {backendStatus === "offline" && "unreachable (start the Flask server)"}
        </span>
        {typeof overallAccuracy === "number" && (
          <span>Test-set accuracy: {(overallAccuracy * 100).toFixed(2)}%</span>
        )}
        <span>4-class EfficientNet-B0 transfer learning &middot; glioma / meningioma / pituitary / no tumour</span>
      </footer>
    </div>
  );
}

export default App;
