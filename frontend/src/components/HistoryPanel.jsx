import { CLASS_META } from "../classInfo";

export default function HistoryPanel({ history, activeId, onSelect }) {
  return (
    <div className="card">
      <h2>Session History</h2>
      {history.length === 0 ? (
        <p className="history-empty">Analyzed scans will appear here.</p>
      ) : (
        <div className="history-list">
          {history.map((item) => {
            const meta = CLASS_META[item.prediction];
            return (
              <button
                type="button"
                key={item.id}
                className={`history-item${item.id === activeId ? " active" : ""}`}
                onClick={() => onSelect(item.id)}
              >
                <img
                  className="history-thumb"
                  src={item.original_image}
                  alt=""
                  aria-hidden="true"
                />
                <span>
                  <div className="history-label">{meta.label}</div>
                  <div className="history-confidence">
                    {(item.confidence * 100).toFixed(1)}% confidence
                  </div>
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
