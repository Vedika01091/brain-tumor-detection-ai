import { CLASS_META, CLASS_ORDER } from "../classInfo";

export default function ProbabilityBars({ probabilities }) {
  return (
    <div className="prob-bars" role="table" aria-label="Class probability breakdown">
      {CLASS_ORDER.map((cls) => {
        const meta = CLASS_META[cls];
        const value = probabilities[cls] ?? 0;
        const pct = Math.round(value * 1000) / 10;
        return (
          <div className="prob-bar-row" role="row" key={cls}>
            <span className="prob-bar-name" role="rowheader">
              <span
                className="prob-bar-swatch"
                style={{ background: `var(${meta.colorVar})` }}
                aria-hidden="true"
              />
              {meta.label}
            </span>
            <span
              className="prob-bar-track"
              role="cell"
              aria-label={`${meta.label}: ${pct}%`}
            >
              <span
                className="prob-bar-fill"
                style={{
                  width: `${Math.max(pct, 1.5)}%`,
                  background: `var(${meta.colorVar})`,
                }}
              />
            </span>
            <span className="prob-bar-value" role="cell">
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
