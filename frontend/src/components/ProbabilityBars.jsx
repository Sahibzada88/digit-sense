/**
 * Horizontal bar chart showing the model's predicted probability for
 * each digit 0-9, with the top prediction highlighted.
 */
export default function ProbabilityBars({ probabilities, predicted }) {
  if (!probabilities) return null;

  return (
    <div className="prob-list">
      {probabilities.map((p, digit) => {
        const pct = Math.round(p * 1000) / 10; // one decimal place
        const isTop = digit === predicted;
        return (
          <div className={`prob-row${isTop ? " prob-row--top" : ""}`} key={digit}>
            <span className="prob-digit">{digit}</span>
            <div className="prob-track">
              <div
                className="prob-fill"
                style={{ width: `${Math.max(pct, 1.2)}%` }}
              />
            </div>
            <span className="prob-value">{pct.toFixed(1)}%</span>
          </div>
        );
      })}
    </div>
  );
}
