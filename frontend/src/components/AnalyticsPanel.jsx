import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, CartesianGrid } from "recharts";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function StatCard({ label, value, hint }) {
  return (
    <div className="stat-card">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export default function AnalyticsPanel({ refreshKey }) {
  const [stats, setStats] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      fetch(`${API_BASE}/stats`).then((r) => r.json()),
      fetch(`${API_BASE}/history?limit=15`).then((r) => r.json()),
    ])
      .then(([statsData, historyData]) => {
        if (!cancelled) {
          setStats(statsData);
          setHistory(historyData);
        }
      })
      .catch(() => {})
      .finally(() => !cancelled && setLoading(false));

    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  if (loading && !stats) {
    return <p className="analytics-loading">Loading analytics…</p>;
  }

  if (!stats) {
    return <p className="analytics-loading">Couldn't reach the analytics endpoint.</p>;
  }

  const chartData = Object.entries(stats.digit_distribution || {}).map(([digit, count]) => ({
    digit,
    count,
  }));

  const accuracyDisplay =
    stats.live_accuracy === null || stats.live_accuracy === undefined
      ? "—"
      : `${(stats.live_accuracy * 100).toFixed(1)}%`;

  return (
    <div className="analytics">
      <div className="stat-grid">
        <StatCard label="Total predictions" value={stats.total_predictions} />
        <StatCard
          label="Live accuracy"
          value={accuracyDisplay}
          hint={`from ${stats.labeled_predictions} labeled`}
        />
        <StatCard label="Avg. confidence" value={`${(stats.avg_confidence * 100).toFixed(1)}%`} />
        <StatCard label="Avg. latency" value={`${stats.avg_inference_time_ms.toFixed(2)} ms`} />
      </div>

      <h3 className="analytics-subtitle">Predictions by digit</h3>
      <div className="chart-frame">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#242a3a" vertical={false} />
            <XAxis dataKey="digit" stroke="#8a93a8" fontSize={12} tickLine={false} axisLine={{ stroke: "#242a3a" }} />
            <YAxis stroke="#8a93a8" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
            <Tooltip
              contentStyle={{ background: "#181d2a", border: "1px solid #242a3a", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#8a93a8" }}
              cursor={{ fill: "rgba(77,232,201,0.06)" }}
            />
            <Bar dataKey="count" fill="#4de8c9" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h3 className="analytics-subtitle">Recent predictions</h3>
      <div className="history-table-frame">
        <table className="history-table">
          <thead>
            <tr>
              <th>Predicted</th>
              <th>Confidence</th>
              <th>Actual</th>
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 && (
              <tr>
                <td colSpan={4} className="history-empty">
                  No predictions logged yet.
                </td>
              </tr>
            )}
            {history.map((row) => {
              const hasLabel = row.true_label !== null && row.true_label !== undefined;
              const correct = hasLabel && row.true_label === row.predicted_digit;
              return (
                <tr key={row.id}>
                  <td className="mono">{row.predicted_digit}</td>
                  <td className="mono">{(row.confidence * 100).toFixed(1)}%</td>
                  <td className={hasLabel ? (correct ? "tag-correct" : "tag-wrong") : "mono muted"}>
                    {hasLabel ? row.true_label : "—"}
                  </td>
                  <td className="mono muted">{row.inference_time_ms.toFixed(2)} ms</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
