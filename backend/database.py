"""
database.py — a small SQLite-backed store for prediction logs and
human feedback.

Every call to /predict is recorded here. The frontend can optionally
send back the true label via /feedback (a lightweight human-in-the-loop
signal), which lets /stats report a real accuracy figure, a live
confusion matrix, and per-digit volume — the kind of monitoring a
deployed model needs in production, not just at training time.
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "predictions.db"
DB_PATH.parent.mkdir(exist_ok=True)


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    predicted_digit INTEGER NOT NULL,
    confidence REAL NOT NULL,
    probabilities TEXT NOT NULL,
    inference_time_ms REAL NOT NULL,
    true_label INTEGER
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def log_prediction(predicted_digit: int, confidence: float, probabilities: list[float], inference_time_ms: float) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO predictions (created_at, predicted_digit, confidence, probabilities, inference_time_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), predicted_digit, confidence, json.dumps(probabilities), inference_time_ms),
        )
        return cur.lastrowid


def record_feedback(prediction_id: int, true_label: int) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE predictions SET true_label = ? WHERE id = ?",
            (true_label, prediction_id),
        )
        return cur.rowcount > 0


def get_history(limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, created_at, predicted_digit, confidence, true_label, inference_time_ms "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]

        labeled = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE true_label IS NOT NULL"
        ).fetchone()["n"]

        correct = conn.execute(
            "SELECT COUNT(*) AS n FROM predictions WHERE true_label IS NOT NULL AND true_label = predicted_digit"
        ).fetchone()["n"]

        avg_confidence = conn.execute("SELECT AVG(confidence) AS a FROM predictions").fetchone()["a"] or 0.0
        avg_latency = conn.execute("SELECT AVG(inference_time_ms) AS a FROM predictions").fetchone()["a"] or 0.0

        distribution_rows = conn.execute(
            "SELECT predicted_digit, COUNT(*) AS n FROM predictions GROUP BY predicted_digit"
        ).fetchall()
        distribution = {str(d): 0 for d in range(10)}
        for row in distribution_rows:
            distribution[str(row["predicted_digit"])] = row["n"]

        confusion_rows = conn.execute(
            "SELECT true_label, predicted_digit, COUNT(*) AS n FROM predictions "
            "WHERE true_label IS NOT NULL GROUP BY true_label, predicted_digit"
        ).fetchall()
        confusion = [{"true": r["true_label"], "predicted": r["predicted_digit"], "count": r["n"]} for r in confusion_rows]

        return {
            "total_predictions": total,
            "labeled_predictions": labeled,
            "live_accuracy": (correct / labeled) if labeled > 0 else None,
            "avg_confidence": avg_confidence,
            "avg_inference_time_ms": avg_latency,
            "digit_distribution": distribution,
            "confusion": confusion,
        }
