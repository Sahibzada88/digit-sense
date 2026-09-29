import { useState } from "react";

const DIGITS = Array.from({ length: 10 }, (_, i) => i);

/**
 * After a prediction, lets the user confirm it was correct or pick the
 * actual digit they drew. This is the human-in-the-loop signal that
 * powers the /stats live-accuracy figure and confusion breakdown.
 */
export default function FeedbackBar({ predictionId, predictedDigit, apiBase, onSubmitted }) {
  const [state, setState] = useState("idle"); // idle | correcting | sent

  const sendFeedback = async (trueLabel) => {
    try {
      await fetch(`${apiBase}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prediction_id: predictionId, true_label: trueLabel }),
      });
      setState("sent");
      onSubmitted?.();
    } catch {
      // Best-effort — feedback is a nice-to-have, not critical path
      setState("sent");
    }
  };

  if (state === "sent") {
    return <p className="feedback-done">Thanks — logged for the accuracy dashboard.</p>;
  }

  if (state === "correcting") {
    return (
      <div className="feedback-correct">
        <span className="feedback-label">What digit did you actually draw?</span>
        <div className="feedback-digits">
          {DIGITS.map((d) => (
            <button key={d} className="feedback-digit-btn" onClick={() => sendFeedback(d)} type="button">
              {d}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="feedback-row">
      <span className="feedback-label">Was this right?</span>
      <button className="feedback-btn feedback-btn--yes" onClick={() => sendFeedback(predictedDigit)} type="button">
        Yes
      </button>
      <button className="feedback-btn feedback-btn--no" onClick={() => setState("correcting")} type="button">
        No
      </button>
    </div>
  );
}
