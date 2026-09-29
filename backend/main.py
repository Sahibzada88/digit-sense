"""
DigitSense API — FastAPI inference service for the DigitCNN model,
with a lightweight human-in-the-loop feedback and analytics layer.

Endpoints
---------
GET  /health           -> service + model status
GET  /model-info         -> architecture + training metadata (from train.py)
POST /predict            -> {"image": "<base64 png>"} -> prediction + confidence
POST /feedback            -> {"prediction_id": int, "true_label": int} -> confirm/correct a prediction
GET  /history             -> recent predictions (for the dashboard)
GET  /stats                -> aggregate accuracy, digit distribution, confusion counts

Run (dev):
    uvicorn main:app --reload --port 8000
"""

import base64
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import cv2

from model import DigitCNN
import database

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "saved_model", "digit_cnn.pth")
METADATA_PATH = os.path.join(HERE, "saved_model", "metadata.json")
DEVICE = torch.device("cpu")

app = FastAPI(
    title="DigitSense API",
    description="PyTorch CNN inference service for handwritten digit recognition, with feedback logging.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.init_db()

# --- Load model once at startup ------------------------------------------------
model = DigitCNN()
model_loaded = False
try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

train_metadata = {}
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        train_metadata = json.load(f)


# ------------------------------------------------------------- schemas ----

class PredictRequest(BaseModel):
    image: str  # base64-encoded PNG/JPEG (data URL or raw base64)


class PredictionResponse(BaseModel):
    prediction_id: int
    predicted_digit: int
    confidence: float
    probabilities: list[float]
    inference_time_ms: float


class FeedbackRequest(BaseModel):
    prediction_id: int
    true_label: int = Field(ge=0, le=9)


# ---------------------------------------------------------- preprocessing --

def decode_base64_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64_string)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    return np.array(img)


def preprocess(img_array: np.ndarray) -> torch.Tensor:
    """Mirror training-time preprocessing: invert canvas colors to match
    MNIST's bright-on-dark convention, crop to the digit's bounding box,
    resize to 28x28, and normalize to [0, 1]."""
    img = img_array.astype(np.float32)

    if img.mean() > 127:
        img = 255.0 - img

    ys, xs = np.where(img > 20)
    if len(xs) > 0 and len(ys) > 0:
        pad = 20
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        img = img[y0:y1, x0:x1]

    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = np.clip(img / 255.0, 0, 1)

    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return tensor


# --------------------------------------------------------------- routes ----

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "model_not_found",
        "model_loaded": model_loaded,
        "device": str(DEVICE),
    }


@app.get("/model-info")
def model_info():
    return {
        "architecture": train_metadata.get(
            "architecture", "DigitCNN v2 — stem + 3 residual stages"
        ),
        "framework": train_metadata.get("framework", f"PyTorch {torch.__version__}"),
        "dataset": train_metadata.get("dataset", "MNIST"),
        "parameters": train_metadata.get("parameters", model.num_parameters()),
        "epochs_trained": train_metadata.get("epochs_trained"),
        "best_val_accuracy": train_metadata.get("best_val_accuracy"),
        "test_accuracy": train_metadata.get("test_accuracy"),
        "training_time_minutes": train_metadata.get("training_time_minutes"),
        "trained_at": train_metadata.get("trained_at"),
        "input_shape": [1, 1, 28, 28],
        "num_classes": 10,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictRequest):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model weights not found on server.")

    img_array = decode_base64_image(req.image)
    tensor = preprocess(img_array)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    predicted = int(torch.argmax(probs).item())
    confidence = float(probs[predicted].item())
    probabilities = [float(p) for p in probs.tolist()]

    prediction_id = database.log_prediction(predicted, confidence, probabilities, round(elapsed_ms, 3))

    return PredictionResponse(
        prediction_id=prediction_id,
        predicted_digit=predicted,
        confidence=confidence,
        probabilities=probabilities,
        inference_time_ms=round(elapsed_ms, 3),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    ok = database.record_feedback(req.prediction_id, req.true_label)
    if not ok:
        raise HTTPException(status_code=404, detail="prediction_id not found.")
    return {"status": "recorded"}


@app.get("/history")
def history(limit: int = 50):
    return database.get_history(limit=min(limit, 200))


@app.get("/stats")
def stats():
    return database.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
