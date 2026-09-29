"""
tests/test_api.py — unit and integration tests for the DigitSense API.

Run:  pytest -v
"""

import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app, preprocess, decode_base64_image  # noqa: E402

client = TestClient(app)


def make_canvas_png_b64(draw_fn) -> str:
    """Build a 280x280 white-background PNG, draw on it with draw_fn(draw),
    and return it as a base64 data URL like the frontend sends."""
    img = Image.new("L", (280, 280), color=255)
    d = ImageDraw.Draw(img)
    draw_fn(d)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


# ------------------------------------------------------------- preprocessing --

def test_decode_base64_image_handles_data_url_prefix():
    b64_img = make_canvas_png_b64(lambda d: d.ellipse([90, 90, 190, 190], outline=0, width=10))
    arr = decode_base64_image(b64_img)
    assert arr.ndim == 2
    assert arr.shape == (280, 280)


def test_decode_base64_image_rejects_garbage():
    with pytest.raises(Exception):
        decode_base64_image("not-valid-base64!!")


def test_preprocess_output_shape_and_range():
    img = np.full((280, 280), 255, dtype=np.uint8)
    img[100:180, 100:180] = 0  # a black square "stroke" on white background
    tensor = preprocess(img)
    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocess_inverts_dark_background_correctly():
    # A mostly-white canvas should end up with the drawn region as the
    # bright pixels after preprocessing (matching MNIST's convention).
    img = np.full((280, 280), 255, dtype=np.uint8)
    img[120:160, 120:160] = 0
    tensor = preprocess(img)
    assert tensor.mean() > 0.05  # some bright signal remains after crop+normalize


# ------------------------------------------------------------------- API ----

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


def test_model_info_endpoint():
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_classes"] == 10
    assert body["input_shape"] == [1, 1, 28, 28]


def test_predict_endpoint_returns_valid_prediction():
    b64_img = make_canvas_png_b64(lambda d: d.ellipse([80, 60, 190, 220], outline=0, width=16))
    resp = client.post("/predict", json={"image": b64_img})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["predicted_digit"] <= 9
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 10
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-3
    assert body["inference_time_ms"] >= 0
    assert isinstance(body["prediction_id"], int)


def test_predict_endpoint_rejects_bad_payload():
    resp = client.post("/predict", json={"image": "garbage"})
    assert resp.status_code == 400


def test_feedback_roundtrip():
    b64_img = make_canvas_png_b64(lambda d: d.line([100, 80, 100, 200], fill=0, width=16))
    predict_resp = client.post("/predict", json={"image": b64_img})
    prediction_id = predict_resp.json()["prediction_id"]

    fb_resp = client.post("/feedback", json={"prediction_id": prediction_id, "true_label": 1})
    assert fb_resp.status_code == 200
    assert fb_resp.json()["status"] == "recorded"


def test_feedback_unknown_id_returns_404():
    resp = client.post("/feedback", json={"prediction_id": 999999999, "true_label": 3})
    assert resp.status_code == 404


def test_feedback_rejects_out_of_range_label():
    resp = client.post("/feedback", json={"prediction_id": 1, "true_label": 42})
    assert resp.status_code == 422  # pydantic validation error


def test_history_endpoint_shape():
    resp = client.get("/history?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) <= 5


def test_stats_endpoint_shape():
    resp = client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    for key in ["total_predictions", "labeled_predictions", "live_accuracy", "digit_distribution"]:
        assert key in body
    assert len(body["digit_distribution"]) == 10
