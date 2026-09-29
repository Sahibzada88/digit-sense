# DigitSense — Handwritten Digit Recognition

An end-to-end deep learning system: a residual CNN trained from scratch on
the full MNIST dataset, served through a FastAPI backend with a SQLite
logging/feedback layer, and consumed by a separate React frontend where you
draw a digit, get a live prediction, and optionally confirm or correct it.

Built as a demonstration project for the **Research Associate / Research
Staff** role at the NCAI Deep Learning Lab, NUST — the architecture and
tooling choices deliberately mirror the skills listed in that posting.

```
┌───────────────────────┐   POST /predict, /feedback   ┌─────────────────────────┐
│    React frontend       │ ───── base64 PNG image ─────►│    FastAPI backend        │
│  • drawing canvas        │                                │  • preprocessing pipeline │
│  • feedback (yes/correct)│ ◄──── JSON prediction ───────  │  • inference (PyTorch)    │
│  • analytics dashboard   │                                │  • SQLite logging         │
└───────────────────────┘   GET /stats, /history        └─────────────┬───────────┘
                                                                          │
                                                              ┌───────────▼───────────┐
                                                              │  DigitCNN v2 (.pth)      │
                                                              │  residual CNN, trained   │
                                                              │  on full MNIST 60k imgs  │
                                                              └────────────────────────┘
```

## What it does

You draw a digit (0-9) on a canvas. The frontend sends it to the backend,
which preprocesses it (invert, crop to bounding box, resize, normalize) and
runs it through a trained residual CNN. Every prediction is logged to
SQLite; you can then tell the app whether it was right — that feedback
feeds a live **accuracy dashboard** with a digit-distribution chart and a
recent-predictions table, the same kind of monitoring a deployed model
needs beyond the training/test split.

## Project structure

```
digitsense/
├── docker-compose.yml
├── .github/workflows/backend-tests.yml   # CI: runs pytest on every backend change
├── backend/
│   ├── model.py             # DigitCNN v2 — residual CNN (stem + 3 residual stages)
│   ├── train.py              # Full MNIST training pipeline (augmentation, early stopping, plots)
│   ├── database.py            # SQLite logging: predictions, feedback, aggregate stats
│   ├── main.py                 # FastAPI app: /predict, /feedback, /history, /stats, /model-info
│   ├── tests/test_api.py        # pytest suite: preprocessing + full API contract
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/mnist.pkl.gz          # Full MNIST (50k train / 10k val / 10k test)
│   ├── saved_model/
│   │   ├── digit_cnn.pth            # Trained weights
│   │   └── metadata.json             # Run metadata (accuracy, params, training time)
│   └── artifacts/                      # training_curves.png, confusion_matrix.png, classification_report.txt
└── frontend/
    ├── Dockerfile
    ├── src/
    │   ├── App.jsx                          # Tab layout: Predict / History & Analytics
    │   ├── App.css                            # Design system / styling
    │   └── components/
    │       ├── DrawingCanvas.jsx                # Freehand canvas input
    │       ├── ProbabilityBars.jsx                # Per-digit probability visualization
    │       ├── FeedbackBar.jsx                     # "Was this right?" human-in-the-loop widget
    │       └── AnalyticsPanel.jsx                   # Live stats, digit-distribution chart, history table
    ├── package.json
    └── .env.example
```

## Running it

### Option A — Docker Compose (simplest)
```bash
docker compose up --build
```
Backend on `http://localhost:8000`, frontend on `http://localhost:5173`.

### Option B — run natively

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend** (separate terminal)
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

**Retraining the model** (optional — a trained checkpoint is already
included). This is a real, full-size training run — budget ~15-30 minutes
on a modern multi-core laptop with 16 GB RAM; the default settings (18
epochs, batch size 128, early stopping) were chosen for that class of
machine.
```bash
cd backend
python3 train.py --epochs 18 --batch-size 128
```

**Running tests**
```bash
cd backend
pytest -v
```

## How the model works

- **Data**: the full MNIST dataset — 50,000 training / 10,000 validation /
  10,000 held-out test images, 28×28 grayscale. (v1 of this project used
  scikit-learn's 1,797-sample toy `digits` dataset; this version trains on
  the real, standard benchmark.)
- **Augmentation**: on-the-fly random rotation (±12°), translation, scaling,
  and light Gaussian noise on the training split only, so the model
  generalizes to imperfectly centered freehand canvas strokes rather than
  memorizing clean, centered MNIST digits.
- **Architecture**: a stem convolution followed by three residual stages
  (32→64→128→256 channels, each stage downsampling via a strided
  convolution with a projection shortcut), global average pooling, and a
  dropout-regularized classifier head — about 2.8M parameters, meaningfully
  deeper than a plain CNN and demonstrating a real architectural pattern
  (residual connections) rather than just stacking conv layers.
- **Training**: AdamW optimizer, label smoothing, cosine-annealed learning
  rate, early stopping on validation accuracy, best checkpoint selection —
  see `train.py` for the full loop.
- **Diagnostics**: `train.py` writes a loss/accuracy curve plot, a
  confusion matrix, and a classification report to `backend/artifacts/`
  after every run, and a `metadata.json` the API reads to populate
  `/model-info`.
- **Inference**: the FastAPI backend loads the trained weights once at
  startup and serves predictions in low-single-digit milliseconds on CPU.

## The feedback loop

Training-set accuracy tells you how the model does on MNIST's clean,
centered digits — not on someone's actual freehand strokes on a laptop
trackpad. So after each prediction, the UI asks "was this right?" If you
correct it, that label is stored alongside the prediction. `/stats`
aggregates this into:

- a live accuracy figure computed only from human-labeled predictions
  (distinct from the offline test-set accuracy)
- per-digit prediction volume
- a raw confusion count table (true label vs. predicted label) for anything
  that's been labeled

This is a small, self-contained version of the monitoring/labeling
pipelines used to catch real-world drift in deployed models — a natural
thing to talk through in an interview.

## Why this project fits the role

| Job requirement | Where it shows up here |
|---|---|
| Strong understanding of ML, deep learning, neural networks | Residual CNN design in `model.py` — skip connections, batch norm, dropout, global average pooling |
| Proficiency in Python for AI/ML | Entire backend, training pipeline, and data layer |
| Hands-on experience with PyTorch | Custom `nn.Module` architecture, training loop, LR scheduling, checkpointing, inference — all built directly on `torch` |
| Develop, train, evaluate, optimize models | `train.py` — augmentation, early stopping, cosine LR schedule, held-out test evaluation, confusion matrix, classification report |
| End-to-end AI application development, integration, deployment | Trained model served via FastAPI, logged to SQLite, consumed by an independent React frontend, containerized with Docker Compose |
| Familiarity with relevant Python libraries/tools | `torch`, `scikit-learn`, `numpy`, `opencv`, `matplotlib`, `FastAPI`, `pydantic`, `sqlite3`, `pytest` |
| Ability to troubleshoot technical challenges | Preprocessing pipeline explicitly bridges the gap between clean MNIST statistics and real freehand canvas input; feedback loop surfaces real-world drift the offline test set can't |
| Communication skills | This README, inline docstrings, a typed API contract, and a CI workflow that documents how the project is meant to be verified |

## Next steps / how this would be extended further

- Swap in a larger backbone (e.g. a small EfficientNet or a proper
  ResNet-18) and compare against this baseline.
- Add active learning: periodically retrain on the human-corrected
  examples collected via `/feedback`.
- Track experiments (MLflow or Weights & Biases) across architectures and
  hyperparameters instead of a single `metadata.json`.
- Add authentication/rate limiting to the API for a public deployment.
- Multi-digit support: segment a drawn sequence into individual digits
  before classification.

---
