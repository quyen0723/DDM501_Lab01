# DDM501 Lab 1 — First ML Product (Movie Rating Prediction API)

> The actual project lives in [`lab1/`](./lab1) — open [`lab1/README.md`](./lab1/README.md) for the full documentation.

## What this is

A production-ready **Movie Rating Prediction** service for course **DDM501** (Lab 1, weight 5%).
A collaborative-filtering model (SVD / Matrix Factorization) trained on the **MovieLens 100K**
dataset is wrapped in a **FastAPI** REST API, containerized with **Docker**, and covered by a
**pytest** test suite.

## Repository layout

```
DDM501_Lab01/
├── README.md            ← you are here (project landing page)
└── lab1/                ← the deliverable
    ├── app/             # FastAPI app, model wrapper, Pydantic schemas, config
    ├── scripts/         # train_model.py — trains & saves the SVD model
    ├── tests/           # pytest suite (18 tests) + conftest.py (self-contained)
    ├── models/          # svd_model.pkl is generated here (gitignored)
    ├── Dockerfile       # with HEALTHCHECK
    ├── docker-compose.yml
    ├── requirements.txt
    ├── pytest.ini
    └── README.md        # full documentation (setup, API, tests, Docker)
```

## Deliverables vs. grading rubric

| Rubric (weight) | Where |
|-----------------|-------|
| Working ML Model (25%) — loads, valid predictions 1–5, error handling | `lab1/app/model.py`, `lab1/scripts/train_model.py` |
| REST API (25%) — `/health`, `/predict`, input validation, error responses | `lab1/app/main.py`, `lab1/app/schemas.py` |
| Docker Setup (20%) — Dockerfile builds, compose works, healthcheck | `lab1/Dockerfile`, `lab1/docker-compose.yml` |
| Test Cases (20%) — happy path, edge cases, tests pass | `lab1/tests/test_api.py`, `lab1/tests/conftest.py` |
| Documentation (10%) — README + Swagger `/docs` | `lab1/README.md`, FastAPI auto-docs |

## Quick start

```bash
cd lab1

# 1. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train the model (downloads MovieLens 100K, saves models/svd_model.pkl)
python scripts/train_model.py

# 3. Run the API  →  Swagger at http://localhost:8000/docs
uvicorn app.main:app --reload

# 4. Run the tests (18 tests)
pytest tests/ -v

# 5. Run with Docker (train the model first so ./models/svd_model.pkl exists)
docker compose build && docker compose up -d
curl http://localhost:8000/health   # → {"status":"healthy","model_loaded":true}
```

> **Tests are self-contained.** `tests/conftest.py` guarantees a model exists before the app
> loads, so `pytest tests/ -v` passes **18/18 on a fresh clone** even before you run
> `train_model.py` (it trains a small SVD automatically if `models/svd_model.pkl` is absent).
> Once you train the real model, the same suite runs against the actual deliverable.

## Team

DDM501 — Lab 1 (team lab, 3–4 members).