# Lab 1: First ML Product - Movie Rating Prediction API

A production-ready Movie Rating Prediction service. A collaborative filtering model (SVD, Matrix Factorization) trained on the MovieLens 100K dataset is wrapped in a FastAPI REST API, containerized with Docker, and covered by a pytest test suite.

## Features

- **SVD collaborative filtering** model trained on MovieLens 100K (100,000 ratings, 943 users, 1,682 movies)
- **REST API** with single and batch prediction endpoints
- **Input validation** with Pydantic (missing/invalid fields → HTTP 422)
- **Health check** endpoint for monitoring, wired into Docker's `HEALTHCHECK`
- **Graceful error handling**: 503 when the model isn't loaded, 500 on prediction errors
- **Auto-generated Swagger docs** at `/docs`
- **Unit + edge-case tests** with pytest (18 tests)

## Project Structure

```
ddm501-lab1/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application & endpoints
│   ├── model.py          # ML model loading & prediction wrapper
│   ├── schemas.py        # Pydantic request/response models
│   └── config.py         # Configuration (env-driven)
├── models/               # Saved ML models (svd_model.pkl)
├── tests/
│   ├── __init__.py
│   └── test_api.py       # Unit & edge-case tests
├── scripts/
│   └── train_model.py    # Model training script
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

## Quick Start

### 1. Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train_model.py
```

This downloads the MovieLens 100K dataset, runs 5-fold cross-validation (RMSE/MAE), trains an SVD model on the full dataset, and saves it to `models/svd_model.pkl`.

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health
# → {"status":"healthy","model_loaded":true}

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "196", "movie_id": "242"}'
# → {"user_id":"196","movie_id":"242","predicted_rating":3.87,"model_version":"1.0.0"}

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"predictions":[{"user_id":"196","movie_id":"242"},{"user_id":"186","movie_id":"302"}]}'

# Model info
curl http://localhost:8000/model/info
```

### 5. Run with Docker

```bash
# Train the model first so ./models/svd_model.pkl exists (mounted into the container)
python scripts/train_model.py

docker-compose build
docker-compose up -d

# Verify
curl http://localhost:8000/health
docker ps   # STATUS column shows (healthy) once the healthcheck passes
```

## API Endpoints

| Method | Endpoint         | Description                                  |
|--------|------------------|----------------------------------------------|
| GET    | `/`              | API information                              |
| GET    | `/health`        | Health check (status + model_loaded)         |
| GET    | `/model/info`    | Model version and type                       |
| POST   | `/predict`       | Predict rating for one user–movie pair       |
| POST   | `/predict/batch` | Predict ratings for multiple pairs (bonus)   |
| GET    | `/docs`          | Swagger documentation                        |

### Error Responses

| Status | Meaning                                            |
|--------|----------------------------------------------------|
| 422    | Validation error (missing/invalid request fields)  |
| 503    | Model not loaded                                   |
| 500    | Internal prediction error                          |

Unknown user or movie IDs still return a valid prediction — SVD falls back to the global mean rating for unseen users/items.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html
```

The suite covers happy paths (`/health`, `/`, `/predict`, `/model/info`, `/predict/batch`), validation errors (missing fields, empty body, wrong types), and edge cases (unknown user/movie, special characters in IDs).

## Configuration

Settings are read from environment variables (see `app/config.py`):

| Variable        | Default                     | Description              |
|-----------------|-----------------------------|--------------------------|
| `MODEL_PATH`    | `models/svd_model.pkl`      | Path to the saved model  |
| `MODEL_VERSION` | `1.0.0`                     | Reported model version   |
| `HOST` / `PORT` | `0.0.0.0` / `8000`          | Server bind address      |

## Model Details

- **Algorithm**: SVD (Matrix Factorization–based collaborative filtering, `scikit-surprise`)
- **Hyperparameters**: 100 latent factors, 20 epochs, lr=0.005, reg=0.02
- **Evaluation**: 5-fold cross-validation, RMSE ≈ 0.93, MAE ≈ 0.74 on MovieLens 100K
- **Output**: predicted rating in the range 1.0–5.0, rounded to 2 decimals
