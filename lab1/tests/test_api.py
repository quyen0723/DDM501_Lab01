"""
Unit tests for Movie Rating Prediction API.

Run tests with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=html
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# Create test client
client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def run_app_lifespan():
    """
    Run the FastAPI startup/shutdown events for the whole test session.

    The module-level TestClient does not trigger the @app.on_event("startup")
    handler by itself, so the model would never be loaded during tests.
    Entering the client as a context manager runs the lifespan events.
    """
    with client:
        yield


# =============================================================================
# Health Check Tests (PROVIDED)
# =============================================================================
class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 status code."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_format(self):
        """Test that health response has correct format."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


# =============================================================================
# Root Endpoint Tests (PROVIDED)
# =============================================================================
class TestRootEndpoint:
    """Tests for the / endpoint."""

    def test_root_returns_200(self):
        """Test that root endpoint returns 200 status code."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_api_info(self):
        """Test that root response contains API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs" in data


# =============================================================================
# TODO 1: Prediction Endpoint Tests
# =============================================================================
class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_response_format(self):
        """Test that prediction response has correct format."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "242"}
        )
        assert response.status_code == 200
        data = response.json()

        assert "user_id" in data
        assert "movie_id" in data
        assert "predicted_rating" in data
        assert "model_version" in data
        assert data["user_id"] == "196"
        assert data["movie_id"] == "242"
        assert isinstance(data["predicted_rating"], float)
        assert isinstance(data["model_version"], str)

    def test_predict_missing_user_id(self):
        """Test prediction with missing user_id."""
        response = client.post(
            "/predict",
            json={"movie_id": "242"}  # Missing user_id
        )
        assert response.status_code == 422

    def test_predict_missing_movie_id(self):
        """Test prediction with missing movie_id."""
        response = client.post(
            "/predict",
            json={"user_id": "196"}  # Missing movie_id
        )
        assert response.status_code == 422

    def test_predict_empty_body(self):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        assert response.status_code == 422


# =============================================================================
# TODO 2: Edge Case Tests (BONUS)
# =============================================================================
class TestEdgeCases:
    """Edge case tests."""

    def test_predict_unknown_user(self):
        """Test prediction with unknown user ID."""
        # SVD falls back to the global mean for unknown users,
        # so the API should still return a valid rating.
        response = client.post(
            "/predict",
            json={"user_id": "999999", "movie_id": "242"}
        )
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_unknown_movie(self):
        """Test prediction with unknown movie ID."""
        response = client.post(
            "/predict",
            json={"user_id": "196", "movie_id": "999999"}
        )
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_special_characters_in_id(self):
        """Test prediction with special characters in IDs."""
        # IDs are strings, so special characters are treated as
        # unknown users/movies and should still get a fallback rating.
        response = client.post(
            "/predict",
            json={"user_id": "user@#$%", "movie_id": "movie!*()"}
        )
        assert response.status_code == 200
        data = response.json()
        assert 1.0 <= data["predicted_rating"] <= 5.0

    def test_predict_wrong_type_returns_422_or_coerces(self):
        """Non-string JSON types should be rejected by validation."""
        response = client.post(
            "/predict",
            json={"user_id": ["196"], "movie_id": {"id": "242"}}
        )
        assert response.status_code == 422


# =============================================================================
# TODO 3: Model Info Endpoint Tests
# =============================================================================
class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self):
        """Test that model info endpoint returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_contains_version(self):
        """Test that model info contains version."""
        response = client.get("/model/info")
        data = response.json()

        assert "model_version" in data
        assert "model_type" in data
        assert "is_loaded" in data
        assert isinstance(data["model_version"], str)


# =============================================================================
# Batch Prediction Tests (BONUS)
# =============================================================================
class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint (BONUS)."""

    def test_batch_predict_multiple_items(self):
        """Test batch prediction with multiple items."""
        response = client.post(
            "/predict/batch",
            json={
                "predictions": [
                    {"user_id": "196", "movie_id": "242"},
                    {"user_id": "186", "movie_id": "302"},
                    {"user_id": "22", "movie_id": "377"},
                ]
            }
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 3
        assert len(data["predictions"]) == 3
        for item in data["predictions"]:
            assert 1.0 <= item["predicted_rating"] <= 5.0

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty list."""
        response = client.post(
            "/predict/batch",
            json={"predictions": []}
        )
        assert response.status_code == 200
        data = response.json()

        assert data["total_count"] == 0
        assert data["predictions"] == []

    def test_batch_predict_missing_field(self):
        """Test batch prediction with malformed body."""
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422


# =============================================================================
# Run tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
