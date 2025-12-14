"""
Tests for the FastAPI application.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_predict_endpoint_valid_input():
    """Test prediction endpoint with valid input."""
    test_data = {"recency": 15.5, "frequency": 5.0, "monetary": 250.75}

    response = client.post("/predict", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert "risk_probability" in data
    assert "risk_category" in data
    assert "model_version" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["risk_category"] in ["low", "high"]


def test_predict_endpoint_invalid_input():
    """Test prediction endpoint with invalid input."""
    test_data = {
        "recency": -5.0,  # Invalid: negative value
        "frequency": 5.0,
        "monetary": 250.75,
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error


def test_predict_batch_endpoint():
    """Test batch prediction endpoint."""
    test_data = [
        {"recency": 15.5, "frequency": 5.0, "monetary": 250.75},
        {"recency": 30.0, "frequency": 2.0, "monetary": 100.0},
    ]

    response = client.post("/predict/batch", json=test_data)
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 2

    for prediction in data:
        assert "risk_probability" in prediction
        assert "risk_category" in prediction
        assert "model_version" in prediction
        assert 0.0 <= prediction["risk_probability"] <= 1.0
        assert prediction["risk_category"] in ["low", "high"]


def test_predict_missing_fields():
    """Test prediction with missing required fields."""
    test_data = {
        "recency": 15.5,
        # Missing frequency and monetary
    }

    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error


def test_openapi_docs():
    """Test that OpenAPI docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

    response = client.get("/redoc")
    assert response.status_code == 200
