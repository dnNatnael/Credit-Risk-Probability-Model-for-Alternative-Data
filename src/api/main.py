"""
FastAPI application for ML model serving.

This module provides a REST API for serving machine learning models
with MLflow integration for model loading and prediction.
"""

import os
import logging
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from .pydantic_models import (
        CustomerFeatures,
        RiskPrediction,
        HealthResponse,
        ErrorResponse,
    )
except ImportError:
    # Fallback for when running directly
    from pydantic_models import (
        CustomerFeatures,
        RiskPrediction,
        HealthResponse,
        ErrorResponse,
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
model = None
scaler = None
model_version = None

# FastAPI app initialization
app = FastAPI(
    title="Credit Risk Prediction API",
    description="ML-powered API for predicting customer credit risk using RFM analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model_from_mlflow(
    model_name: str = "credit_risk_random_forest", stage: str = "Production"
) -> tuple:
    """
    Load model and scaler from MLflow Model Registry.

    Parameters:
    -----------
    model_name : str
        Name of the registered model in MLflow
    stage : str
        Model stage (Production, Staging, etc.)

    Returns:
    --------
    tuple
        (model, scaler, version) or (None, None, None) if loading fails
    """
    try:
        # Set MLflow tracking URI if specified in environment
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        logger.info(f"Loading model {model_name} from stage {stage}")

        # Load model from MLflow Model Registry
        model_uri = f"models:/{model_name}/{stage}"
        loaded_model = mlflow.sklearn.load_model(model_uri)

        # Try to load scaler (if it exists)
        try:
            # Attempt to load scaler from the same run
            client = mlflow.tracking.MlflowClient()
            model_version_info = client.get_latest_versions(model_name, stages=[stage])[
                0
            ]
            run_id = model_version_info.run_id

            # Download scaler artifact if it exists
            scaler_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="scaler.pkl"
            )
            loaded_scaler = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")

        except Exception as e:
            logger.warning(f"Could not load scaler: {e}. Creating default scaler.")
            # Create a default scaler if not found
            loaded_scaler = StandardScaler()
            # Fit with dummy data (this should be replaced with proper scaler persistence)
            dummy_data = np.array([[30, 5, 100], [10, 10, 500], [60, 2, 50]])
            loaded_scaler.fit(dummy_data)

        version = (
            model_version_info.version
            if "model_version_info" in locals()
            else "unknown"
        )

        logger.info(f"Model loaded successfully. Version: {version}")
        return loaded_model, loaded_scaler, version

    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")
        return None, None, None


def load_model_fallback() -> tuple:
    """
    Fallback method to load a simple model if MLflow is not available.

    Returns:
    --------
    tuple
        (model, scaler, version)
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        logger.warning("Using fallback model - training a simple RandomForest")

        # Create and train a simple model with dummy data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        scaler = StandardScaler()

        # Dummy training data (RFM features)
        X_dummy = np.array(
            [
                [30, 5, 100],  # Low risk
                [10, 10, 500],  # Low risk
                [60, 2, 50],  # High risk
                [45, 3, 75],  # High risk
                [15, 8, 300],  # Low risk
                [50, 1, 25],  # High risk
            ]
        )
        y_dummy = np.array([0, 0, 1, 1, 0, 1])  # 0: low risk, 1: high risk

        # Fit scaler and model
        X_scaled = scaler.fit_transform(X_dummy)
        model.fit(X_scaled, y_dummy)

        logger.info("Fallback model created successfully")
        return model, scaler, "fallback-v1.0"

    except Exception as e:
        logger.error(f"Failed to create fallback model: {e}")
        return None, None, None


@app.on_event("startup")
async def startup_event():
    """
    Load the ML model when the application starts.
    """
    global model, scaler, model_version

    logger.info("Starting up the application...")

    # Try to load from MLflow first
    model, scaler, model_version = load_model_from_mlflow()

    # If MLflow loading fails, use fallback
    if model is None:
        logger.warning("MLflow model loading failed, using fallback model")
        model, scaler, model_version = load_model_fallback()

    if model is None:
        logger.error("Failed to load any model - application may not work correctly")
    else:
        logger.info(f"Application startup complete. Model version: {model_version}")


@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with basic API information.
    """
    return {
        "message": "Credit Risk Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status and model availability.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=RiskPrediction)
async def predict_risk(features: CustomerFeatures):
    """
    Predict credit risk based on customer RFM features.

    Parameters:
    -----------
    features : CustomerFeatures
        Customer features (Recency, Frequency, Monetary)

    Returns:
    --------
    RiskPrediction
        Risk probability and category

    Raises:
    -------
    HTTPException
        If model is not loaded or prediction fails
    """
    global model, scaler, model_version

    # Check if model is loaded
    if model is None or scaler is None:
        logger.error("Model or scaler not loaded")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please check service health.",
        )

    try:
        # Prepare input data
        input_data = pd.DataFrame(
            {
                "Recency": [features.recency],
                "Frequency": [features.frequency],
                "Monetary": [features.monetary],
            }
        )

        logger.info(f"Received prediction request: {input_data.iloc[0].to_dict()}")

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        risk_probability = model.predict_proba(input_scaled)[0][
            1
        ]  # Probability of class 1 (high risk)

        # Determine risk category based on threshold
        risk_threshold = 0.5  # This could be configurable
        risk_category = "high" if risk_probability >= risk_threshold else "low"

        logger.info(
            f"Prediction result: probability={risk_probability:.4f}, category={risk_category}"
        )

        return RiskPrediction(
            risk_probability=float(risk_probability),
            risk_category=risk_category,
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=list[RiskPrediction])
async def predict_risk_batch(features_list: list[CustomerFeatures]):
    """
    Predict credit risk for multiple customers in batch.

    Parameters:
    -----------
    features_list : list[CustomerFeatures]
        List of customer features

    Returns:
    --------
    list[RiskPrediction]
        List of risk predictions
    """
    global model, scaler, model_version

    # Check if model is loaded
    if model is None or scaler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Please check service health.",
        )

    try:
        # Prepare batch input data
        input_data = pd.DataFrame(
            [
                {
                    "Recency": features.recency,
                    "Frequency": features.frequency,
                    "Monetary": features.monetary,
                }
                for features in features_list
            ]
        )

        logger.info(
            f"Received batch prediction request for {len(features_list)} customers"
        )

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Make predictions
        risk_probabilities = model.predict_proba(input_scaled)[:, 1]

        # Prepare results
        results = []
        risk_threshold = 0.5

        for prob in risk_probabilities:
            risk_category = "high" if prob >= risk_threshold else "low"
            results.append(
                RiskPrediction(
                    risk_probability=float(prob),
                    risk_category=risk_category,
                    model_version=model_version,
                )
            )

        logger.info(f"Batch prediction completed for {len(results)} customers")
        return results

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error", detail=str(exc)).dict(),
    )


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
