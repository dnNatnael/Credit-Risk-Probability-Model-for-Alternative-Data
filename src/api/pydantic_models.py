"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """
    Request model for customer features used in risk prediction.

    Based on RFM (Recency, Frequency, Monetary) analysis features.
    """

    recency: float = Field(
        ..., ge=0, description="Days since last purchase (Recency)", example=15.5
    )
    frequency: float = Field(
        ...,
        ge=0,
        description="Number of purchases in the period (Frequency)",
        example=5.0,
    )
    monetary: float = Field(
        ...,
        ge=0,
        description="Total monetary value of purchases (Monetary)",
        example=250.75,
    )

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {"recency": 15.5, "frequency": 5.0, "monetary": 250.75}
        }


class RiskPrediction(BaseModel):
    """
    Response model for risk prediction results.
    """

    risk_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of high risk (0.0 to 1.0)"
    )
    risk_category: str = Field(
        ..., description="Risk category based on probability threshold"
    )
    model_version: Optional[str] = Field(
        None, description="Version of the model used for prediction"
    )

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "risk_probability": 0.75,
                "risk_category": "high",
                "model_version": "v1.0",
            }
        }


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint.
    """

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether ML model is loaded")
    version: str = Field(..., description="API version")

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {"status": "healthy", "model_loaded": True, "version": "1.0.0"}
        }


class ErrorResponse(BaseModel):
    """
    Response model for error cases.
    """

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        """Pydantic configuration."""

        schema_extra = {
            "example": {
                "error": "Model prediction failed",
                "detail": "Invalid input format",
            }
        }
