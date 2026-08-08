"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List


# =============================================================================
# TODO 1: PredictionRequest schema
# =============================================================================
class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""

    user_id: str = Field(
        ...,
        description="ID of the user",
        json_schema_extra={"example": "196"},
    )
    movie_id: str = Field(
        ...,
        description="ID of the movie",
        json_schema_extra={"example": "242"},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": "196", "movie_id": "242"}]
        }
    }


# =============================================================================
# TODO 2: PredictionResponse schema
# =============================================================================
class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""

    user_id: str = Field(..., description="ID of the user")
    movie_id: str = Field(..., description="ID of the movie")
    predicted_rating: float = Field(
        ...,
        ge=1.0,
        le=5.0,
        description="Predicted rating (1.0 - 5.0)",
        json_schema_extra={"example": 3.87},
    )
    model_version: str = Field(..., description="Version of the model used")

    model_config = {"protected_namespaces": ()}


# =============================================================================
# TODO 3: HealthResponse schema
# =============================================================================
class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    status: str = Field(
        ...,
        description='Health status: "healthy" or "unhealthy"',
        json_schema_extra={"example": "healthy"},
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the ML model is loaded",
        json_schema_extra={"example": True},
    )

    model_config = {"protected_namespaces": ()}


# =============================================================================
# BONUS: Batch prediction schemas
# =============================================================================
class PredictionItem(BaseModel):
    """Single prediction item for batch requests."""

    user_id: str
    movie_id: str


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction endpoint."""

    predictions: List[PredictionItem]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction endpoint."""

    predictions: List[PredictionResponse]
    total_count: int
