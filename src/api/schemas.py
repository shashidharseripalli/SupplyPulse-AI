"""Pydantic schemas for API request and response models."""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ForecastRequest(BaseModel):
    """Demand forecast request."""
    product_id: str = Field(..., description="Product identifier")
    horizon: int = Field(default=30, ge=1, le=365, description="Forecast horizon in days")


class ForecastResponse(BaseModel):
    """Demand forecast response."""
    product_id: str
    predictions: list[float]
    model_used: str


class RiskAssessmentRequest(BaseModel):
    """Risk assessment request."""
    supplier_id: str = Field(..., description="Supplier identifier")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response."""
    supplier_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="low, medium, high, or critical")
