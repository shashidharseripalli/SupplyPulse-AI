"""Pydantic schemas for API request and response models."""

from typing import Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class ForecastRequest(BaseModel):
    """Demand forecast request schema."""

    product_id: str = Field(..., description="Product or SKU identifier")
    horizon: int = Field(
        default=30, ge=1, le=365, description="Forecast horizon in days"
    )
    model_type: Literal["seasonal_naive", "xgboost", "lightgbm"] = Field(
        default="seasonal_naive",
        description="Forecasting model algorithm to invoke",
    )
    seasonal_period: int = Field(
        default=7, ge=1, description="Cycle length for seasonal naive baseline"
    )
    historical_demand: list[float] | None = Field(
        default=None,
        description="Optional historical demand values (defaults to sample historical series)",
    )


class ForecastResponse(BaseModel):
    """Demand forecast response schema."""

    product_id: str = Field(..., description="Product identifier")
    horizon: int = Field(..., description="Forecast horizon in days")
    model_used: str = Field(..., description="Algorithm used to generate predictions")
    predictions: list[float] = Field(..., description="Point forecasts for future steps")
    metrics: dict[str, float] | None = Field(
        default=None, description="In-sample or evaluation metrics if available"
    )


class RiskAssessmentRequest(BaseModel):
    """Risk assessment request."""

    supplier_id: str = Field(..., description="Supplier identifier")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response."""

    supplier_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="low, medium, high, or critical")
