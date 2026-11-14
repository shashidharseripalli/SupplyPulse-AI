"""Demand forecast endpoints."""
from fastapi import APIRouter, HTTPException
from src.api.schemas import ForecastRequest, ForecastResponse
from src.common.logger import get_logger

router = APIRouter(prefix="/api", tags=["forecast"])
logger = get_logger(__name__)


@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Generate demand forecast for a product.
    
    Args:
        request: Forecast request with product_id and horizon.
        
    Raises:
        HTTPException: 501 Not Implemented until Module 1 is complete.
    """
    logger.info(f"Forecast requested for product: {request.product_id}")
    raise HTTPException(
        status_code=501,
        detail="Forecasting module not yet implemented. See Module 1.",
    )
