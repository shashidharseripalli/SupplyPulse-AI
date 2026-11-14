"""Health check endpoint."""
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.common.config_loader import load_config

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health status."""
    config = load_config()
    return HealthResponse(
        status="healthy",
        version=config.get("project", {}).get("version", "0.1.0"),
    )
