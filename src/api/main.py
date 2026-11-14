"""FastAPI application factory for SupplyPulse AI.

Creates and configures the FastAPI application with CORS,
routers, and lifecycle event handlers.
"""
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.logger import get_logger
from src.common.config_loader import load_config
from src.api.routes import health, forecast, risk

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events."""
    logger.info("SupplyPulse AI API starting up...")
    yield
    logger.info("SupplyPulse AI API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    config = load_config()
    
    app = FastAPI(
        title="SupplyPulse AI",
        description="Event-Aware Predictive Logistics and Supply Chain Resilience Platform",
        version=config.get("project", {}).get("version", "0.1.0"),
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api", {}).get("cors_origins", ["*"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router)
    app.include_router(forecast.router)
    app.include_router(risk.router)
    
    logger.info("FastAPI application created successfully.")
    return app


app = create_app()

if __name__ == "__main__":
    config = load_config()
    uvicorn.run(
        "src.api.main:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=config.get("api", {}).get("reload", True),
    )
