"""Risk assessment endpoints."""
from fastapi import APIRouter, HTTPException
from src.api.schemas import RiskAssessmentRequest, RiskAssessmentResponse
from src.common.logger import get_logger

router = APIRouter(prefix="/api", tags=["risk"])
logger = get_logger(__name__)


@router.post("/risk", response_model=RiskAssessmentResponse)
async def assess_risk(request: RiskAssessmentRequest) -> RiskAssessmentResponse:
    """Assess supply risk for a supplier.
    
    Args:
        request: Risk assessment request with supplier_id.
        
    Raises:
        HTTPException: 501 Not Implemented until Module 3 is complete.
    """
    logger.info(f"Risk assessment requested for supplier: {request.supplier_id}")
    raise HTTPException(
        status_code=501,
        detail="Risk prediction module not yet implemented. See Module 3.",
    )
