# app/routers/scanner.py
from fastapi import APIRouter, HTTPException, Query
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/opportunities",
    summary="Multi-signal opportunity scanner",
    description=(
        "3-layer scanner: Finviz volume discovery → SEC EDGAR 8-K + Yahoo RSS catalysts "
        "→ technical confirmation. Returns separate scores per category. Cached 15 min."
    ),
)
async def get_opportunities(
    max_results: int = Query(default=10, ge=1, le=25, description="Max results to return"),
):
    try:
        from app.services.scanner import run_opportunity_scan
        return await run_opportunity_scan(max_results=max_results)
    except Exception as e:
        logger.error("Scanner error: %s", e, exc_info=True)
        raise HTTPException(500, f"Scanner error: {str(e)[:300]}")
