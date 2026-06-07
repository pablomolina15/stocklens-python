from fastapi import APIRouter, HTTPException
from app.services import fundamental as svc
from app.models import FundamentalResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/fundamental/{ticker}",
    response_model=FundamentalResponse,
    summary="Análisis fundamental + Value Score",
    description="Devuelve PER, PEG, ROE, márgenes, deuda, dividendo y checklist Value Investing.",
)
async def get_fundamental(ticker: str):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=422, detail="Ticker inválido")

    try:
        return svc.fetch_and_analyze(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Fundamental error for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)[:200]}")
