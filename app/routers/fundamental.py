from fastapi import APIRouter, HTTPException
from app.models import FundamentalResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/fundamental/{ticker}",
    response_model=FundamentalResponse,
    summary="Análisis fundamental + Value Score",
)
async def get_fundamental(ticker: str):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(422, "Ticker inválido")
    try:
        from app.services.fundamental import fetch_and_analyze
        return fetch_and_analyze(ticker)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("Fundamental error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])
