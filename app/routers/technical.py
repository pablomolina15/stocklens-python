from fastapi import APIRouter, HTTPException, Query
from app.services import technical as svc
from app.models import TechnicalResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

VALID_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}


@router.get(
    "/technical/{ticker}",
    response_model=TechnicalResponse,
    summary="Análisis técnico completo",
    description="Devuelve OHLCV + SMA/EMA 50/200 + RSI + MACD + Bollinger Bands + señales detectadas.",
)
async def get_technical(
    ticker: str,
    period: str = Query(default="1y", description="Período: 3mo | 6mo | 1y | 2y | 5y"),
):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=422, detail="Ticker inválido")
    if period not in VALID_PERIODS:
        raise HTTPException(status_code=422, detail=f"Período inválido. Usa: {VALID_PERIODS}")

    try:
        return svc.fetch_and_analyze(ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Technical error for {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)[:200]}")
