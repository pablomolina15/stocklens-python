from fastapi import APIRouter, HTTPException, Query
from app.models import TechnicalResponse
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

VALID_PERIODS = {"3mo", "6mo", "1y", "2y", "5y"}


@router.get(
    "/technical/{ticker}",
    response_model=TechnicalResponse,
    summary="Análisis técnico completo",
    description="OHLCV + SMA/EMA 50/200 + RSI + MACD + Bollinger Bands + señales.",
)
async def get_technical(
    ticker: str,
    period: str = Query(default="1y", description="3mo | 6mo | 1y | 2y | 5y"),
):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(422, "Ticker inválido")
    if period not in VALID_PERIODS:
        raise HTTPException(422, f"Período inválido. Usa: {VALID_PERIODS}")
    try:
        # Lazy import — only loads pandas/sklearn when first called
        from app.services.technical import fetch_and_analyze
        return fetch_and_analyze(ticker, period)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("Technical error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])
