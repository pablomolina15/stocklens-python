from fastapi import APIRouter, HTTPException
from app.models import MLPredictionResponse, PredictionRequest
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/random-forest/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción Random Forest")
async def predict_rf(ticker: str, body: PredictionRequest = PredictionRequest()):
    ticker = ticker.upper().strip()
    if not 1 <= body.days_ahead <= 30:
        raise HTTPException(422, "days_ahead debe estar entre 1 y 30")
    try:
        from app.services.ml import predict_random_forest
        return predict_random_forest(ticker, body.days_ahead)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("RF error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])


@router.post("/gradient-boosting/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción Gradient Boosting")
async def predict_gb(ticker: str, body: PredictionRequest = PredictionRequest()):
    ticker = ticker.upper().strip()
    if not 1 <= body.days_ahead <= 30:
        raise HTTPException(422, "days_ahead debe estar entre 1 y 30")
    try:
        from app.services.ml import predict_gradient_boosting
        return predict_gradient_boosting(ticker, body.days_ahead)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("GB error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])


@router.post("/lstm/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción LSTM Neural Network")
async def predict_lstm(ticker: str, body: PredictionRequest = PredictionRequest()):
    ticker = ticker.upper().strip()
    if not 1 <= body.days_ahead <= 30:
        raise HTTPException(422, "days_ahead debe estar entre 1 y 30")
    try:
        from app.services.lstm import predict_lstm as _predict
        return _predict(ticker, body.days_ahead)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("LSTM error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])
