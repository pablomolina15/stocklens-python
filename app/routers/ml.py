from fastapi import APIRouter, HTTPException
from app.services import ml as svc_rf
from app.models import MLPredictionResponse, PredictionRequest
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/random-forest/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción Random Forest",
    description="RandomForestRegressor con 200 árboles. Features: SMA/EMA/RSI/MACD/BB/momentum/volatilidad.")
async def predict_rf(ticker: str, body: PredictionRequest = PredictionRequest()):
    ticker = ticker.upper().strip()
    if not 1 <= body.days_ahead <= 30:
        raise HTTPException(422, "days_ahead debe estar entre 1 y 30")
    try:
        return svc_rf.predict_random_forest(ticker, body.days_ahead)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("RF error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])


@router.post("/gradient-boosting/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción Gradient Boosting",
    description="GradientBoostingRegressor — más preciso que RF, algo más lento (~15s).")
async def predict_gb(ticker: str, body: PredictionRequest = PredictionRequest()):
    ticker = ticker.upper().strip()
    if not 1 <= body.days_ahead <= 30:
        raise HTTPException(422, "days_ahead debe estar entre 1 y 30")
    try:
        return svc_rf.predict_gradient_boosting(ticker, body.days_ahead)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error("GB error %s: %s", ticker, e, exc_info=True)
        raise HTTPException(500, str(e)[:300])


@router.post("/lstm/{ticker}", response_model=MLPredictionResponse,
    summary="Predicción LSTM Neural Network",
    description=(
        "Red LSTM bidireccional con Monte Carlo Dropout. "
        "Input: (samples, timesteps=60, features=7). "
        "Intervalos de confianza al 90% via 50 muestras MC. "
        "Requiere TensorFlow instalado — primer entrenamiento ~60-90s."
    ))
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
