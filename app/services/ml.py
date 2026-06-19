"""
ML: Random Forest + Gradient Boosting
pandas>=2.3.2 + pandas-ta 0.4.71b0 + scikit-learn 1.5.1

✅ FIXES v4:
  - Ventana de entrenamiento: 6 meses (antes 3 años).
    Evita que crashes históricos de hace 2 años contaminen la predicción.
  - Target normalizado por volatilidad reciente (60d):
    En vez de predecir retorno bruto, predice retorno en sigmas.
    Al desnormalizar, el resultado es proporcional al riesgo actual
    de la acción — no al crash que tuvo hace 2 años.
  - Clamp final a ±3% diario / ±15% absoluto como última red de seguridad.
  - _sf() sanitiza todos los floats antes del JSON.
"""
import logging
import math
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from app.models import MLPredictionResponse, PredictionPoint
from app.config import settings

logger = logging.getLogger(__name__)

# Training window — 6 months is enough to capture current regime
# without overfitting to historical crashes from 2+ years ago
TRAIN_PERIOD   = "6mo"
MIN_ROWS       = 80       # minimum rows after feature engineering
NORM_WINDOW    = 60       # days for volatility normalization of target

# Safety clamp — last line of defense against absurd predictions
MAX_DAILY_RET  = 0.02   # 2% max per day implied     # 3% max per day implied
MAX_ABS_RET    = 0.10   # 10% absolute ceiling     # 15% absolute ceiling


def _sf(v, default: float = 0.0) -> float:
    """Convierte cualquier float a valor JSON-safe (sin nan/inf)."""
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def _clamp(ret: float, days_ahead: int) -> float:
    """Limita el retorno predicho a un rango plausible."""
    cap = min(MAX_ABS_RET, MAX_DAILY_RET * days_ahead)
    return float(np.clip(ret, -cap, cap))


def _find_col(df: pd.DataFrame, *prefixes: str) -> Optional[str]:
    for p in prefixes:
        m = next((c for c in df.columns if c.startswith(p)), None)
        if m:
            return m
    return None


def _download(ticker: str) -> pd.DataFrame:
    for attempt in range(settings.max_retries):
        try:
            df = yf.Ticker(ticker).history(
                period=TRAIN_PERIOD, interval="1d", auto_adjust=True
            )
            if df is not None and len(df) >= MIN_ROWS:
                return df
        except Exception as e:
            logger.warning("Download attempt %d for %s: %s", attempt + 1, ticker, e)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay)
    raise ValueError(
        f"No se pudieron descargar datos para '{ticker}'. "
        "Verifica que el ticker sea válido y tenga historial suficiente."
    )


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    closes = df["Close"]; highs = df["High"]
    lows   = df["Low"];   vols  = df["Volume"]

    try:
        df.ta.sma(length=20,  close=closes, append=True)
        df.ta.sma(length=50,  close=closes, append=True)
        df.ta.ema(length=20,  close=closes, append=True)
        df.ta.ema(length=50,  close=closes, append=True)
        df.ta.rsi(length=14,  close=closes, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, close=closes, append=True)
        df.ta.bbands(length=20, std=2, close=closes, append=True)
        df.ta.atr(length=14, high=highs, low=lows, close=closes, append=True)
    except Exception as e:
        logger.warning("pandas-ta error: %s", e)

    # Short-term returns (most predictive over 6mo window)
    for n in [1, 3, 5, 10]:
        df[f"return_{n}d"] = closes.pct_change(n)

    df["volatility_10d"] = closes.pct_change().rolling(10).std()
    df["volatility_20d"] = closes.pct_change().rolling(20).std()
    df["volume_sma20"]   = vols.rolling(20).mean()
    df["volume_ratio"]   = vols / (df["volume_sma20"] + 1e-9)

    bb_u = _find_col(df, "BBU_20"); bb_l = _find_col(df, "BBL_20"); bb_m = _find_col(df, "BBM_20")
    if bb_u and bb_l and bb_m:
        bw = df[bb_u] - df[bb_l]
        df["bb_width"] = bw / (df[bb_m] + 1e-9)
        df["bb_pct_b"] = (closes - df[bb_l]) / (bw + 1e-9)

    sma20 = _find_col(df, "SMA_20"); sma50 = _find_col(df, "SMA_50")
    if sma20: df["dist_sma20"] = (closes - df[sma20]) / (df[sma20] + 1e-9)
    if sma50: df["dist_sma50"] = (closes - df[sma50]) / (df[sma50] + 1e-9)

    candidates = [
        sma20, sma50,
        _find_col(df, "EMA_20"), _find_col(df, "EMA_50"),
        _find_col(df, "RSI_14"),
        _find_col(df, "MACD_12_26_9"), _find_col(df, "MACDs_12_26_9"), _find_col(df, "MACDh_12_26_9"),
        bb_u, bb_m, bb_l,
        "bb_width" if "bb_width" in df.columns else None,
        "bb_pct_b" if "bb_pct_b" in df.columns else None,
        _find_col(df, "ATRr_14"),
        "return_1d", "return_3d", "return_5d", "return_10d",
        "volatility_10d", "volatility_20d", "volume_ratio",
        "dist_sma20" if "dist_sma20" in df.columns else None,
        "dist_sma50" if "dist_sma50" in df.columns else None,
    ]
    feature_cols = [f for f in candidates if f is not None and f in df.columns]
    return df, feature_cols


def _normalize_target(df: pd.DataFrame, days_ahead: int) -> tuple[pd.Series, float, float]:
    """
    Build a volatility-normalized target.

    Instead of predicting raw return (which is dominated by the stock's
    long-term trend), we predict return / recent_volatility (z-score units).
    This makes the model focus on whether the current setup is strong/weak
    relative to the stock's OWN recent behavior — not vs. a 2-year crash.

    Returns:
        target_series: normalized return series
        vol_mean:      rolling vol mean (for denormalization)
        vol_std:       rolling vol std  (for denormalization)
    """
    raw_return = df["Close"].shift(-days_ahead) / df["Close"] - 1

    # Rolling volatility over NORM_WINDOW days
    roll_vol = df["Close"].pct_change().rolling(NORM_WINDOW, min_periods=20).std()
    roll_vol = roll_vol.replace(0, np.nan).ffill().fillna(0.01)

    # Normalize: how many "typical daily moves" is this return?
    normalized = raw_return / (roll_vol * np.sqrt(days_ahead) + 1e-9)

    # Clip extreme training targets (e.g. earnings gaps) to ±3 sigma
    normalized = normalized.clip(-3.0, 3.0)

    # Stats for denormalization at prediction time
    recent_vol = float(roll_vol.iloc[-1])
    vol_mean   = recent_vol  # use current vol for denormalization

    return normalized, vol_mean, float(roll_vol.std() or 0.005)


def _denormalize_prediction(pred_norm: float, vol_mean: float, days_ahead: int) -> float:
    """Convert normalized prediction back to price return."""
    raw = pred_norm * vol_mean * np.sqrt(days_ahead)
    return _clamp(_sf(raw, 0.0), days_ahead)


def _build_prediction_points(
    df_raw: pd.DataFrame,
    last_close: float,
    pred_return: float,   # already denormalized and clamped
    pred_std_norm: float, # in normalized units
    vol_mean: float,
    days_ahead: int,
) -> list[PredictionPoint]:
    last_close    = _sf(last_close, 100.0)
    pred_return   = _sf(pred_return, 0.0)
    pred_std_norm = _sf(pred_std_norm, 0.1)
    vol_mean      = _sf(vol_mean, 0.01)

    # Convert std to price units for confidence intervals
    pred_std_price = pred_std_norm * vol_mean * np.sqrt(days_ahead)
    pred_std_price = float(np.clip(pred_std_price, 0.005, 0.12))

    points: list[PredictionPoint] = []
    last_date = df_raw.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    business_day = 0
    current_date = last_date
    while business_day < days_ahead:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
        business_day += 1

        frac     = business_day / days_ahead
        day_ret  = pred_return * frac
        day_std  = pred_std_price * (frac ** 0.5)

        pred  = round(_sf(last_close * (1 + day_ret),               last_close),       2)
        lower = round(_sf(last_close * (1 + day_ret - 1.96 * day_std), pred * 0.97),  2)
        upper = round(_sf(last_close * (1 + day_ret + 1.96 * day_std), pred * 1.03),  2)
        conf  = round(_sf(max(0.1, min(0.92, 1.0 - abs(day_ret) * 5 - day_std * 3)), 0.5), 4)

        points.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=pred,
            lower_bound=lower,
            upper_bound=upper,
            confidence=conf,
        ))
    return points


def predict_random_forest(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("RF: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df, feature_cols = _build_features(df_raw)

    # ✅ Volatility-normalized target
    target, vol_mean, vol_std = _normalize_target(df, days_ahead)
    df["target"] = target
    df_clean = df[feature_cols + ["target", "Close"]].dropna()

    if len(df_clean) < MIN_ROWS:
        raise ValueError(f"Datos insuficientes para {ticker}: {len(df_clean)} filas tras feature engineering")

    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=3)  # 3 splits for 6mo window (5 would be too small)
    tr, te = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, y[tr])

    y_pred = model.predict(Xte)
    try:
        mape = _sf(mean_absolute_percentage_error(y[te], y_pred), 0.0)
    except Exception:
        mape = 0.0
    rmse = _sf(np.sqrt(mean_squared_error(y[te], y_pred)), 0.0)

    # Predict on latest data point
    last_row      = np.nan_to_num(X[-1].copy(), nan=0.0, posinf=0.0, neginf=0.0)
    last_s        = scaler.transform(last_row.reshape(1, -1))
    tree_preds    = np.array([_sf(t.predict(last_s)[0]) for t in model.estimators_])
    pred_norm     = _sf(model.predict(last_s)[0], 0.0)
    pred_std_norm = _sf(tree_preds.std(), 0.1)

    # ✅ Denormalize using current volatility
    pred_return = _denormalize_prediction(pred_norm, vol_mean, days_ahead)
    last_close  = _sf(df_raw["Close"].iloc[-1], 100.0)

    importance = dict(sorted(
        zip(feature_cols, [round(_sf(v, 0.0), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    logger.info("RF %s: pred_norm=%.3f vol_mean=%.4f pred_return=%.2f%%",
                ticker, pred_norm, vol_mean, pred_return * 100)

    return MLPredictionResponse(
        ticker=ticker,
        model="random-forest",
        days_ahead=days_ahead,
        predictions=_build_prediction_points(
            df_raw, last_close, pred_return, pred_std_norm, vol_mean, days_ahead
        ),
        feature_importance=importance,
        accuracy_metrics={
            "mape":          round(mape * 100, 2),
            "rmse_norm":     round(rmse, 4),
            "train_samples": len(y[tr]),
            "test_samples":  len(y[te]),
            "train_period":  TRAIN_PERIOD,
            "vol_mean_pct":  round(vol_mean * 100, 3),
        },
        last_updated=datetime.now().isoformat(),
    )


def predict_gradient_boosting(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("GB: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df, feature_cols = _build_features(df_raw)

    # ✅ Volatility-normalized target
    target, vol_mean, vol_std = _normalize_target(df, days_ahead)
    df["target"] = target
    df_clean = df[feature_cols + ["target", "Close"]].dropna()

    if len(df_clean) < MIN_ROWS:
        raise ValueError(f"Datos insuficientes para {ticker}")

    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=3)
    tr, te = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        max_depth=3, subsample=0.8, random_state=42,
    )
    model.fit(Xtr, y[tr])

    y_pred = model.predict(Xte)
    try:
        mape = _sf(mean_absolute_percentage_error(y[te], y_pred), 0.0)
    except Exception:
        mape = 0.0

    last_row      = np.nan_to_num(X[-1].copy(), nan=0.0, posinf=0.0, neginf=0.0)
    last_s        = scaler.transform(last_row.reshape(1, -1))
    pred_norm     = _sf(model.predict(last_s)[0], 0.0)
    staged        = list(model.staged_predict(last_s))
    pred_std_norm = _sf(np.std([p[0] for p in staged[-50:]]), 0.1) if len(staged) >= 50 else 0.1

    # ✅ Denormalize using current volatility
    pred_return = _denormalize_prediction(pred_norm, vol_mean, days_ahead)
    last_close  = _sf(df_raw["Close"].iloc[-1], 100.0)

    importance = dict(sorted(
        zip(feature_cols, [round(_sf(v, 0.0), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    logger.info("GB %s: pred_norm=%.3f vol_mean=%.4f pred_return=%.2f%%",
                ticker, pred_norm, vol_mean, pred_return * 100)

    return MLPredictionResponse(
        ticker=ticker,
        model="gradient-boosting",
        days_ahead=days_ahead,
        predictions=_build_prediction_points(
            df_raw, last_close, pred_return, pred_std_norm, vol_mean, days_ahead
        ),
        feature_importance=importance,
        accuracy_metrics={
            "mape":         round(mape * 100, 2),
            "test_samples": len(y[te]),
            "train_period": TRAIN_PERIOD,
            "vol_mean_pct": round(vol_mean * 100, 3),
        },
        last_updated=datetime.now().isoformat(),
    )
