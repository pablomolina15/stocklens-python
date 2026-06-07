"""
Servicio de Machine Learning — Random Forest + feature engineering.
Fase 2: predice precio a N días vista usando indicadores técnicos como features.

Features usadas:
  - SMA50, SMA200, EMA50, EMA200
  - RSI, MACD, MACD Signal, MACD Histogram
  - BB Upper/Middle/Lower, BB Width, %B
  - Price momentum 5d, 10d, 20d
  - Volume change 5d
  - Day of week, Month (estacionalidad básica)
"""
import logging
from typing import Optional
from datetime import datetime, timedelta

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

# ─── Feature engineering ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade todas las features técnicas al DataFrame.
    Requiere columnas: Open, High, Low, Close, Volume
    """
    closes = df["Close"]
    highs  = df["High"]
    lows   = df["Low"]
    vols   = df["Volume"]

    # Indicadores base
    df.ta.sma(length=50,  close=closes, append=True)
    df.ta.sma(length=200, close=closes, append=True)
    df.ta.ema(length=50,  close=closes, append=True)
    df.ta.ema(length=200, close=closes, append=True)
    df.ta.rsi(length=14,  close=closes, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, close=closes, append=True)
    df.ta.bbands(length=20, std=2, close=closes, append=True)
    df.ta.atr(length=14, high=highs, low=lows, close=closes, append=True)

    # Momentum: retornos a N días
    for n in [1, 5, 10, 20]:
        df[f"return_{n}d"] = closes.pct_change(n)

    # Volatilidad realizada
    df["volatility_20d"] = closes.pct_change().rolling(20).std()

    # Volume features
    df["volume_sma20"] = vols.rolling(20).mean()
    df["volume_ratio"] = vols / (df["volume_sma20"] + 1e-9)

    # Bollinger %B y ancho de banda
    bb_upper_col = "BBU_20_2.0"
    bb_lower_col = "BBL_20_2.0"
    bb_mid_col   = "BBM_20_2.0"
    if bb_upper_col in df.columns and bb_lower_col in df.columns:
        bb_width = df[bb_upper_col] - df[bb_lower_col]
        df["bb_width"]  = bb_width / (df[bb_mid_col] + 1e-9)
        df["bb_pct_b"]  = (closes - df[bb_lower_col]) / (bb_width + 1e-9)

    # Distancia precio / medias móviles (%)
    if "SMA_50"  in df.columns: df["dist_sma50"]  = (closes - df["SMA_50"])  / (df["SMA_50"]  + 1e-9)
    if "SMA_200" in df.columns: df["dist_sma200"] = (closes - df["SMA_200"]) / (df["SMA_200"] + 1e-9)

    # Estacionalidad simple
    df["day_of_week"] = pd.to_datetime(df.index).dayofweek if df.index.dtype != "object" else 0
    df["month"]       = pd.to_datetime(df.index).month     if df.index.dtype != "object" else 1

    return df


FEATURE_COLS = [
    "SMA_50", "SMA_200", "EMA_50", "EMA_200",
    "RSI_14",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0",
    "bb_width", "bb_pct_b",
    "ATRr_14",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_20d",
    "volume_ratio",
    "dist_sma50", "dist_sma200",
    "day_of_week", "month",
]


# ─── Entrenamiento y predicción ──────────────────────────────────────────────

def predict_random_forest(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info(f"RF prediction for {ticker}, {days_ahead} days ahead")

    # Descargar 3 años para tener suficientes datos de entrenamiento
    stock = yf.Ticker(ticker)
    df_raw = stock.history(period="3y", interval="1d", auto_adjust=True)

    if df_raw is None or len(df_raw) < 300:
        raise ValueError(f"Insufficient data for {ticker} (need ≥ 300 sessions)")

    df = df_raw.copy()
    df = build_features(df)

    # Target: retorno a N días (predecimos % cambio, luego convertimos a precio)
    df[f"target_{days_ahead}d"] = df["Close"].shift(-days_ahead) / df["Close"] - 1

    # Columnas disponibles
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    # Limpiar NaN
    df_clean = df[available_features + [f"target_{days_ahead}d", "Close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Not enough clean samples for {ticker}: {len(df_clean)}")

    X = df_clean[available_features].values
    y = df_clean[f"target_{days_ahead}d"].values
    closes_clean = df_clean["Close"].values

    # ── Train/test con TimeSeriesSplit ────────────────────────────────────────
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    # ── Métricas en test ─────────────────────────────────────────────────────
    y_pred_test = model.predict(X_test_s)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

    accuracy_metrics = {
        "mape": round(mape * 100, 2),        # en %
        "rmse_return": round(rmse * 100, 2),  # en % de retorno
        "test_samples": len(y_test),
        "train_samples": len(y_train),
    }

    # ── Feature importance ────────────────────────────────────────────────────
    importance = dict(zip(
        available_features,
        [round(float(v), 4) for v in model.feature_importances_]
    ))
    top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])

    # ── Generar predicciones futuras ─────────────────────────────────────────
    # Usamos el último vector de features conocido
    last_features = X[-1].reshape(1, -1)
    last_features_s = scaler.transform(last_features)
    last_close = float(df_raw["Close"].iloc[-1])

    # Estimación de incertidumbre: std de los árboles individuales
    tree_preds = np.array([tree.predict(last_features_s)[0] for tree in model.estimators_])
    pred_return_mean = float(model.predict(last_features_s)[0])
    pred_return_std  = float(tree_preds.std())

    predictions: list[PredictionPoint] = []
    last_date = df_raw.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()

    # Generamos un punto por día hábil hasta days_ahead
    business_day = 0
    current_date = last_date
    while business_day < days_ahead:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() >= 5:   # skip weekends
            continue
        business_day += 1

        # Interpolación lineal del retorno esperado hasta el horizonte
        frac = business_day / days_ahead
        day_return = pred_return_mean * frac
        day_std    = pred_return_std  * frac * 1.5   # ampliar incertidumbre con el tiempo

        predicted_price = round(last_close * (1 + day_return), 2)
        lower_bound     = round(last_close * (1 + day_return - 1.96 * day_std), 2)
        upper_bound     = round(last_close * (1 + day_return + 1.96 * day_std), 2)
        confidence      = round(max(0.0, min(1.0, 1.0 - abs(day_return) - day_std * 2)), 4)

        predictions.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=predicted_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            confidence=confidence,
        ))

    return MLPredictionResponse(
        ticker=ticker,
        model="random_forest",
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance=top_features,
        accuracy_metrics=accuracy_metrics,
        last_updated=datetime.now().isoformat(),
    )


def predict_gradient_boosting(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    """
    Alternativa con Gradient Boosting — misma interfaz, suele ser
    más preciso con menos datos pero más lento de entrenar.
    """
    ticker = ticker.upper()
    logger.info(f"GB prediction for {ticker}, {days_ahead} days ahead")

    stock = yf.Ticker(ticker)
    df_raw = stock.history(period="3y", interval="1d", auto_adjust=True)

    if df_raw is None or len(df_raw) < 300:
        raise ValueError(f"Insufficient data for {ticker}")

    df = df_raw.copy()
    df = build_features(df)
    df[f"target_{days_ahead}d"] = df["Close"].shift(-days_ahead) / df["Close"] - 1

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[available_features + [f"target_{days_ahead}d", "Close"]].dropna()

    X = df_clean[available_features].values
    y = df_clean[f"target_{days_ahead}d"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s  = scaler.transform(X[test_idx])

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_s, y[train_idx])

    y_pred_test = model.predict(X_test_s)
    mape = mean_absolute_percentage_error(y[test_idx], y_pred_test)

    last_features_s = scaler.transform(X[-1].reshape(1, -1))
    pred_return = float(model.predict(last_features_s)[0])
    last_close  = float(df_raw["Close"].iloc[-1])

    # Incertidumbre aproximada (staged predictions variance)
    staged = list(model.staged_predict(last_features_s))
    pred_std = float(np.std([p[0] for p in staged[-50:]])) if len(staged) >= 50 else abs(pred_return) * 0.5

    predictions: list[PredictionPoint] = []
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
        frac = business_day / days_ahead
        day_return = pred_return * frac
        day_std    = pred_std * frac * 1.5

        predictions.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=round(last_close * (1 + day_return), 2),
            lower_bound=round(last_close * (1 + day_return - 1.96 * day_std), 2),
            upper_bound=round(last_close * (1 + day_return + 1.96 * day_std), 2),
            confidence=round(max(0.0, min(1.0, 1.0 - abs(day_return) - day_std * 2)), 4),
        ))

    importance = dict(zip(
        available_features,
        [round(float(v), 4) for v in model.feature_importances_]
    ))

    return MLPredictionResponse(
        ticker=ticker,
        model="gradient_boosting",
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance=dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8]),
        accuracy_metrics={"mape": round(mape * 100, 2), "test_samples": len(y[test_idx])},
        last_updated=datetime.now().isoformat(),
    )
