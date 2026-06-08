"""
ML service — Random Forest + Gradient Boosting
yfinance 0.2.65 + pandas-ta 0.4.71b0 + scikit-learn 1.5.1
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


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build technical features — compatible with yfinance 0.2.x output."""
    df = df.copy()

    # yfinance 0.2.x returns capitalised columns after reset_index
    # Normalise to lowercase
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df = df.set_index("date")

    closes = df["close"]
    highs  = df["high"]
    lows   = df["low"]
    vols   = df["volume"]

    try:
        df.ta.sma(length=50,  close=closes, append=True)
        df.ta.sma(length=200, close=closes, append=True)
        df.ta.ema(length=50,  close=closes, append=True)
        df.ta.ema(length=200, close=closes, append=True)
        df.ta.rsi(length=14,  close=closes, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, close=closes, append=True)
        df.ta.bbands(length=20, std=2, close=closes, append=True)
        df.ta.atr(length=14, high=highs, low=lows, close=closes, append=True)
    except Exception as e:
        logger.warning("pandas-ta error in feature build: %s", e)

    # Momentum
    for n in [1, 5, 10, 20]:
        df[f"return_{n}d"] = closes.pct_change(n)

    df["volatility_20d"] = closes.pct_change().rolling(20).std()
    df["volume_sma20"]   = vols.rolling(20).mean()
    df["volume_ratio"]   = vols / (df["volume_sma20"] + 1e-9)

    # Bollinger derived
    bb_u = "BBU_20_2.0"; bb_l = "BBL_20_2.0"; bb_m = "BBM_20_2.0"
    if bb_u in df.columns and bb_l in df.columns:
        bb_width = df[bb_u] - df[bb_l]
        df["bb_width"]  = bb_width / (df[bb_m] + 1e-9)
        df["bb_pct_b"]  = (closes - df[bb_l]) / (bb_width + 1e-9)

    if "SMA_50"  in df.columns: df["dist_sma50"]  = (closes - df["SMA_50"])  / (df["SMA_50"]  + 1e-9)
    if "SMA_200" in df.columns: df["dist_sma200"] = (closes - df["SMA_200"]) / (df["SMA_200"] + 1e-9)

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
]


def _download(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Download with retries, return raw DataFrame."""
    for attempt in range(settings.max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and len(df) >= 300:
                return df
        except Exception as e:
            logger.warning("Download attempt %d for %s: %s", attempt + 1, ticker, e)
            import time; time.sleep(1)
    raise ValueError(f"No se pudieron descargar datos suficientes para {ticker}")


def predict_random_forest(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("RF prediction: %s, %d days", ticker, days_ahead)

    df_raw  = _download(ticker)
    df      = _build_features(df_raw.reset_index())
    df["target"] = df["close"].shift(-days_ahead) / df["close"] - 1

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[available + ["target", "close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}: {len(df_clean)} filas limpias")

    X = df_clean[available].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s  = scaler.transform(X[test_idx])

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    model.fit(X_train_s, y[train_idx])

    y_pred = model.predict(X_test_s)
    mape = mean_absolute_percentage_error(y[test_idx], y_pred)
    rmse = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))

    importance = dict(sorted(
        zip(available, [round(float(v), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    last_close = float(df_raw["Close"].iloc[-1])
    last_features_s = scaler.transform(X[-1].reshape(1, -1))

    tree_preds = np.array([t.predict(last_features_s)[0] for t in model.estimators_])
    pred_return_mean = float(model.predict(last_features_s)[0])
    pred_return_std  = float(tree_preds.std())

    predictions = _build_prediction_points(
        df_raw, last_close, pred_return_mean, pred_return_std, days_ahead
    )

    return MLPredictionResponse(
        ticker=ticker, model="random-forest", days_ahead=days_ahead,
        predictions=predictions, feature_importance=importance,
        accuracy_metrics={
            "mape": round(mape * 100, 2),
            "rmse_return": round(rmse * 100, 2),
            "test_samples": len(y[test_idx]),
            "train_samples": len(y[train_idx]),
        },
        last_updated=datetime.now().isoformat(),
    )


def predict_gradient_boosting(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("GB prediction: %s, %d days", ticker, days_ahead)

    df_raw  = _download(ticker)
    df      = _build_features(df_raw.reset_index())
    df["target"] = df["close"].shift(-days_ahead) / df["close"] - 1

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[available + ["target", "close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}")

    X = df_clean[available].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X[train_idx])
    X_test_s  = scaler.transform(X[test_idx])

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )
    model.fit(X_train_s, y[train_idx])

    y_pred = model.predict(X_test_s)
    mape = mean_absolute_percentage_error(y[test_idx], y_pred)

    last_close = float(df_raw["Close"].iloc[-1])
    last_features_s = scaler.transform(X[-1].reshape(1, -1))
    pred_return = float(model.predict(last_features_s)[0])

    staged = list(model.staged_predict(last_features_s))
    pred_std = float(np.std([p[0] for p in staged[-50:]])) if len(staged) >= 50 else abs(pred_return) * 0.5

    importance = dict(sorted(
        zip(available, [round(float(v), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    predictions = _build_prediction_points(
        df_raw, last_close, pred_return, pred_std, days_ahead
    )

    return MLPredictionResponse(
        ticker=ticker, model="gradient-boosting", days_ahead=days_ahead,
        predictions=predictions, feature_importance=importance,
        accuracy_metrics={"mape": round(mape * 100, 2), "test_samples": len(y[test_idx])},
        last_updated=datetime.now().isoformat(),
    )


def _build_prediction_points(
    df_raw: pd.DataFrame,
    last_close: float,
    pred_return_mean: float,
    pred_return_std: float,
    days_ahead: int,
) -> list[PredictionPoint]:
    points = []
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

        frac      = business_day / days_ahead
        day_ret   = pred_return_mean * frac
        day_std   = pred_return_std  * frac * 1.5

        price  = round(last_close * (1 + day_ret), 2)
        lower  = round(last_close * (1 + day_ret - 1.96 * day_std), 2)
        upper  = round(last_close * (1 + day_ret + 1.96 * day_std), 2)
        conf   = round(max(0.0, min(1.0, 1.0 - abs(day_ret) - day_std * 2)), 4)

        points.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=price, lower_bound=lower,
            upper_bound=upper, confidence=conf,
        ))
    return points
