"""
ML service: Random Forest + Gradient Boosting
yfinance 0.2.65 + pandas-ta 0.4.71b0 + scikit-learn 1.5.1
"""
import logging
import time
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

FEATURE_COLS = [
    "SMA_50", "SMA_200", "EMA_50", "EMA_200",
    "RSI_14",
    "MACD_12_26_9", "MACDs_12_26_9", "MACDh_12_26_9",
    "BBU_20_2.0", "BBM_20_2.0", "BBL_20_2.0",
    "bb_width", "bb_pct_b",
    "ATRr_14",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_20d", "volume_ratio",
    "dist_sma50", "dist_sma200",
]


def _download(ticker: str, period: str = "3y") -> pd.DataFrame:
    for attempt in range(settings.max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and len(df) >= 250:
                return df
        except Exception as e:
            logger.warning("Download attempt %d for %s: %s", attempt + 1, ticker, e)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay)
    raise ValueError(f"No se pudieron descargar datos suficientes para {ticker}. "
                     "Verifica que el ticker sea válido.")


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using Title Case column names from yfinance."""
    df = df.copy()

    closes = df["Close"]
    highs  = df["High"]
    lows   = df["Low"]
    vols   = df["Volume"]

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
        logger.warning("pandas-ta error: %s", e)

    for n in [1, 5, 10, 20]:
        df[f"return_{n}d"] = closes.pct_change(n)

    df["volatility_20d"] = closes.pct_change().rolling(20).std()
    df["volume_sma20"]   = vols.rolling(20).mean()
    df["volume_ratio"]   = vols / (df["volume_sma20"] + 1e-9)

    bb_u = "BBU_20_2.0"; bb_l = "BBL_20_2.0"; bb_m = "BBM_20_2.0"
    if bb_u in df.columns and bb_l in df.columns:
        bw = df[bb_u] - df[bb_l]
        df["bb_width"] = bw / (df[bb_m] + 1e-9)
        df["bb_pct_b"] = (closes - df[bb_l]) / (bw + 1e-9)

    if "SMA_50"  in df.columns: df["dist_sma50"]  = (closes - df["SMA_50"])  / (df["SMA_50"]  + 1e-9)
    if "SMA_200" in df.columns: df["dist_sma200"] = (closes - df["SMA_200"]) / (df["SMA_200"] + 1e-9)

    return df


def _build_prediction_points(df_raw, last_close, pred_return_mean, pred_return_std, days_ahead):
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
        frac    = business_day / days_ahead
        day_ret = pred_return_mean * frac
        day_std = pred_return_std  * frac * 1.5
        points.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=round(last_close * (1 + day_ret), 2),
            lower_bound=round(last_close * (1 + day_ret - 1.96 * day_std), 2),
            upper_bound=round(last_close * (1 + day_ret + 1.96 * day_std), 2),
            confidence=round(max(0.0, min(1.0, 1.0 - abs(day_ret) - day_std * 2)), 4),
        ))
    return points


def predict_random_forest(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("RF prediction: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df     = _build_features(df_raw)
    df["target"] = df["Close"].shift(-days_ahead) / df["Close"] - 1

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean  = df[available + ["target", "Close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}: {len(df_clean)} filas")

    X = df_clean[available].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, y[train_idx])

    y_pred = model.predict(Xte)
    mape   = float(mean_absolute_percentage_error(y[test_idx], y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y[test_idx], y_pred)))

    last_close      = float(df_raw["Close"].iloc[-1])
    last_s          = scaler.transform(X[-1].reshape(1, -1))
    tree_preds      = np.array([t.predict(last_s)[0] for t in model.estimators_])
    pred_mean       = float(model.predict(last_s)[0])
    pred_std        = float(tree_preds.std())

    importance = dict(sorted(
        zip(available, [round(float(v), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    return MLPredictionResponse(
        ticker=ticker, model="random-forest", days_ahead=days_ahead,
        predictions=_build_prediction_points(df_raw, last_close, pred_mean, pred_std, days_ahead),
        feature_importance=importance,
        accuracy_metrics={
            "mape":          round(mape * 100, 2),
            "rmse_return":   round(rmse * 100, 2),
            "train_samples": len(y[train_idx]),
            "test_samples":  len(y[test_idx]),
        },
        last_updated=datetime.now().isoformat(),
    )


def predict_gradient_boosting(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("GB prediction: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df     = _build_features(df_raw)
    df["target"] = df["Close"].shift(-days_ahead) / df["Close"] - 1

    available = [c for c in FEATURE_COLS if c in df.columns]
    df_clean  = df[available + ["target", "Close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}")

    X = df_clean[available].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[train_idx])
    Xte = scaler.transform(X[test_idx])

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )
    model.fit(Xtr, y[train_idx])

    y_pred    = model.predict(Xte)
    mape      = float(mean_absolute_percentage_error(y[test_idx], y_pred))
    last_close = float(df_raw["Close"].iloc[-1])
    last_s    = scaler.transform(X[-1].reshape(1, -1))
    pred_mean = float(model.predict(last_s)[0])

    staged    = list(model.staged_predict(last_s))
    pred_std  = float(np.std([p[0] for p in staged[-50:]])) if len(staged) >= 50 else abs(pred_mean) * 0.5

    importance = dict(sorted(
        zip(available, [round(float(v), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    return MLPredictionResponse(
        ticker=ticker, model="gradient-boosting", days_ahead=days_ahead,
        predictions=_build_prediction_points(df_raw, last_close, pred_mean, pred_std, days_ahead),
        feature_importance=importance,
        accuracy_metrics={"mape": round(mape * 100, 2), "test_samples": len(y[test_idx])},
        last_updated=datetime.now().isoformat(),
    )
