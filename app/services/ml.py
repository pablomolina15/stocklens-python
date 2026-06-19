"""
ML: Random Forest + Gradient Boosting
pandas>=2.3.2 + pandas-ta 0.4.71b0 + scikit-learn 1.5.1
"""
import logging
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


def _find_col(df: pd.DataFrame, *prefixes: str) -> Optional[str]:
    for prefix in prefixes:
        match = next((c for c in df.columns if c.startswith(prefix)), None)
        if match:
            return match
    return None


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
    raise ValueError(
        f"No se pudieron descargar datos para {ticker}. "
        "Verifica que el ticker sea válido."
    )


def _build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build technical features. Returns (df_with_features, feature_col_names).
    Auto-detects Bollinger band column names for pandas-ta version compat.
    """
    df = df.copy()
    closes = df["Close"]; highs = df["High"]
    lows   = df["Low"];   vols  = df["Volume"]

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

    # Auto-detect Bollinger column names (version-agnostic)
    bb_u = _find_col(df, "BBU_20"); bb_l = _find_col(df, "BBL_20"); bb_m = _find_col(df, "BBM_20")
    if bb_u and bb_l and bb_m:
        bw = df[bb_u] - df[bb_l]
        df["bb_width"] = bw / (df[bb_m] + 1e-9)
        df["bb_pct_b"] = (closes - df[bb_l]) / (bw + 1e-9)

    sma50  = _find_col(df, "SMA_50");  sma200 = _find_col(df, "SMA_200")
    if sma50:  df["dist_sma50"]  = (closes - df[sma50])  / (df[sma50]  + 1e-9)
    if sma200: df["dist_sma200"] = (closes - df[sma200]) / (df[sma200] + 1e-9)

    # Build feature list from what's actually available
    base_features = [
        sma50, sma200,
        _find_col(df, "EMA_50"), _find_col(df, "EMA_200"),
        _find_col(df, "RSI_14"),
        _find_col(df, "MACD_12_26_9"), _find_col(df, "MACDs_12_26_9"), _find_col(df, "MACDh_12_26_9"),
        bb_u, bb_m, bb_l,
        "bb_width" if "bb_width" in df.columns else None,
        "bb_pct_b" if "bb_pct_b" in df.columns else None,
        _find_col(df, "ATRr_14"),
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_20d", "volume_ratio",
        "dist_sma50" if "dist_sma50" in df.columns else None,
        "dist_sma200" if "dist_sma200" in df.columns else None,
    ]
    feature_cols = [f for f in base_features if f is not None and f in df.columns]
    return df, feature_cols


def _build_prediction_points(df_raw, last_close, mean_ret, std_ret, days_ahead):
    points = []
    last_date = df_raw.index[-1]
    if hasattr(last_date, "to_pydatetime"):
        last_date = last_date.to_pydatetime()
    business_day = 0; current_date = last_date
    while business_day < days_ahead:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() >= 5: continue
        business_day += 1
        frac = business_day / days_ahead
        dr = mean_ret * frac; ds = std_ret * frac * 1.5
        points.append(PredictionPoint(
            date=current_date.strftime("%Y-%m-%d"),
            predicted_price=round(last_close * (1 + dr), 2),
            lower_bound=round(last_close * (1 + dr - 1.96 * ds), 2),
            upper_bound=round(last_close * (1 + dr + 1.96 * ds), 2),
            confidence=round(max(0.0, min(1.0, 1.0 - abs(dr) - ds * 2)), 4),
        ))
    return points


def predict_random_forest(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("RF: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df, feature_cols = _build_features(df_raw)
    df["target"] = df["Close"].shift(-days_ahead) / df["Close"] - 1
    df_clean = df[feature_cols + ["target", "Close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}: {len(df_clean)} filas")

    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    tr, te = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])

    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, y[tr])

    y_pred = model.predict(Xte)
    try:
        mape = float(mean_absolute_percentage_error(y[te], y_pred))
    except Exception:
        mape = 0.0
    mape = float(np.nan_to_num(mape, nan=0.0, posinf=99.0, neginf=0.0))
    rmse = float(np.nan_to_num(np.sqrt(mean_squared_error(y[te], y_pred)), nan=0.0, posinf=99.0, neginf=0.0))

    last_row = X[-1].copy()
    last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)
    last_s     = scaler.transform(last_row.reshape(1, -1))
    tree_preds = np.array([t.predict(last_s)[0] for t in model.estimators_])
    pred_mean  = float(np.nan_to_num(model.predict(last_s)[0], nan=0.0))
    pred_std   = float(np.nan_to_num(tree_preds.std(), nan=0.01))
    last_close = float(df_raw["Close"].iloc[-1])

    importance = dict(sorted(
        zip(feature_cols, [round(float(np.nan_to_num(v, nan=0.0)), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    return MLPredictionResponse(
        ticker=ticker, model="random-forest", days_ahead=days_ahead,
        predictions=_build_prediction_points(df_raw, last_close, pred_mean, pred_std, days_ahead),
        feature_importance=importance,
        accuracy_metrics={
            "mape": round(mape * 100, 2), "rmse_return": round(rmse * 100, 2),
            "train_samples": len(y[tr]), "test_samples": len(y[te]),
        },
        last_updated=datetime.now().isoformat(),
    )


def predict_gradient_boosting(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("GB: %s %dd", ticker, days_ahead)

    df_raw = _download(ticker)
    df, feature_cols = _build_features(df_raw)
    df["target"] = df["Close"].shift(-days_ahead) / df["Close"] - 1
    df_clean = df[feature_cols + ["target", "Close"]].dropna()

    if len(df_clean) < 100:
        raise ValueError(f"Datos insuficientes para {ticker}")

    X = df_clean[feature_cols].values
    y = df_clean["target"].values

    tscv = TimeSeriesSplit(n_splits=5)
    tr, te = list(tscv.split(X))[-1]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])

    model = GradientBoostingRegressor(
        n_estimators=150, learning_rate=0.05,
        max_depth=4, subsample=0.8, random_state=42,
    )
    model.fit(Xtr, y[tr])

    y_pred = model.predict(Xte)
    try:
        mape = float(mean_absolute_percentage_error(y[te], y_pred))
    except Exception:
        mape = 0.0
    mape = float(np.nan_to_num(mape, nan=0.0, posinf=99.0, neginf=0.0))
    last_row = X[-1].copy()
    last_row = np.nan_to_num(last_row, nan=0.0, posinf=0.0, neginf=0.0)
    last_s     = scaler.transform(last_row.reshape(1, -1))
    pred_mean  = float(np.nan_to_num(model.predict(last_s)[0], nan=0.0))
    staged     = list(model.staged_predict(last_s))
    pred_std   = float(np.nan_to_num(np.std([p[0] for p in staged[-50:]]), nan=0.01)) if len(staged) >= 50 else abs(pred_mean) * 0.5
    last_close = float(df_raw["Close"].iloc[-1])

   importance = dict(sorted(
        zip(feature_cols, [round(float(np.nan_to_num(v, nan=0.0)), 4) for v in model.feature_importances_]),
        key=lambda x: x[1], reverse=True
    )[:8])

    return MLPredictionResponse(
        ticker=ticker, model="gradient-boosting", days_ahead=days_ahead,
        predictions=_build_prediction_points(df_raw, last_close, pred_mean, pred_std, days_ahead),
        feature_importance=importance,
        accuracy_metrics={"mape": round(mape * 100, 2), "test_samples": len(y[te])},
        last_updated=datetime.now().isoformat(),
    )
