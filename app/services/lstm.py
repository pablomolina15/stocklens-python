"""
LSTM Neural Network para predicción de precios.
TensorFlow/Keras — entrenamiento en tiempo real sobre datos de yfinance.

Arquitectura:
  Input  → (samples, timesteps=60, features=7)
  LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
  Dense(32) → Dense(1)
  Output → precio normalizado (se desnormaliza con MinMaxScaler)

Nota: En Railway free tier el primer entrenamiento tarda ~60-90s.
      Se recomienda Railway Starter plan ($5/mes) o Google Colab para GPU.
"""
import logging
import warnings
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Lazy import TensorFlow para no romper el startup si no está instalado
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
    tf.get_logger().setLevel('ERROR')
    logger.info("TensorFlow available: %s", tf.__version__)
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not installed — LSTM endpoint will return 503")

from app.models import MLPredictionResponse, PredictionPoint

TIMESTEPS = 60   # ventana de entrada: 60 días
FEATURES  = 7    # OHLCV + RSI + MACD


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame de features normalizadas."""
    df = df.copy()
    closes = df['Close']
    df.ta.rsi(length=14, close=closes, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, close=closes, append=True)

    # Seleccionar columnas relevantes
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_14', 'MACD_12_26_9']
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_cols].copy()
    df = df.ffill().bfill().dropna()
    return df


def _build_sequences(data: np.ndarray, timesteps: int, target_col: int = 3):
    """
    Construye pares (X, y) para entrenamiento LSTM.
    X shape: (samples, timesteps, features)
    y shape: (samples,) — precio de cierre normalizado al día siguiente
    """
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i - timesteps:i])
        y.append(data[i, target_col])  # col 3 = Close
    return np.array(X), np.array(y)


def _build_model(timesteps: int, features: int) -> 'keras.Model':
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(64, return_sequences=True),
        Dropout(0.2),

        LSTM(32, return_sequences=False),
        Dropout(0.1),

        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='huber',            # más robusto a outliers que MSE
        metrics=['mae']
    )
    return model


def predict_lstm(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    """
    Descarga 3 años de datos, entrena LSTM y predice a N días.
    Returns MLPredictionResponse con bandas de confianza basadas en
    Monte Carlo Dropout (múltiples pases forward con dropout activo).
    """
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow no está instalado. Añade 'tensorflow-cpu==2.15.0' a requirements.txt "
            "y redespliega en Railway. Requiere al menos 1GB RAM."
        )

    ticker = ticker.upper()
    logger.info("LSTM prediction start: %s, %d days", ticker, days_ahead)

    # ── Datos ────────────────────────────────────────────────────────────────
    stock = yf.Ticker(ticker)
    df_raw = stock.history(period='3y', interval='1d', auto_adjust=True)

    if df_raw is None or len(df_raw) < TIMESTEPS + 50:
        raise ValueError(f"Insufficient data for {ticker} (need ≥ {TIMESTEPS + 50} sessions)")

    df_feat = _build_features(df_raw)
    feature_names = df_feat.columns.tolist()
    n_features = len(feature_names)

    # ── Normalización por feature ────────────────────────────────────────────
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df_feat.values)

    # ── Secuencias ───────────────────────────────────────────────────────────
    X, y = _build_sequences(scaled, TIMESTEPS, target_col=3)  # col 3 = Close

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    logger.info("Sequences: train=%d, test=%d, features=%d", len(X_train), len(X_test), n_features)

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = _build_model(TIMESTEPS, n_features)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=0,
        shuffle=False,   # crítico en series temporales
    )

    # ── Métricas ──────────────────────────────────────────────────────────────
    y_pred_test_norm = model.predict(X_test, verbose=0).flatten()

    # Desnormalizar para calcular MAPE real
    dummy = np.zeros((len(y_test), n_features))
    dummy[:, 3] = y_test
    y_test_real = scaler.inverse_transform(dummy)[:, 3]
    dummy[:, 3] = y_pred_test_norm
    y_pred_real = scaler.inverse_transform(dummy)[:, 3]

    mape = float(np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-9))) * 100)
    mae  = float(np.mean(np.abs(y_test_real - y_pred_real)))
    epochs_trained = len(history.history['loss'])

    logger.info("Training done: %d epochs, MAPE=%.2f%%, MAE=$%.2f", epochs_trained, mape, mae)

    # ── Predicción recursiva con Monte Carlo Dropout ──────────────────────────
    # Usamos los últimos TIMESTEPS días como seed
    current_seq = scaled[-TIMESTEPS:].copy()   # (60, features)
    last_close  = float(df_raw['Close'].iloc[-1])

    n_mc = 50   # número de muestras Monte Carlo
    all_predictions: list[list[float]] = [[] for _ in range(days_ahead)]

    for mc_run in range(n_mc):
        seq = current_seq.copy()
        for day in range(days_ahead):
            x_in = seq[-TIMESTEPS:].reshape(1, TIMESTEPS, n_features)
            # training=True mantiene dropout activo → incertidumbre epistémica
            pred_norm = float(model(x_in, training=True).numpy()[0, 0])

            # Construir nuevo vector de features (solo actualizamos Close)
            new_row = seq[-1].copy()
            new_row[3] = pred_norm   # Close normalizado
            seq = np.vstack([seq, new_row])

            all_predictions[day].append(pred_norm)

    # ── Convertir predicciones a precios reales ──────────────────────────────
    predictions: list[PredictionPoint] = []
    last_date = df_raw.index[-1]
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()

    business_day = 0
    current_date = last_date

    for day_idx in range(days_ahead):
        # Avanzar al siguiente día hábil
        current_date = current_date + timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date = current_date + timedelta(days=1)
        business_day += 1

        mc_preds_norm = np.array(all_predictions[day_idx])

        # Desnormalizar distribución MC
        dummy_mc = np.zeros((len(mc_preds_norm), n_features))
        dummy_mc[:, 3] = mc_preds_norm
        mc_prices = scaler.inverse_transform(dummy_mc)[:, 3]

        mean_price  = float(np.mean(mc_prices))
        std_price   = float(np.std(mc_prices))
        lower       = float(np.percentile(mc_prices, 5))   # 90% CI
        upper       = float(np.percentile(mc_prices, 95))
        confidence  = float(max(0.0, min(1.0, 1.0 - (std_price / (abs(mean_price) + 1e-9)))))

        predictions.append(PredictionPoint(
            date=current_date.strftime('%Y-%m-%d'),
            predicted_price=round(mean_price, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            confidence=round(confidence, 4),
        ))

    # Limpiar modelo de memoria
    del model
    tf.keras.backend.clear_session()

    return MLPredictionResponse(
        ticker=ticker,
        model='lstm',
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance={f: round(1.0 / n_features, 4) for f in feature_names},
        accuracy_metrics={
            'mape':           round(mape, 2),
            'mae_usd':        round(mae, 2),
            'epochs_trained': epochs_trained,
            'train_samples':  len(X_train),
            'test_samples':   len(X_test),
            'mc_samples':     n_mc,
            'timesteps':      TIMESTEPS,
        },
        last_updated=datetime.now().isoformat(),
        disclaimer=(
            'Predicción experimental con LSTM + Monte Carlo Dropout. '
            'Intervalos de confianza al 90%. No constituye consejo de inversión.'
        ),
    )
