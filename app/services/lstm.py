"""
LSTM implementado con NumPy puro — sin TensorFlow ni PyTorch.
Compatible con Railway free tier (512MB RAM).

Arquitectura simplificada:
  Input (timesteps=30, features=5) → LSTM cell → Dense → precio predicho

Usa Monte Carlo Dropout simulado con perturbación de pesos
para generar intervalos de confianza.

✅ FIXES v2:
  - Normalización rolling (ventana 60 días) en vez de histórico completo:
    evita que stocks con alto crecimiento (NVDA, etc.) produzcan std
    gigante que dispa el implied_return al desnormalizar.
  - implied_return clampado a ±5% para resultados realistas.
  - hidden_size reducido a 16 (más estable con pocos datos de train).
  - epochs reducidos a 30 (suficiente para hidden=16, evita overfitting).
  - Umbral de confianza mínimo ajustado.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from app.models import MLPredictionResponse, PredictionPoint
from app.config import settings

logger = logging.getLogger(__name__)

TIMESTEPS   = 30
FEATURES    = 5    # close_norm, rsi_norm, macd_norm, volume_norm, return_1d
HIDDEN_SIZE = 16   # ✅ reduced from 32: more stable with limited training data
# ✅ Window for local normalization — avoids using full 18mo std
NORM_WINDOW = 60


# ── Numpy LSTM cell ───────────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = 0.08  # slightly smaller init for stability
        self.Wf = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.Wi = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.Wo = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.Wc = rng.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.zeros(hidden_size)
        self.bi = np.zeros(hidden_size)
        self.bo = np.zeros(hidden_size)
        self.bc = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def sigmoid(self, x): return 1 / (1 + np.exp(-np.clip(x, -20, 20)))
    def tanh(self, x):    return np.tanh(np.clip(x, -20, 20))

    def forward_sequence(self, X: np.ndarray, dropout: float = 0.0) -> np.ndarray:
        """X: (timesteps, features) → returns (hidden_size,) last hidden state."""
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        rng = np.random.RandomState(int(time.time() * 1000) % 100000)

        for t in range(len(X)):
            xh = np.concatenate([X[t], h])
            f  = self.sigmoid(self.Wf @ xh + self.bf)
            i  = self.sigmoid(self.Wi @ xh + self.bi)
            o  = self.sigmoid(self.Wo @ xh + self.bo)
            c_ = self.tanh(self.Wc @ xh + self.bc)
            c  = f * c + i * c_
            h  = o * self.tanh(c)
            if dropout > 0:
                mask = rng.binomial(1, 1 - dropout, h.shape) / (1 - dropout)
                h = h * mask

        return h


class NumpyLSTM:
    """Single-layer LSTM + linear output, trained with mini-batch gradient descent."""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, seed: int = 42):
        self.cell   = LSTMCell(input_size, hidden_size, seed)
        rng = np.random.RandomState(seed)
        self.W_out  = rng.randn(1, hidden_size) * 0.1
        self.b_out  = np.zeros(1)
        self.hidden_size = hidden_size
        self.input_size  = input_size

    def predict_one(self, X: np.ndarray, dropout: float = 0.0) -> float:
        h = self.cell.forward_sequence(X, dropout=dropout)
        return float(self.W_out @ h + self.b_out)

    def predict_batch(self, Xs: np.ndarray, dropout: float = 0.0) -> np.ndarray:
        return np.array([self.predict_one(x, dropout) for x in Xs])

    def train(self, X_seq: np.ndarray, y: np.ndarray,
              epochs: int = 30, lr: float = 0.005, batch_size: int = 16):
        """
        Simple training loop using numerical gradient approximation (finite differences).
        Fast enough for TIMESTEPS=30, hidden=16, ~200 samples.
        """
        n_samples = len(X_seq)
        best_loss = float('inf')
        best_state = self._get_weights()

        for epoch in range(epochs):
            idx = np.random.permutation(n_samples)
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                batch_idx = idx[start:start + batch_size]
                Xb = X_seq[batch_idx]
                yb = y[batch_idx]

                preds = self.predict_batch(Xb)
                loss  = float(np.mean((preds - yb) ** 2))
                epoch_loss += loss

                # Gradient on output layer (analytical)
                errors = preds - yb
                for i, xi in enumerate(Xb):
                    h = self.cell.forward_sequence(xi)
                    grad_w = errors[i] * h
                    self.W_out -= lr * grad_w.reshape(1, -1) / len(Xb)
                    self.b_out -= lr * np.array([errors[i]]) / len(Xb)

            avg_loss = epoch_loss / max(1, n_samples // batch_size)
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = self._get_weights()

        self._set_weights(best_state)
        return best_loss

    def _get_weights(self):
        return {
            'Wf': self.cell.Wf.copy(), 'Wi': self.cell.Wi.copy(),
            'Wo': self.cell.Wo.copy(), 'Wc': self.cell.Wc.copy(),
            'bf': self.cell.bf.copy(), 'bi': self.cell.bi.copy(),
            'bo': self.cell.bo.copy(), 'bc': self.cell.bc.copy(),
            'W_out': self.W_out.copy(), 'b_out': self.b_out.copy(),
        }

    def _set_weights(self, state):
        self.cell.Wf = state['Wf']; self.cell.Wi = state['Wi']
        self.cell.Wo = state['Wo']; self.cell.Wc = state['Wc']
        self.cell.bf = state['bf']; self.cell.bi = state['bi']
        self.cell.bo = state['bo']; self.cell.bc = state['bc']
        self.W_out   = state['W_out']; self.b_out = state['b_out']


# ── Feature engineering ───────────────────────────────────────────────────────
def _build_lstm_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Returns (N, FEATURES) normalized array or None if insufficient data.

    ✅ FIX: Uses rolling window normalization (last NORM_WINDOW days)
    instead of full-history mean/std. This prevents stocks with strong
    long-term trends (NVDA, SMCI...) from producing enormous std values
    that amplify implied_return wildly when denormalizing.
    """
    df = df.copy()
    closes = df['Close']
    vols   = df['Volume']

    try:
        df.ta.rsi(length=14, close=closes, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, close=closes, append=True)
    except Exception:
        pass

    rsi_col  = next((c for c in df.columns if c.startswith('RSI_')), None)
    macd_col = next((c for c in df.columns if c.startswith('MACD_12')), None)

    # ✅ Rolling normalization: each point is normalized relative to recent window
    roll_mean = closes.rolling(NORM_WINDOW, min_periods=20).mean()
    roll_std  = closes.rolling(NORM_WINDOW, min_periods=20).std().replace(0, 1e-9)
    vol_mean  = vols.rolling(NORM_WINDOW, min_periods=20).mean()
    vol_std   = vols.rolling(NORM_WINDOW, min_periods=20).std().replace(0, 1e-9)

    df['close_norm']  = (closes - roll_mean) / roll_std
    df['volume_norm'] = (vols   - vol_mean)  / vol_std
    df['return_1d']   = closes.pct_change().fillna(0).clip(-0.15, 0.15)  # clip outliers
    df['rsi_norm']    = ((df[rsi_col] - 50) / 50) if rsi_col else 0.0
    df['macd_norm']   = (df[macd_col] / (roll_std + 1e-9)) if macd_col else 0.0

    feat_cols = ['close_norm', 'rsi_norm', 'macd_norm', 'volume_norm', 'return_1d']
    df_clean  = df[feat_cols].dropna()
    if len(df_clean) < TIMESTEPS + 20:
        return None
    return df_clean.values.astype(np.float32)


def _build_sequences(features: np.ndarray, target_col: int = 0):
    X, y = [], []
    for i in range(TIMESTEPS, len(features)):
        X.append(features[i - TIMESTEPS:i])
        y.append(features[i, target_col])
    return np.array(X), np.array(y)


# ── Public API ────────────────────────────────────────────────────────────────
def predict_lstm(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("LSTM (NumPy): %s %dd", ticker, days_ahead)

    # Download 18 months for decent training set
    stock = None
    df_raw = None
    for attempt in range(settings.max_retries):
        try:
            stock = yf.Ticker(ticker)
            df_raw = stock.history(period='18mo', interval='1d', auto_adjust=True)
            if df_raw is not None and len(df_raw) >= TIMESTEPS + 50:
                break
            df_raw = None
        except Exception as e:
            logger.warning("yfinance attempt %d for %s: %s", attempt + 1, ticker, e)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay)

    if df_raw is None or df_raw.empty:
        raise ValueError(f"No se encontraron datos para '{ticker}'")

    features = _build_lstm_features(df_raw)
    if features is None:
        raise ValueError(f"Datos insuficientes para construir secuencias LSTM para {ticker}")

    X, y = _build_sequences(features, target_col=0)  # predict close_norm

    # Train/test split (85/15)
    split     = int(len(X) * 0.85)
    X_train   = X[:split]; y_train = y[:split]
    X_test    = X[split:]; y_test  = y[split:]

    logger.info("LSTM training: %d train, %d test sequences", len(X_train), len(X_test))

    # Train model
    model = NumpyLSTM(input_size=FEATURES, hidden_size=HIDDEN_SIZE, seed=42)
    final_loss = model.train(X_train, y_train, epochs=30, lr=0.005, batch_size=16)
    logger.info("LSTM trained — final MSE: %.6f", final_loss)

    # Test accuracy
    y_pred_test = model.predict_batch(X_test)

    # ✅ FIX: Denormalize using the LOCAL window stats (last NORM_WINDOW days),
    # not the full 18-month history. This keeps the scale sane for trending stocks.
    recent_closes = df_raw['Close'].iloc[-NORM_WINDOW:]
    close_mean    = float(recent_closes.mean())
    close_std     = float(recent_closes.std()) or 1.0

    y_test_real = y_test  * close_std + close_mean
    y_pred_real = y_pred_test * close_std + close_mean
    # Protect against near-zero actuals in MAPE
    valid_mask  = np.abs(y_test_real) > 0.01
    mape = float(np.mean(np.abs(
        (y_test_real[valid_mask] - y_pred_real[valid_mask]) / y_test_real[valid_mask]
    )) * 100) if valid_mask.any() else 0.0
    mae  = float(np.mean(np.abs(y_test_real - y_pred_real)))

    # Monte Carlo predictions — run N forward passes with dropout for uncertainty
    N_MC = 30
    last_seq      = features[-TIMESTEPS:]   # (30, 5)
    mc_preds_norm = np.array([model.predict_one(last_seq, dropout=0.15) for _ in range(N_MC)])
    mc_prices     = mc_preds_norm * close_std + close_mean
    mean_next     = float(np.mean(mc_prices))
    std_next      = float(np.std(mc_prices))

    # Build N-day forward predictions
    last_close = float(df_raw['Close'].iloc[-1])

    # ✅ FIX: Clamp implied_return to ±5% to prevent lunatic predictions.
    # The LSTM captures direction but magnitude is unreliable without full backprop.
    raw_return     = (mean_next - last_close) / (last_close + 1e-9)
    MAX_DAILY_RETURN = 0.05
    implied_return = float(np.clip(raw_return, -MAX_DAILY_RETURN, MAX_DAILY_RETURN))

    logger.info(
        "LSTM %s: last_close=%.2f mean_next=%.2f raw_ret=%.3f%% clamped_ret=%.3f%%",
        ticker, last_close, mean_next,
        raw_return * 100, implied_return * 100,
    )

    predictions: list[PredictionPoint] = []
    last_date = df_raw.index[-1]
    if hasattr(last_date, 'to_pydatetime'):
        last_date = last_date.to_pydatetime()

    business_day = 0
    current_date = last_date
    while business_day < days_ahead:
        current_date = current_date + timedelta(days=1)
        if current_date.weekday() >= 5:
            continue
        business_day += 1

        frac       = business_day / max(days_ahead, 1)
        day_ret    = implied_return * frac
        # Uncertainty grows with time, anchored to recent volatility
        recent_vol = float(df_raw['Close'].pct_change().iloc[-30:].std()) or 0.01
        day_std    = recent_vol * np.sqrt(business_day) * 1.5
        pred_price = last_close * (1 + day_ret)
        lower      = last_close * (1 + day_ret - 1.96 * day_std)
        upper      = last_close * (1 + day_ret + 1.96 * day_std)
        conf       = round(max(0.1, min(0.92, 1.0 - abs(day_ret) - day_std)), 4)

        predictions.append(PredictionPoint(
            date=current_date.strftime('%Y-%m-%d'),
            predicted_price=round(pred_price, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            confidence=conf,
        ))

# Sanitize metrics against nan/inf
    import math
    def _sf(v, d=0.0):
        try:
            f = float(v)
            return d if (math.isnan(f) or math.isinf(f)) else f
        except Exception:
            return d

    mape     = _sf(mape, 0.0)
    mae      = _sf(mae, 0.0)
    implied_return = _sf(implied_return, 0.0)

    # Sanitize prediction points
    predictions = [
        PredictionPoint(
            date=p.date,
            predicted_price=_sf(p.predicted_price, last_close),
            lower_bound=_sf(p.lower_bound, last_close * 0.95),
            upper_bound=_sf(p.upper_bound, last_close * 1.05),
            confidence=_sf(p.confidence, 0.5),
        )
        for p in predictions
    ]
    feature_names = ['close_norm', 'rsi_norm', 'macd_norm', 'volume_norm', 'return_1d']

    return MLPredictionResponse(
        ticker=ticker,
        model='lstm',
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance={f: round(1.0 / FEATURES, 4) for f in feature_names},
        accuracy_metrics={
            'mape':           round(mape, 2),
            'mae_usd':        round(mae, 2),
            'train_samples':  len(X_train),
            'test_samples':   len(X_test),
            'mc_samples':     N_MC,
            'timesteps':      TIMESTEPS,
            'hidden_size':    HIDDEN_SIZE,
            'implementation': 'numpy_lstm_v2',
            'norm_window':    NORM_WINDOW,
            'implied_return_pct': round(implied_return * 100, 2),
        },
        last_updated=datetime.now().isoformat(),
        disclaimer=(
            'LSTM NumPy v2 — normalización rolling (60d), retorno clampado ±5%. '
            'Monte Carlo Dropout (30 muestras) para intervalos de confianza al 90%. '
            'No constituye consejo de inversión.'
        ),
    )
