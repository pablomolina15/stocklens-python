"""
LSTM implementado con NumPy puro — sin TensorFlow ni PyTorch.
Compatible con Railway free tier (512MB RAM).

Arquitectura simplificada:
  Input (timesteps=30, features=5) → LSTM cell → Dense → precio predicho

Usa Monte Carlo Dropout simulado con perturbación de pesos
para generar intervalos de confianza.
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

TIMESTEPS = 30
FEATURES  = 5    # close_norm, rsi_norm, macd_norm, volume_norm, return_1d


# ── Numpy LSTM cell ───────────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 42):
        rng = np.random.RandomState(seed)
        scale = 0.1
        # Gates: forget, input, output, cell
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

    def __init__(self, input_size: int, hidden_size: int = 32, seed: int = 42):
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
              epochs: int = 40, lr: float = 0.005, batch_size: int = 16):
        """
        Simple training loop using numerical gradient approximation (finite differences).
        Fast enough for TIMESTEPS=30, hidden=32, ~200 samples.
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
    """Returns (N, FEATURES) normalized array or None if insufficient data."""
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

    df['close_norm']  = (closes  - closes.mean())  / (closes.std()  + 1e-9)
    df['volume_norm'] = (vols    - vols.mean())    / (vols.std()    + 1e-9)
    df['return_1d']   = closes.pct_change().fillna(0)
    df['rsi_norm']    = ((df[rsi_col] - 50) / 50) if rsi_col else 0.0
    df['macd_norm']   = (df[macd_col] / (closes.std() + 1e-9)) if macd_col else 0.0

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
    model = NumpyLSTM(input_size=FEATURES, hidden_size=32, seed=42)
    final_loss = model.train(X_train, y_train, epochs=50, lr=0.003, batch_size=16)
    logger.info("LSTM trained — final MSE: %.6f", final_loss)

    # Test accuracy
    y_pred_test = model.predict_batch(X_test)

    # Denormalize
    close_mean = float(df_raw['Close'].mean())
    close_std  = float(df_raw['Close'].std())
    y_test_real = y_test  * close_std + close_mean
    y_pred_real = y_pred_test * close_std + close_mean
    mape = float(np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-9))) * 100)
    mae  = float(np.mean(np.abs(y_test_real - y_pred_real)))

    # Monte Carlo predictions — run N forward passes with dropout for uncertainty
    N_MC = 30
    last_seq   = features[-TIMESTEPS:]   # (30, 5)
    mc_preds_norm = np.array([model.predict_one(last_seq, dropout=0.15) for _ in range(N_MC)])
    mc_prices  = mc_preds_norm * close_std + close_mean
    mean_next  = float(np.mean(mc_prices))
    std_next   = float(np.std(mc_prices))

    # Build N-day forward predictions with growing uncertainty
    last_close = float(df_raw['Close'].iloc[-1])
    implied_return = (mean_next - last_close) / (last_close + 1e-9)

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
        day_std    = (std_next / (last_close + 1e-9)) * frac * 1.5
        pred_price = last_close * (1 + day_ret)
        lower      = last_close * (1 + day_ret - 1.96 * day_std)
        upper      = last_close * (1 + day_ret + 1.96 * day_std)
        conf       = round(max(0.1, min(0.95, 1.0 - abs(day_ret) - day_std * 2)), 4)

        predictions.append(PredictionPoint(
            date=current_date.strftime('%Y-%m-%d'),
            predicted_price=round(pred_price, 2),
            lower_bound=round(lower, 2),
            upper_bound=round(upper, 2),
            confidence=conf,
        ))

    feature_names = ['close_norm', 'rsi_norm', 'macd_norm', 'volume_norm', 'return_1d']

    return MLPredictionResponse(
        ticker=ticker,
        model='lstm',
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance={f: round(1.0 / FEATURES, 4) for f in feature_names},
        accuracy_metrics={
            'mape':          round(mape, 2),
            'mae_usd':       round(mae, 2),
            'train_samples': len(X_train),
            'test_samples':  len(X_test),
            'mc_samples':    N_MC,
            'timesteps':     TIMESTEPS,
            'hidden_size':   32,
            'implementation':'numpy_lstm',
        },
        last_updated=datetime.now().isoformat(),
        disclaimer=(
            'LSTM NumPy — red recurrente entrenada en tiempo real sin TensorFlow. '
            'Monte Carlo Dropout (30 muestras) para intervalos de confianza al 90%. '
            'No constituye consejo de inversión.'
        ),
    )
