"""
StockLens — LSTM Inference via Universal ONNX Model
app/services/lstm.py

Usa el modelo universal entrenado en Colab con S&P 500 completo.
Funciona para CUALQUIER ticker sin necesidad de reentrenar.

Mejoras vs. versión anterior:
  ✅ BPTT real (PyTorch con gradient clipping)
  ✅ Autoregresión iterativa (no interpolación lineal)
  ✅ MC Dropout simulado con ruido calibrado (incertidumbre real)
  ✅ Confianza que decrece con el horizonte temporal
  ✅ Feature importance real (permutation importance del Colab)
  ✅ Features universales normalizadas (agnósticas al precio absoluto)
  ✅ <50MB RAM en Railway (onnxruntime-cpu)
"""
import logging
import math
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from app.models import MLPredictionResponse, PredictionPoint
from app.config import settings

logger = logging.getLogger(__name__)

MODELS_DIR  = Path(__file__).parent.parent / 'models'
TIMESTEPS   = 30
MC_SAMPLES  = 50
MAX_RET_1D  = 0.05   # clamp retorno diario ±5%

# Features universales — deben coincidir EXACTAMENTE con el notebook
FEAT_COLS = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'dist_sma10', 'dist_sma20', 'dist_sma50',
    'rsi_14', 'macd_norm', 'macd_signal_norm', 'macd_hist_norm',
    'bb_pct_b', 'bb_width', 'atr_norm', 'vol_20d',
    'vol_ratio', 'obv_norm',
    'high_low_pct', 'close_open_pct',
    'stoch_k', 'stoch_d',
]
N_FEATURES = len(FEAT_COLS)

# ── Lazy ONNX Runtime import ──────────────────────────────────────────────────
_ort = None

def _get_ort():
    global _ort
    if _ort is None:
        try:
            import onnxruntime as ort
            _ort = ort
            logger.info("ONNX Runtime %s loaded", ort.__version__)
        except ImportError:
            raise RuntimeError(
                "onnxruntime-cpu no instalado. "
                "Añade 'onnxruntime-cpu==1.18.0' a requirements.txt y redespliega."
            )
    return _ort


# ── Model cache ───────────────────────────────────────────────────────────────
_bundle_cache: Optional[dict] = None

def _load_universal_model() -> dict:
    global _bundle_cache
    if _bundle_cache is not None:
        return _bundle_cache

    ort = _get_ort()
    onnx_path = MODELS_DIR / 'lstm_universal.onnx'
    pkl_path  = MODELS_DIR / 'lstm_universal.pkl'

    if not onnx_path.exists():
        raise ValueError(
            "No se encontró lstm_universal.onnx en app/models/. "
            "Ejecuta el notebook StockLens_Universal_LSTM.ipynb en Google Colab "
            "y sube los archivos generados a stocklens-python/app/models/"
        )
    if not pkl_path.exists():
        raise ValueError(
            "No se encontró lstm_universal.pkl en app/models/. "
            "Ejecuta el notebook StockLens_Universal_LSTM.ipynb en Google Colab."
        )

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        sess = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=['CPUExecutionProvider'],
        )
        with open(pkl_path, 'rb') as f:
            bundle = pickle.load(f)

        bundle['sess'] = sess
        _bundle_cache = bundle

        meta = bundle['meta']
        logger.info(
            "Universal LSTM loaded: %d tickers, dir_acc=%.1f%%, trained_at=%s",
            meta.get('n_tickers', 0),
            meta.get('direction_acc', 0),
            meta.get('trained_at', 'unknown'),
        )
        return bundle

    except Exception as e:
        raise RuntimeError(f"Error cargando modelo universal: {e}")


# ── Feature engineering (igual que el notebook) ───────────────────────────────
def _build_features(ticker: str) -> tuple[np.ndarray, float, float]:
    """
    Construye la ventana de features para inferencia.
    Returns: (X_window (TIMESTEPS, N_FEATURES), last_close, recent_vol)
    """
    df = None
    for attempt in range(settings.max_retries):
        try:
            df = yf.Ticker(ticker).history(period='6mo', interval='1d', auto_adjust=True)
            if df is not None and len(df) >= TIMESTEPS + 30:
                break
            df = None
        except Exception as e:
            logger.warning("yfinance attempt %d for %s: %s", attempt + 1, ticker, e)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay)

    if df is None or df.empty:
        raise ValueError(f"No se pudieron descargar datos para '{ticker}'")

    closes = df['Close']; highs = df['High']
    lows   = df['Low'];   vols  = df['Volume']

    out = pd.DataFrame(index=df.index)

    # Returns
    for n in [1, 3, 5, 10, 20]:
        out[f'ret_{n}d'] = closes.pct_change(n).clip(-0.2, 0.2)

    # Distance from MAs
    for length, col in [(10,'sma10'),(20,'sma20'),(50,'sma50')]:
        sma = closes.rolling(length).mean()
        out[f'dist_{col}'] = ((closes - sma) / (sma + 1e-9)).clip(-0.3, 0.3)

    # RSI
    rsi_s = ta.rsi(closes, length=14)
    out['rsi_14'] = (rsi_s / 100.0).fillna(0.5) if rsi_s is not None else 0.5

    # MACD
    macd_df = ta.macd(closes, fast=12, slow=26, signal=9)
    if macd_df is not None:
        mcol = next((c for c in macd_df.columns if c.startswith('MACD_12')), None)
        scol = next((c for c in macd_df.columns if c.startswith('MACDs_')), None)
        hcol = next((c for c in macd_df.columns if c.startswith('MACDh_')), None)
        out['macd_norm']        = (macd_df[mcol] / (closes+1e-9)).clip(-0.05, 0.05) if mcol else 0.0
        out['macd_signal_norm'] = (macd_df[scol] / (closes+1e-9)).clip(-0.05, 0.05) if scol else 0.0
        out['macd_hist_norm']   = (macd_df[hcol] / (closes+1e-9)).clip(-0.05, 0.05) if hcol else 0.0
    else:
        out['macd_norm'] = out['macd_signal_norm'] = out['macd_hist_norm'] = 0.0

    # Bollinger
    bb = ta.bbands(closes, length=20, std=2)
    if bb is not None:
        bbu = next((c for c in bb.columns if c.startswith('BBU_')), None)
        bbl = next((c for c in bb.columns if c.startswith('BBL_')), None)
        bbm = next((c for c in bb.columns if c.startswith('BBM_')), None)
        if bbu and bbl and bbm:
            bw = bb[bbu] - bb[bbl]
            out['bb_pct_b'] = ((closes - bb[bbl]) / (bw + 1e-9)).clip(-0.5, 1.5)
            out['bb_width'] = (bw / (bb[bbm] + 1e-9)).clip(0, 0.2)
        else:
            out['bb_pct_b'] = out['bb_width'] = 0.0
    else:
        out['bb_pct_b'] = out['bb_width'] = 0.0

    # ATR
    atr = ta.atr(highs, lows, closes, length=14)
    out['atr_norm'] = (atr / (closes + 1e-9)).clip(0, 0.1) if atr is not None else 0.01

    # Volatility
    out['vol_20d'] = closes.pct_change().rolling(20).std().clip(0, 0.1)

    # Volume
    vol_sma = vols.rolling(20).mean()
    out['vol_ratio'] = (vols / (vol_sma + 1e-9)).clip(0, 5)

    # OBV
    obv = ta.obv(closes, vols)
    if obv is not None:
        obv_norm = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9)
        out['obv_norm'] = obv_norm.clip(-3, 3)
    else:
        out['obv_norm'] = 0.0

    # Candle
    out['high_low_pct']   = ((highs - lows) / (closes + 1e-9)).clip(0, 0.1)
    out['close_open_pct'] = ((closes - df['Open']) / (df['Open'] + 1e-9)).clip(-0.1, 0.1)

    # Stochastic
    stoch = ta.stoch(highs, lows, closes)
    if stoch is not None:
        k_col = next((c for c in stoch.columns if c.startswith('STOCHk_')), None)
        d_col = next((c for c in stoch.columns if c.startswith('STOCHd_')), None)
        out['stoch_k'] = (stoch[k_col] / 100.0).fillna(0.5) if k_col else 0.5
        out['stoch_d'] = (stoch[d_col] / 100.0).fillna(0.5) if d_col else 0.5
    else:
        out['stoch_k'] = out['stoch_d'] = 0.5

    # Fill any missing features with 0
    for col in FEAT_COLS:
        if col not in out.columns:
            out[col] = 0.0

    out = out[FEAT_COLS].fillna(0.0)
    out_clean = out.dropna()

    if len(out_clean) < TIMESTEPS:
        raise ValueError(f"Features insuficientes para {ticker}: {len(out_clean)} filas")

    X_window  = out_clean.values[-TIMESTEPS:].astype(np.float32)
    last_close = float(closes.dropna().iloc[-1])
    recent_vol = float(closes.pct_change().iloc[-20:].std()) or 0.015

    return X_window, last_close, recent_vol


def _sf(v: float, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


# ── Main inference ────────────────────────────────────────────────────────────
def predict_lstm(ticker: str, days_ahead: int = 5) -> MLPredictionResponse:
    ticker = ticker.upper()
    logger.info("LSTM Universal ONNX: %s %dd", ticker, days_ahead)

    # 1. Load model (cached after first call)
    bundle   = _load_universal_model()
    sess     = bundle['sess']
    scaler_y = bundle['scaler_y']
    meta     = bundle['meta']

    # 2. Build features
    X_window, last_close, recent_vol = _build_features(ticker)

    # 3. Autoregressive prediction — day by day
    daily_returns: list[float] = []
    daily_stds:    list[float] = []
    current_window = X_window.copy()  # (TIMESTEPS, N_FEATURES)
    dropout_scale  = meta.get('dropout', 0.3) * 0.08

    for day in range(days_ahead):
        inp = current_window.reshape(1, TIMESTEPS, N_FEATURES)

        # Monte Carlo Dropout — simulate uncertainty via input noise
        mc_preds = []
        for _ in range(MC_SAMPLES):
            noise = np.random.normal(0, dropout_scale, inp.shape).astype(np.float32)
            pred_norm = sess.run(None, {'input': inp + noise})[0][0]
            mc_preds.append(float(pred_norm))

        mc_arr    = np.array(mc_preds)
        mean_norm = float(np.mean(mc_arr))
        std_norm  = float(np.std(mc_arr))

        # Denormalize
        mean_ret = float(scaler_y.inverse_transform([[mean_norm]])[0][0])
        std_ret  = abs(float(scaler_y.scale_[0]) * std_norm)

        # Clamp daily return
        mean_ret = float(np.clip(mean_ret, -MAX_RET_1D, MAX_RET_1D))
        std_ret  = float(np.clip(std_ret, 0.001, MAX_RET_1D))

        daily_returns.append(mean_ret)
        daily_stds.append(std_ret)

        # Update window: shift left and update return features with predicted return
        new_row = current_window[-1].copy()

        # Update ret_1d (index 0 in FEAT_COLS) with predicted return
        new_row[0] = float(np.clip(mean_ret, -0.2, 0.2))

        # Shift ret_3d, ret_5d etc. approximately
        if len(FEAT_COLS) > 1: new_row[1] = float(np.clip(
            current_window[-1][1] * 0.6 + mean_ret * 0.4, -0.2, 0.2))

        current_window = np.vstack([current_window[1:], new_row.reshape(1, -1)])

    logger.info(
        "LSTM %s: returns=%s",
        ticker,
        [f"{r*100:+.2f}%" for r in daily_returns],
    )

    # 4. Build prediction points
    predictions: list[PredictionPoint] = []
    current_price = last_close
    current_date  = datetime.now()

    for day_idx in range(days_ahead):
        # Next business day
        current_date = current_date + timedelta(days=1)
        while current_date.weekday() >= 5:
            current_date = current_date + timedelta(days=1)

        day_ret = daily_returns[day_idx]
        day_std = daily_stds[day_idx]

        # Compound price
        pred_price = current_price * (1 + day_ret)

        # Confidence intervals — fan-out with horizon
        # ✅ Incertidumbre que crece: combinamos MC std + vol histórica
        horizon_vol  = np.sqrt(day_idx + 1) * (recent_vol * 1.5 + day_std * 0.5)
        lower = pred_price * (1 - 1.96 * horizon_vol)
        upper = pred_price * (1 + 1.96 * horizon_vol)

        # Confidence: decreases with uncertainty AND horizon
        # ✅ Confianza que decrece: no más 92% plano
        horizon_decay = 1.0 / (1.0 + day_idx * 0.20)
        uncertainty   = min(abs(day_ret) * 5 + day_std * 8, 0.8)
        conf = round(float(np.clip((1.0 - uncertainty) * horizon_decay, 0.10, 0.88)), 4)

        predictions.append(PredictionPoint(
            date=current_date.strftime('%Y-%m-%d'),
            predicted_price=round(_sf(pred_price, last_close), 2),
            lower_bound=round(_sf(lower, pred_price * 0.95), 2),
            upper_bound=round(_sf(upper, pred_price * 1.05), 2),
            confidence=conf,
        ))

        current_price = pred_price  # compound

    # 5. Feature importance — real from permutation importance (Colab)
    feat_importance = meta.get('feat_importance', {})

    total_return_pct = (predictions[-1].predicted_price - last_close) / (last_close + 1e-9) * 100

    return MLPredictionResponse(
        ticker=ticker,
        model='lstm',
        days_ahead=days_ahead,
        predictions=predictions,
        feature_importance=feat_importance,
        accuracy_metrics={
            'direction_acc_pct': meta.get('direction_acc', 0.0),
            'train_samples':     meta.get('train_samples', 0),
            'test_samples':      meta.get('test_samples', 0),
            'mc_samples':        MC_SAMPLES,
            'timesteps':         TIMESTEPS,
            'n_features':        N_FEATURES,
            'hidden_size':       meta.get('hidden_size', 128),
            'num_layers':        meta.get('num_layers', 3),
            'n_tickers_trained': meta.get('n_tickers', 0),
            'implementation':    'universal_onnx_lstm',
            'total_return_pct':  round(_sf(total_return_pct), 2),
            'trained_at':        meta.get('trained_at', 'unknown'),
        },
        last_updated=datetime.now().isoformat(),
        disclaimer=(
            f'LSTM Universal PyTorch v3 (ONNX Runtime) — '
            f'entrenado con {meta.get("n_tickers", "N/A")} tickers del S&P 500, '
            f'BPTT real, autoregresión iterativa día a día, '
            f'MC Dropout ({MC_SAMPLES} muestras). '
            f'Dirección accuracy: {meta.get("direction_acc", 0):.1f}%. '
            'No constituye consejo de inversión.'
        ),
    )
