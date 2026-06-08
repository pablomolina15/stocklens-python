"""
Servicio de análisis técnico.
Compatible con yfinance 0.2.65 + pandas-ta 0.4.71b0 + pandas 2.2.x
"""
import logging
import time
from typing import Optional
from datetime import datetime

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

from app.models import OHLCVPoint, TechnicalSignals, TechnicalResponse
from app.config import settings

logger = logging.getLogger(__name__)

PERIOD_INTERVAL = {
    "3mo": "1d", "6mo": "1d", "1y": "1d", "2y": "1wk", "5y": "1wk",
}


def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int:
    try:
        v = int(val)
        return max(0, v)
    except (TypeError, ValueError):
        return 0


def fetch_and_analyze(ticker: str, period: str) -> TechnicalResponse:
    ticker = ticker.upper()
    interval = PERIOD_INTERVAL.get(period, "1d")

    df = None
    for attempt in range(settings.max_retries):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval, auto_adjust=True)
            if df is not None and not df.empty and len(df) > 10:
                break
            df = None
        except Exception as e:
            logger.warning("yfinance attempt %d/%d for %s: %s",
                           attempt + 1, settings.max_retries, ticker, e)
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay * (attempt + 1))

    if df is None or df.empty:
        raise ValueError(
            f"No se encontraron datos para '{ticker}'. "
            "Verifica que el ticker sea válido (ej: AAPL, MSFT, NVDA)."
        )

    # ── Reset index and normalise date column ─────────────────────────────
    df = df.reset_index()
    date_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["_date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["Close"]).copy()

    # Keep original column names for pandas-ta (it expects Title Case)
    # but add lowercase aliases for our OHLCV access
    df["_open"]   = df["Open"]
    df["_high"]   = df["High"]
    df["_low"]    = df["Low"]
    df["_close"]  = df["Close"]
    df["_volume"] = df["Volume"]

    # ── Calculate indicators ──────────────────────────────────────────────
    try:
        df.ta.sma(length=50,  close="Close", append=True)
        df.ta.sma(length=200, close="Close", append=True)
        df.ta.ema(length=50,  close="Close", append=True)
        df.ta.ema(length=200, close="Close", append=True)
        df.ta.rsi(length=14,  close="Close", append=True)
        df.ta.macd(fast=12, slow=26, signal=9, close="Close", append=True)
        df.ta.bbands(length=20, std=2, close="Close", append=True)
    except Exception as e:
        logger.warning("pandas-ta partial error for %s: %s", ticker, e)

    # ── Map generated column names ────────────────────────────────────────
    col_map = {
        "SMA_50": "sma50", "SMA_200": "sma200",
        "EMA_50": "ema50", "EMA_200": "ema200",
        "RSI_14": "rsi",
        "MACD_12_26_9": "macd",
        "MACDs_12_26_9": "macdSignal",
        "MACDh_12_26_9": "macdHistogram",
        "BBU_20_2.0": "bbUpper",
        "BBM_20_2.0": "bbMiddle",
        "BBL_20_2.0": "bbLower",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Build point list ──────────────────────────────────────────────────
    points: list[OHLCVPoint] = []
    for _, row in df.iterrows():
        points.append(OHLCVPoint(
            date=str(row["_date"]),
            open=_safe_float(row["_open"]) or 0,
            high=_safe_float(row["_high"]) or 0,
            low=_safe_float(row["_low"]) or 0,
            close=_safe_float(row["_close"]) or 0,
            volume=_safe_int(row["_volume"]),
            sma50=_safe_float(row.get("sma50")),
            sma200=_safe_float(row.get("sma200")),
            ema50=_safe_float(row.get("ema50")),
            ema200=_safe_float(row.get("ema200")),
            rsi=_safe_float(row.get("rsi")),
            macd=_safe_float(row.get("macd")),
            macdSignal=_safe_float(row.get("macdSignal")),
            macdHistogram=_safe_float(row.get("macdHistogram")),
            bbUpper=_safe_float(row.get("bbUpper")),
            bbMiddle=_safe_float(row.get("bbMiddle")),
            bbLower=_safe_float(row.get("bbLower")),
        ))

    signals = _detect_signals(points)

    return TechnicalResponse(
        ticker=ticker,
        period=period,
        data=points,
        signals=signals,
        last_updated=datetime.now().isoformat(),
        source="live",
    )


def _detect_signals(data: list[OHLCVPoint]) -> TechnicalSignals:
    s = TechnicalSignals()
    if len(data) < 2:
        return s
    last = data[-1]
    prev = data[-2]

    if all(v is not None for v in [last.sma50, last.sma200, prev.sma50, prev.sma200]):
        if prev.sma50 < prev.sma200 and last.sma50 > last.sma200:   # type: ignore
            s.golden_cross = True; s.trend = "bullish"
        elif prev.sma50 > prev.sma200 and last.sma50 < last.sma200: # type: ignore
            s.death_cross = True; s.trend = "bearish"
        else:
            s.trend = "bullish" if last.sma50 > last.sma200 else "bearish" # type: ignore

    if last.rsi is not None:
        s.rsi_value      = round(last.rsi, 2)
        s.rsi_overbought = last.rsi > 70
        s.rsi_oversold   = last.rsi < 30

    if all(v is not None for v in [last.macd, last.macdSignal, prev.macd, prev.macdSignal]):
        if prev.macd < prev.macdSignal and last.macd > last.macdSignal:  # type: ignore
            s.macd_bullish = True
        elif prev.macd > prev.macdSignal and last.macd < last.macdSignal: # type: ignore
            s.macd_bearish = True

    if last.bbUpper is not None:
        s.price_above_bb_upper = last.close > last.bbUpper
    if last.bbLower is not None:
        s.price_below_bb_lower = last.close < last.bbLower

    return s
