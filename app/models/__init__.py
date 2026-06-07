from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


# ─── Technical ───────────────────────────────────────────────────────────────

class OHLCVPoint(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    ema50: Optional[float] = None
    ema200: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macdSignal: Optional[float] = None
    macdHistogram: Optional[float] = None
    bbUpper: Optional[float] = None
    bbMiddle: Optional[float] = None
    bbLower: Optional[float] = None


class TechnicalSignals(BaseModel):
    golden_cross: bool = False
    death_cross: bool = False
    rsi_overbought: bool = False
    rsi_oversold: bool = False
    rsi_value: Optional[float] = None
    macd_bullish: bool = False
    macd_bearish: bool = False
    price_above_bb_upper: bool = False
    price_below_bb_lower: bool = False
    trend: str = "neutral"


class TechnicalResponse(BaseModel):
    ticker: str
    period: str
    data: List[OHLCVPoint]
    signals: TechnicalSignals
    last_updated: str
    source: str = "live"


# ─── Fundamental ─────────────────────────────────────────────────────────────

class FundamentalMetrics(BaseModel):
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth: Optional[float] = None
    current_price: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    market_cap: Optional[float] = None


class ValueCriterion(BaseModel):
    name: str
    description: str
    passed: bool
    points_earned: int
    points_max: int
    detail: str


class ValueScore(BaseModel):
    score: int
    rating: str
    color: str
    criteria: List[ValueCriterion]
    summary: str


class FundamentalResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    industry: str
    metrics: FundamentalMetrics
    value_score: ValueScore
    last_updated: str
    source: str = "live"


# ─── ML ──────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    days_ahead: int = 5


class PredictionPoint(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float
    confidence: float


class MLPredictionResponse(BaseModel):
    ticker: str
    model: str
    days_ahead: int
    predictions: List[PredictionPoint]
    feature_importance: Optional[dict] = None
    accuracy_metrics: Optional[dict] = None
    last_updated: str
    disclaimer: str = "Predicción experimental. No constituye consejo de inversión."
