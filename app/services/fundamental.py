"""
Servicio de análisis fundamental.
Extrae métricas de yfinance .info, .financials, .balance_sheet
y calcula el Value Investing Score.
"""
import logging
import time
from typing import Optional
from datetime import datetime

import yfinance as yf
import numpy as np

from app.models import (
    FundamentalMetrics, FundamentalResponse,
    ValueScore, ValueCriterion
)
from app.config import settings

logger = logging.getLogger(__name__)


def _v(info: dict, key: str) -> Optional[float]:
    """Extrae float de info dict, devuelve None si no existe o es NaN."""
    val = info.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def fetch_and_analyze(ticker: str) -> FundamentalResponse:
    ticker = ticker.upper()

    # ── Descargar con reintentos ──────────────────────────────────────────────
    stock = None
    info = {}
    for attempt in range(settings.max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info or {}
            if info.get("regularMarketPrice") or info.get("currentPrice"):
                break
        except Exception as e:
            logger.warning(f"yfinance attempt {attempt+1} for {ticker}: {e}")
            if attempt < settings.max_retries - 1:
                time.sleep(settings.retry_delay)

    # ── Revenue Growth YoY desde income statement ────────────────────────────
    revenue_growth_yoy: Optional[float] = None
    try:
        if stock is not None:
            fin = stock.financials
            if fin is not None and not fin.empty:
                rev_row = None
                for label in ["Total Revenue", "Revenue"]:
                    if label in fin.index:
                        rev_row = fin.loc[label]
                        break
                if rev_row is not None and len(rev_row) >= 2:
                    r_curr = float(rev_row.iloc[0])
                    r_prev = float(rev_row.iloc[1])
                    if r_prev and r_prev != 0:
                        revenue_growth_yoy = round(((r_curr - r_prev) / abs(r_prev)) * 100, 2)
    except Exception as e:
        logger.warning(f"Could not compute revenue growth for {ticker}: {e}")

    # ── Construir métricas ────────────────────────────────────────────────────
    eg = _v(info, "earningsGrowth")
    metrics = FundamentalMetrics(
        eps=_v(info, "trailingEps"),
        pe_ratio=_v(info, "trailingPE"),
        forward_pe=_v(info, "forwardPE"),
        pb_ratio=_v(info, "priceToBook"),
        ps_ratio=_v(info, "priceToSalesTrailing12Months"),
        peg_ratio=_v(info, "pegRatio"),
        profit_margin=_v(info, "profitMargins"),
        operating_margin=_v(info, "operatingMargins"),
        roe=_v(info, "returnOnEquity"),
        roa=_v(info, "returnOnAssets"),
        dividend_yield=_v(info, "dividendYield"),
        payout_ratio=_v(info, "payoutRatio"),
        debt_to_equity=_v(info, "debtToEquity"),
        current_ratio=_v(info, "currentRatio"),
        quick_ratio=_v(info, "quickRatio"),
        revenue_growth_yoy=revenue_growth_yoy,
        earnings_growth=round(eg * 100, 2) if eg else None,
        current_price=_v(info, "currentPrice") or _v(info, "regularMarketPrice"),
        week52_high=_v(info, "fiftyTwoWeekHigh"),
        week52_low=_v(info, "fiftyTwoWeekLow"),
        market_cap=_v(info, "marketCap"),
    )

    value_score = _calculate_value_score(metrics)

    return FundamentalResponse(
        ticker=ticker,
        company_name=info.get("longName") or info.get("shortName") or ticker,
        sector=info.get("sector") or "N/A",
        industry=info.get("industry") or "N/A",
        metrics=metrics,
        value_score=value_score,
        last_updated=datetime.now().isoformat(),
        source="live",
    )


def _calculate_value_score(m: FundamentalMetrics) -> ValueScore:
    criteria: list[ValueCriterion] = []
    total = 0
    max_pts = 0

    def add(name: str, desc: str, passed: bool, pts: int, detail: str):
        nonlocal total, max_pts
        max_pts += pts
        earned = pts if passed else 0
        total += earned
        criteria.append(ValueCriterion(
            name=name, description=desc, passed=passed,
            points_earned=earned, points_max=pts, detail=detail
        ))

    # 1. PER
    if m.pe_ratio is not None:
        add("PER Razonable", "P/E < 25", m.pe_ratio < 25, 15,
            f"PER: {m.pe_ratio:.1f}")

    # 2. P/Book
    if m.pb_ratio is not None:
        add("P/Book Bajo", "P/B < 3.0", m.pb_ratio < 3.0, 10,
            f"P/B: {m.pb_ratio:.2f}")

    # 3. Deuda
    if m.debt_to_equity is not None:
        add("Deuda Controlada", "D/E < 100%", m.debt_to_equity < 100, 15,
            f"D/E: {m.debt_to_equity:.1f}%")

    # 4. Margen neto
    if m.profit_margin is not None:
        pct = m.profit_margin * 100 if m.profit_margin < 1 else m.profit_margin
        add("Rentabilidad", "Margen neto > 5%", pct > 5, 15,
            f"Margen: {pct:.1f}%")

    # 5. Current ratio
    if m.current_ratio is not None:
        add("Liquidez", "Current Ratio > 1.5", m.current_ratio > 1.5, 10,
            f"Ratio: {m.current_ratio:.2f}")

    # 6. ROE
    if m.roe is not None:
        pct = m.roe * 100 if m.roe < 1 else m.roe
        add("ROE Elevado", "ROE > 15%", pct > 15, 15,
            f"ROE: {pct:.1f}%")

    # 7. Revenue growth
    if m.revenue_growth_yoy is not None:
        add("Crecimiento YoY", "Ingresos crecen > 5%", m.revenue_growth_yoy > 5, 10,
            f"Crecimiento: {m.revenue_growth_yoy:.1f}%")

    # 8. PEG
    if m.peg_ratio is not None and m.peg_ratio > 0:
        add("PEG Ratio", "PEG < 1.5", m.peg_ratio < 1.5, 10,
            f"PEG: {m.peg_ratio:.2f}")

    # 9. Dividendo (bonus)
    if m.dividend_yield is not None:
        dy = m.dividend_yield * 100 if m.dividend_yield < 1 else m.dividend_yield
        add("Dividendo", "Yield > 1%", dy > 1, 5,
            f"Yield: {dy:.2f}%")

    score = round((total / max_pts * 100)) if max_pts else 0
    passed_count = sum(1 for c in criteria if c.passed)

    if score >= 75:   rating, color = "EXCELENTE", "green"
    elif score >= 55: rating, color = "BUENA",     "blue"
    elif score >= 35: rating, color = "MODERADA",  "yellow"
    else:             rating, color = "DÉBIL",     "red"

    return ValueScore(
        score=score,
        rating=rating,
        color=color,
        criteria=criteria,
        summary=f"{passed_count}/{len(criteria)} criterios cumplidos",
    )
