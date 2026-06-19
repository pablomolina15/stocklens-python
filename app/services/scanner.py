"""
StockLens — Multi-Signal Opportunity Scanner
app/services/scanner.py

Pipeline de 3 capas:
  1. DISCOVERY  — Finviz screener público → top movers por volumen anómalo
  2. CATALYST   — SEC EDGAR 8-K (24h) + Yahoo RSS → noticias + clasificación NLP
  3. TECHNICAL  — RSI, MACD, trend, momentum 5d sobre los candidatos filtrados

Score separado por categoría:
  - catalyst_score  (0-100): calidad y tipo de catalizador
  - technical_score (0-100): confirmación técnica
  - momentum_score  (0-100): fuerza de precio reciente
  - composite_score (0-100): ponderación final

Cache interno de 15 minutos para no re-escanear en cada request.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 15 * 60  # 15 min


def _cached(key: str):
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < CACHE_TTL:
        return entry["data"]
    return None


def _set_cache(key: str, data):
    _cache[key] = {"data": data, "ts": time.time()}


# ── Universe: S&P 500 + Nasdaq 100 tickers (static seed, extended dynamically) ─
# We use a curated 200-ticker seed across sectors, then extend with Finviz movers
SP500_SEED = [
    # Mega cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "ORCL", "AMD",
    # Financials
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "SCHW", "COF",
    # Healthcare / Biotech
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "AMGN", "GILD", "REGN",
    "MRNA", "BIIB", "VRTX", "BMRN", "SGEN", "RARE", "ARWR", "RCUS", "RVMD", "KYMR",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "VLO", "EOG", "DVN",
    # Consumer
    "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "WMT", "LOW", "TJX",
    # Industrials
    "CAT", "DE", "GE", "HON", "RTX", "LMT", "BA", "NOC", "UPS", "FDX",
    # Semis
    "INTC", "QCOM", "TXN", "MRVL", "AMAT", "LRCX", "KLAC", "ASML", "SMCI", "ARM",
    # Growth / Tech
    "CRM", "NOW", "SNOW", "PLTR", "COIN", "SHOP", "UBER", "LYFT", "DASH", "ABNB",
    "ZM", "DDOG", "NET", "MDB", "OKTA", "CRWD", "PANW", "ZS", "FTNT", "CYBR",
    # Telecom / Media
    "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "WBD", "PARA", "SPOT", "RBLX",
    # Real estate / Utilities
    "AMT", "PLD", "EQIX", "NEE", "DUK", "SO", "D", "PCG", "EXC", "AEP",
    # ETF-driven movers (sector ETF components that move a lot)
    "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "ARKK",
    # Small/mid cap high-beta
    "IONQ", "RKLB", "LUNR", "ACHR", "JOBY", "LILM", "SPCE", "ASTS", "SATL",
    "SOUN", "BBAI", "BIGB", "KULR", "MSTR", "HOOD", "SOFI", "AFRM", "UPST", "LC",
    # Pharma catalysts frequent
    "PFE", "AZN", "NVO", "SNY", "RGEN", "INCY", "EXAS", "HALO", "ACAD", "AXSM",
]

# Deduplicate preserving order
_seen: set = set()
UNIVERSE: list[str] = []
for _t in SP500_SEED:
    if _t not in _seen:
        _seen.add(_t)
        UNIVERSE.append(_t)


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class NewsItem:
    title: str
    published: str
    url: str
    source: str


@dataclass
class CatalystInfo:
    type: str           # earnings_beat | fda_approval | contract | partnership | upgrade | macro | other
    headline: str
    sentiment: str      # bullish | bearish | neutral
    source: str
    recency_hours: float
    raw_score: float    # 0-1 raw catalyst quality


@dataclass
class OpportunityCandidate:
    ticker: str
    company_name: str = ""
    current_price: float = 0.0
    market_cap: float = 0.0
    sector: str = ""
    # Scores (0-100)
    catalyst_score: int = 0
    technical_score: int = 0
    momentum_score: int = 0
    composite_score: int = 0
    # Details
    catalysts: list[CatalystInfo] = field(default_factory=list)
    technical_signals: list[str] = field(default_factory=list)
    change_pct_1d: float = 0.0
    change_pct_5d: float = 0.0
    volume_ratio: float = 1.0   # today vs avg20
    rsi: float = 50.0
    trend: str = "neutral"
    macd_bullish: bool = False
    # Metadata
    has_earnings: bool = False
    has_fda: bool = False
    has_sec_filing: bool = False


# ── Layer 1: Discovery via Finviz + yfinance volume screener ──────────────────

async def _fetch_finviz_movers(client: httpx.AsyncClient) -> list[str]:
    """
    Scrape Finviz's public screener for tickers with unusual volume today.
    Returns up to 100 tickers sorted by volume change.
    """
    cached = _cached("finviz_movers")
    if cached:
        return cached

    tickers: list[str] = []
    try:
        # Finviz screener: unusual volume, price > $2, exclude ETFs
        url = (
            "https://finviz.com/screener.ashx"
            "?v=111&f=sh_avgvol_o500,sh_price_o2,sh_curvol_o500"
            "&o=-volume&r=1"
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = await client.get(url, headers=headers, timeout=20, follow_redirects=True)
        html = resp.text

        # Parse ticker symbols from screener table
        # Finviz renders tickers as links: /quote.ashx?t=TICKER
        matches = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})"', html)
        seen: set = set()
        for t in matches:
            if t not in seen and len(t) <= 5:
                seen.add(t)
                tickers.append(t)
                if len(tickers) >= 100:
                    break

        logger.info("Finviz discovery: %d tickers found", len(tickers))

    except Exception as e:
        logger.warning("Finviz scrape failed: %s — falling back to seed universe", e)

    _set_cache("finviz_movers", tickers)
    return tickers


def _get_full_universe(finviz_tickers: list[str]) -> list[str]:
    """Merge Finviz movers with static seed, dedup, limit to 200."""
    combined: list[str] = []
    seen: set = set()
    # Finviz movers first (highest priority)
    for t in finviz_tickers:
        if t not in seen:
            seen.add(t)
            combined.append(t)
    # Then static seed
    for t in UNIVERSE:
        if t not in seen:
            seen.add(t)
            combined.append(t)
    return combined[:250]  # hard cap to keep scan time reasonable


# ── Layer 2a: SEC EDGAR 8-K filings (last 24h) ───────────────────────────────

async def _fetch_edgar_8k(client: httpx.AsyncClient) -> dict[str, list[str]]:
    """
    Pull SEC EDGAR ATOM feed for latest 8-K filings.
    Returns dict: ticker → [headline, ...]
    """
    cached = _cached("edgar_8k")
    if cached:
        return cached

    ticker_filings: dict[str, list[str]] = {}
    try:
        url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=100&search_text=&output=atom"
        headers = {"User-Agent": "StockLens research@stocklens.app", "Accept": "application/atom+xml"}
        resp = await client.get(url, headers=headers, timeout=15)
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        for entry in root.findall("atom:entry", ns):
            title_el  = entry.find("atom:title", ns)
            updated_el = entry.find("atom:updated", ns)
            if title_el is None or updated_el is None:
                continue

            try:
                updated = datetime.fromisoformat(updated_el.text.replace("Z", "+00:00"))
            except Exception:
                continue
            if updated < cutoff:
                continue

            title = title_el.text or ""
            # Extract ticker from title like "8-K - APPLE INC (0000320193) ..."
            # or from company name lookup — try to match known tickers
            for ticker in UNIVERSE:
                # SEC titles don't always include ticker, match by CIK or name later
                # For now, we extract company name and match loosely
                pass

            # Extract company name between first dash and parenthesis
            m = re.match(r"8-K\s*-\s*(.+?)\s*\(", title)
            if m:
                company = m.group(1).strip().upper()
                # Map common company names to tickers
                for ticker, names in _COMPANY_NAME_MAP.items():
                    if any(n in company for n in names):
                        ticker_filings.setdefault(ticker, []).append(title)

        logger.info("EDGAR 8-K: found filings for %d tickers", len(ticker_filings))
    except Exception as e:
        logger.warning("EDGAR fetch failed: %s", e)

    _set_cache("edgar_8k", ticker_filings)
    return ticker_filings


# Lightweight company name → ticker mapping for SEC parsing
_COMPANY_NAME_MAP: dict[str, list[str]] = {
    "AAPL": ["APPLE"], "MSFT": ["MICROSOFT"], "NVDA": ["NVIDIA"],
    "GOOGL": ["ALPHABET", "GOOGLE"], "AMZN": ["AMAZON"], "META": ["META PLATFORMS"],
    "TSLA": ["TESLA"], "AVGO": ["BROADCOM"], "ORCL": ["ORACLE"], "AMD": ["ADVANCED MICRO"],
    "JPM": ["JPMORGAN", "JP MORGAN"], "BAC": ["BANK OF AMERICA"],
    "GS": ["GOLDMAN SACHS"], "MS": ["MORGAN STANLEY"],
    "JNJ": ["JOHNSON"], "LLY": ["ELI LILLY"], "ABBV": ["ABBVIE"],
    "MRK": ["MERCK"], "AMGN": ["AMGEN"], "GILD": ["GILEAD"],
    "MRNA": ["MODERNA"], "BIIB": ["BIOGEN"], "VRTX": ["VERTEX"],
    "PFE": ["PFIZER"], "AZN": ["ASTRAZENECA"],
    "XOM": ["EXXON"], "CVX": ["CHEVRON"], "COP": ["CONOCOPHILLIPS"],
    "NFLX": ["NETFLIX"], "DIS": ["DISNEY", "WALT DISNEY"],
    "COIN": ["COINBASE"], "PLTR": ["PALANTIR"], "SHOP": ["SHOPIFY"],
    "CRWD": ["CROWDSTRIKE"], "PANW": ["PALO ALTO"], "SNOW": ["SNOWFLAKE"],
    "NOW": ["SERVICENOW"], "CRM": ["SALESFORCE"],
    "HOOD": ["ROBINHOOD"], "SOFI": ["SOFI TECHNOLOGIES"],
    "MSTR": ["MICROSTRATEGY"], "IONQ": ["IONQ"],
    "RKLB": ["ROCKET LAB"], "ASTS": ["AST SPACEMOBILE"],
}


# ── Layer 2b: Yahoo Finance RSS news per ticker ────────────────────────────────

async def _fetch_yahoo_news(client: httpx.AsyncClient, ticker: str) -> list[NewsItem]:
    """Fetch Yahoo Finance RSS for a ticker. Returns last 5 items from 24h."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        resp = await client.get(url, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        items: list[NewsItem] = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=48)

        for item in root.findall(".//item")[:10]:
            title = item.findtext("title") or ""
            pub   = item.findtext("pubDate") or ""
            link  = item.findtext("link") or ""

            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass

            items.append(NewsItem(title=title, published=pub, url=link, source="yahoo_rss"))
            if len(items) >= 5:
                break

        return items
    except Exception:
        return []


# ── Layer 2c: NLP catalyst classification (keyword-based, no model needed) ────

# Catalyst patterns: (regex, catalyst_type, base_score, sentiment)
_CATALYST_PATTERNS: list[tuple[str, str, float, str]] = [
    # Earnings
    (r"\b(beat|beats|topped|surpass|exceeded)\b.{0,40}\b(earn|eps|revenue|estimate)", "earnings_beat", 0.90, "bullish"),
    (r"\b(record|all.time).{0,20}\b(revenue|profit|sales|earnings)", "earnings_beat", 0.85, "bullish"),
    (r"\b(raised?|raise|raises|raising|upped?|increases?)\b.{0,30}\b(guidance|outlook|forecast)", "guidance_raise", 0.85, "bullish"),
    (r"\b(miss|misses|missed|below|disappointing)\b.{0,40}\b(earn|eps|revenue|estimate)", "earnings_miss", 0.80, "bearish"),
    (r"\b(cut|cuts|lowers?|reducing?|reduce|withdrew?)\b.{0,30}\b(guidance|outlook|forecast)", "guidance_cut", 0.80, "bearish"),
    # FDA / Regulatory
    (r"\bfda\b.{0,50}\b(approv|authoriz|clear|grant)", "fda_approval", 0.95, "bullish"),
    (r"\b(approv|authoriz|clear).{0,50}\bfda\b", "fda_approval", 0.95, "bullish"),
    (r"\bpdufa\b", "fda_catalyst", 0.80, "bullish"),
    (r"\b(nda|bla|anda)\b.{0,30}\b(approv|accept|grant)", "fda_approval", 0.90, "bullish"),
    (r"\bfda\b.{0,50}\b(reject|refus|complet.response|hold|delay)", "fda_rejection", 0.85, "bearish"),
    (r"\bclinical.trial.{0,30}\b(success|positive|met.{0,10}endpoint)", "clinical_trial", 0.80, "bullish"),
    (r"\bphase [23]\b.{0,50}\b(success|positive|met)", "clinical_trial", 0.80, "bullish"),
    # Contracts / Government
    (r"\b(awarded?|wins?|secures?|receives?)\b.{0,40}\b(contract|deal|agreement)\b.{0,30}\$([\d\.]+[bm])", "contract", 0.80, "bullish"),
    (r"\b(government|pentagon|dod|dhs|nasa|army|navy|air.force)\b.{0,50}\b(contract|deal|award)", "gov_contract", 0.85, "bullish"),
    (r"\b(department.of.defense|u\.s\. government|federal)\b.{0,50}\b(award|select|choose)", "gov_contract", 0.85, "bullish"),
    # Partnerships / M&A
    (r"\b(partnership|collaboration|joint.venture|licensing.agreement)\b", "partnership", 0.70, "bullish"),
    (r"\b(acqui|merger|buyout|takeover|bid)\b", "ma_activity", 0.75, "bullish"),
    (r"\b(strategic.alliance|co.develop|co.promot)\b", "partnership", 0.65, "bullish"),
    # Analyst / Upgrades
    (r"\b(upgrade|upgraded|rais.{0,5}target|rais.{0,5}price.target|rais.{0,5}rating)\b", "analyst_upgrade", 0.65, "bullish"),
    (r"\b(buy|strong.buy|outperform|overweight)\b.{0,30}\b(initiat|reiterat|maintain)\b", "analyst_upgrade", 0.60, "bullish"),
    (r"\b(downgrade|lower.{0,5}target|cut.{0,5}target|underperform|underweight)\b", "analyst_downgrade", 0.60, "bearish"),
    # Macro
    (r"\b(fed|federal.reserve|fomc)\b.{0,50}\b(cut|lower|pivot|pause)\b.{0,30}\brate", "macro_positive", 0.70, "bullish"),
    (r"\b(cpi|inflation|pce)\b.{0,30}\b(cool|fell?|low|below.expect)", "macro_positive", 0.65, "bullish"),
    (r"\b(fed|federal.reserve)\b.{0,50}\b(hike|raise|higher)\b.{0,30}\brate", "macro_negative", 0.65, "bearish"),
    # Share buyback / Dividend
    (r"\b(buyback|repurchas|share.repurchas)\b.{0,30}\$([\d\.]+[bm])", "buyback", 0.70, "bullish"),
    (r"\b(dividend|divid).{0,30}\b(increas|rais|special|extra)\b", "dividend_raise", 0.65, "bullish"),
    # Product launches
    (r"\b(launch|release|unveil|announc).{0,40}\b(product|chip|model|platform|service|drug)", "product_launch", 0.65, "bullish"),
    (r"\b(generative.?ai|ai.chip|llm|gpu|accelerat)\b.{0,30}\b(partner|adopt|integrat|deploy)", "ai_catalyst", 0.70, "bullish"),
]

_COMPILED_PATTERNS = [
    (re.compile(pat, re.IGNORECASE), ctype, score, sentiment)
    for pat, ctype, score, sentiment in _CATALYST_PATTERNS
]


def _classify_catalyst(text: str, source: str, pub_time_str: str) -> Optional[CatalystInfo]:
    """
    Match text against catalyst patterns.
    Returns CatalystInfo or None if no catalyst detected.
    """
    # Calculate recency
    recency_hours = 24.0
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        recency_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        pass

    best_match: Optional[tuple] = None
    for pattern, ctype, base_score, sentiment in _COMPILED_PATTERNS:
        if pattern.search(text):
            # Recency bonus: fresher news scores higher
            recency_bonus = max(0, (48 - recency_hours) / 48) * 0.15
            score = min(1.0, base_score + recency_bonus)
            if best_match is None or score > best_match[2]:
                best_match = (ctype, sentiment, score)

    if best_match is None:
        return None

    ctype, sentiment, score = best_match
    # Truncate headline
    headline = text[:120] + ("…" if len(text) > 120 else "")

    return CatalystInfo(
        type=ctype,
        headline=headline,
        sentiment=sentiment,
        source=source,
        recency_hours=round(recency_hours, 1),
        raw_score=round(score, 3),
    )


def _compute_catalyst_score(catalysts: list[CatalystInfo]) -> int:
    """
    Convert list of catalysts into a 0-100 score.
    Bullish catalysts add, bearish subtract. Multiple catalysts compound.
    """
    if not catalysts:
        return 0

    bullish = [c for c in catalysts if c.sentiment == "bullish"]
    bearish = [c for c in catalysts if c.sentiment == "bearish"]

    # Primary signal: strongest single catalyst
    bull_score = max((c.raw_score for c in bullish), default=0.0)
    bear_score = max((c.raw_score for c in bearish), default=0.0)

    # Secondary signals compound logarithmically
    if len(bullish) > 1:
        bull_score = min(1.0, bull_score + sum(c.raw_score * 0.2 for c in bullish[1:]))

    net = bull_score - bear_score * 0.8  # bearish discounted slightly (short squeeze risk)
    return max(0, min(100, int(net * 100)))


# ── Layer 3: Technical scoring ────────────────────────────────────────────────

def _technical_score_from_df(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    """
    Compute technical score and extract key signals from a price DataFrame.
    Returns (score 0-100, signal_strings, raw_metrics)
    """
    if len(df) < 20:
        return 0, [], {}

    closes  = df["Close"]
    volumes = df["Volume"]

    score   = 0
    signals: list[str] = []
    metrics: dict = {}

    last    = df.iloc[-1]
    prev5   = df.iloc[-6] if len(df) >= 6 else df.iloc[0]
    prev20  = df.iloc[-21] if len(df) >= 21 else df.iloc[0]

    # ── RSI ──────────────────────────────────────────────────────────────────
    import pandas_ta as ta
    try:
        rsi_s = ta.rsi(closes, length=14)
        rsi   = float(rsi_s.iloc[-1]) if rsi_s is not None and not rsi_s.empty else 50.0
    except Exception:
        rsi = 50.0
    metrics["rsi"] = round(rsi, 1)

    if 55 <= rsi <= 70:
        score += 20; signals.append(f"RSI {rsi:.0f} (zona fuerte)")
    elif 70 < rsi < 80:
        score += 8
    elif rsi > 80:
        score -= 10
    elif rsi < 40:
        score -= 15
    elif 40 <= rsi < 50:
        score -= 5

    # ── Trend (SMA50 vs SMA200) ───────────────────────────────────────────────
    trend = "neutral"
    try:
        sma50  = float(ta.sma(closes, length=50).iloc[-1])
        sma200 = float(ta.sma(closes, length=200).iloc[-1]) if len(closes) >= 200 else sma50
        close_now = float(closes.iloc[-1])
        if sma50 > sma200:
            trend = "bullish"; score += 15; signals.append("Tendencia alcista (SMA50>SMA200)")
        elif sma50 < sma200:
            trend = "bearish"; score -= 20
        if close_now > sma50:
            score += 10; signals.append("Precio sobre SMA50")
        else:
            score -= 8
    except Exception:
        pass
    metrics["trend"] = trend

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_bullish = False
    try:
        macd_df = ta.macd(closes, fast=12, slow=26, signal=9)
        if macd_df is not None and len(macd_df) >= 2:
            macd_col = next((c for c in macd_df.columns if c.startswith("MACD_12")), None)
            sig_col  = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
            if macd_col and sig_col:
                m_now, m_prev = float(macd_df[macd_col].iloc[-1]), float(macd_df[macd_col].iloc[-2])
                s_now, s_prev = float(macd_df[sig_col].iloc[-1]),  float(macd_df[sig_col].iloc[-2])
                if m_prev < s_prev and m_now > s_now:
                    macd_bullish = True; score += 15; signals.append("MACD cruce alcista")
                elif m_now > s_now:
                    score += 5  # already crossed, still bullish
                elif m_prev > s_prev and m_now < s_now:
                    score -= 15  # death cross MACD
    except Exception:
        pass
    metrics["macd_bullish"] = macd_bullish

    # ── Momentum 5d & 1d ─────────────────────────────────────────────────────
    change_5d = float((closes.iloc[-1] - prev5["Close"]) / (prev5["Close"] + 1e-9) * 100)
    change_1d = float((closes.iloc[-1] - df.iloc[-2]["Close"]) / (df.iloc[-2]["Close"] + 1e-9) * 100) if len(df) >= 2 else 0.0
    metrics["change_1d"] = round(change_1d, 2)
    metrics["change_5d"] = round(change_5d, 2)

    if change_5d > 0:
        score += min(int(change_5d * 3), 25)
        if change_5d > 3:
            signals.append(f"+{change_5d:.1f}% últimos 5 días")
    else:
        score += max(int(change_5d * 2), -20)

    # ── Volume anomaly ────────────────────────────────────────────────────────
    avg_vol = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
    today_vol = float(volumes.iloc[-1])
    vol_ratio = today_vol / (avg_vol + 1) if avg_vol > 0 else 1.0
    metrics["volume_ratio"] = round(vol_ratio, 2)

    if vol_ratio > 2.5:
        score += 20; signals.append(f"Volumen {vol_ratio:.1f}x la media (anomalía)")
    elif vol_ratio > 1.5:
        score += 10; signals.append(f"Volumen elevado {vol_ratio:.1f}x")
    elif vol_ratio < 0.5:
        score -= 5

    # ── Bollinger breakout potential ──────────────────────────────────────────
    try:
        bb = ta.bbands(closes, length=20, std=2)
        if bb is not None:
            bbu_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
            if bbu_col:
                bbu = float(bb[bbu_col].iloc[-1])
                pct_from_upper = (float(closes.iloc[-1]) - bbu) / (bbu + 1e-9) * 100
                if -3 < pct_from_upper < 2:
                    score += 10; signals.append("Cerca de ruptura Bollinger superior")
                elif pct_from_upper > 5:
                    score -= 10
    except Exception:
        pass

    return max(0, min(100, score)), signals, metrics


# ── Momentum score (pure price action, no indicators) ────────────────────────

def _momentum_score(metrics: dict) -> int:
    """Fast momentum score from pre-computed metrics."""
    score = 0
    c1 = metrics.get("change_1d", 0)
    c5 = metrics.get("change_5d", 0)
    vr = metrics.get("volume_ratio", 1.0)

    # 1-day price action
    if c1 > 0:
        score += min(int(c1 * 8), 30)
    else:
        score += max(int(c1 * 5), -25)

    # 5-day trend
    if c5 > 0:
        score += min(int(c5 * 3), 25)
    else:
        score += max(int(c5 * 2), -20)

    # Volume confirms momentum
    if vr > 1.5 and (c1 > 0 or c5 > 0):
        score += min(int((vr - 1) * 10), 25)

    return max(0, min(100, score))


# ── Main scan pipeline ─────────────────────────────────────────────────────────

async def run_opportunity_scan(max_results: int = 10) -> dict:
    """
    Full 3-layer scan. Returns structured opportunity list with separate scores.
    Cached for 15 minutes to avoid hammering external services.
    """
    cached = _cached("full_scan")
    if cached:
        logger.info("Returning cached scan results")
        return cached

    scan_start = time.time()
    logger.info("Starting full opportunity scan")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(25.0, connect=10.0),
        headers={"User-Agent": "StockLens/2.0 research@stocklens.app"},
        follow_redirects=True,
    ) as client:

        # ── Layer 1: Discovery ────────────────────────────────────────────────
        finviz_tickers = await _fetch_finviz_movers(client)
        universe = _get_full_universe(finviz_tickers)
        logger.info("Universe: %d tickers to scan", len(universe))

        # ── Layer 2a: SEC EDGAR (single request, covers all tickers) ─────────
        edgar_map = await _fetch_edgar_8k(client)

        # ── Layer 2b: Yahoo news — batch with concurrency limit ───────────────
        # Only fetch news for tickers that appear in Finviz movers OR have EDGAR filing
        priority_tickers = list(set(finviz_tickers[:80]) | set(edgar_map.keys()) | set(UNIVERSE[:60]))
        priority_tickers = [t for t in priority_tickers if t in set(universe)][:120]

        sem = asyncio.Semaphore(12)  # max 12 concurrent Yahoo RSS requests

        async def fetch_news_limited(ticker: str):
            async with sem:
                return ticker, await _fetch_yahoo_news(client, ticker)

        news_tasks = [fetch_news_limited(t) for t in priority_tickers]
        news_results = await asyncio.gather(*news_tasks, return_exceptions=True)

        news_map: dict[str, list[NewsItem]] = {}
        for result in news_results:
            if isinstance(result, Exception):
                continue
            ticker, items = result
            if items:
                news_map[ticker] = items

    # ── Layer 3: Technical analysis + scoring ─────────────────────────────────
    # Download price data for all universe tickers in batch (yfinance handles this well)
    logger.info("Downloading price data for %d tickers", len(universe))

    try:
        raw = yf.download(
            universe,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
        raw = None

    candidates: list[OpportunityCandidate] = []

    for ticker in universe:
        try:
            # Extract single-ticker data from batch download
            if raw is None:
                continue

            if len(universe) == 1:
                df = raw
            elif isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    continue
                df = raw[ticker].dropna(subset=["Close"])
            else:
                continue

            if df is None or len(df) < 10:
                continue

            # ── Classify catalysts ────────────────────────────────────────────
            catalysts: list[CatalystInfo] = []

            # From Yahoo RSS news
            for item in news_map.get(ticker, []):
                cat = _classify_catalyst(item.title, item.source, item.published)
                if cat:
                    catalysts.append(cat)

            # From SEC EDGAR
            for headline in edgar_map.get(ticker, []):
                cat = _classify_catalyst(headline, "sec_edgar", "")
                if cat:
                    catalysts.append(cat)

            # ── Technical score ───────────────────────────────────────────────
            tech_score, tech_signals, metrics = _technical_score_from_df(df)

            # ── Catalyst score ────────────────────────────────────────────────
            cat_score = _compute_catalyst_score(catalysts)

            # ── Momentum score ────────────────────────────────────────────────
            mom_score = _momentum_score(metrics)

            # ── Composite (weighted) ──────────────────────────────────────────
            # Catalyst is the primary driver for this scanner
            composite = int(
                cat_score  * 0.45 +
                tech_score * 0.30 +
                mom_score  * 0.25
            )

            # ── Filter: need at least SOME catalyst OR exceptional technicals ─
            # Pure technical plays need score >= 55 (handled by momentum-scan already)
            # This scanner prioritises catalyst-driven moves
            has_catalyst = cat_score >= 30
            strong_technical = tech_score >= 60 and mom_score >= 50

            if not (has_catalyst or strong_technical):
                continue

            # ── Get metadata (name, sector, price) from df ────────────────────
            current_price = float(df["Close"].iloc[-1])
            company_name  = ticker  # yfinance batch doesn't return names; use ticker

            candidate = OpportunityCandidate(
                ticker=ticker,
                company_name=company_name,
                current_price=round(current_price, 2),
                sector="",
                catalyst_score=min(100, cat_score),
                technical_score=min(100, tech_score),
                momentum_score=min(100, mom_score),
                composite_score=min(100, composite),
                catalysts=catalysts,
                technical_signals=tech_signals,
                change_pct_1d=metrics.get("change_1d", 0.0),
                change_pct_5d=metrics.get("change_5d", 0.0),
                volume_ratio=metrics.get("volume_ratio", 1.0),
                rsi=metrics.get("rsi", 50.0),
                trend=metrics.get("trend", "neutral"),
                macd_bullish=metrics.get("macd_bullish", False),
                has_sec_filing=ticker in edgar_map,
            )
            candidates.append(candidate)

        except Exception as e:
            logger.debug("Error processing %s: %s", ticker, e)
            continue

    # ── Sort by composite score and return top N ──────────────────────────────
    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    top = candidates[:max_results]

    # Serialize to dict
    results = {
        "opportunities": [_serialize(c) for c in top],
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": len(universe),
        "candidates_found": len(candidates),
        "scan_duration_s": round(time.time() - scan_start, 1),
        "sources": ["finviz_screener", "sec_edgar_8k", "yahoo_rss", "yfinance_technical"],
        "cache_ttl_min": CACHE_TTL // 60,
        "disclaimer": (
            "Scanner multi-señal experimental. Catalizadores detectados por NLP "
            "sobre titulares — pueden contener errores. No constituye consejo de inversión."
        ),
    }

    _set_cache("full_scan", results)
    logger.info(
        "Scan complete: %d candidates from %d tickers in %.1fs",
        len(candidates), len(universe), time.time() - scan_start,
    )
    return results


def _serialize(c: OpportunityCandidate) -> dict:
    return {
        "ticker": c.ticker,
        "company_name": c.company_name,
        "current_price": c.current_price,
        "sector": c.sector,
        "scores": {
            "catalyst":  c.catalyst_score,
            "technical": c.technical_score,
            "momentum":  c.momentum_score,
            "composite": c.composite_score,
        },
        "catalysts": [
            {
                "type":          cat.type,
                "headline":      cat.headline,
                "sentiment":     cat.sentiment,
                "source":        cat.source,
                "recency_hours": cat.recency_hours,
            }
            for cat in c.catalysts
        ],
        "technicals": {
            "signals":        c.technical_signals,
            "rsi":            c.rsi,
            "trend":          c.trend,
            "macd_bullish":   c.macd_bullish,
            "change_pct_1d":  c.change_pct_1d,
            "change_pct_5d":  c.change_pct_5d,
            "volume_ratio":   c.volume_ratio,
        },
        "flags": {
            "has_sec_filing": c.has_sec_filing,
        },
    }
