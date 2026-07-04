"""
StockLens — Multi-Signal Opportunity Scanner v2
app/services/scanner.py

Mejoras v2 sobre v1:
  ✅ S&P 500 completo dinámico via Wikipedia (actualizado automáticamente)
  ✅ Earnings calendar via Yahoo Finance (catalizador máximo esta semana)
  ✅ SEC EDGAR con CIK lookup para mapear nombre→ticker correctamente
  ✅ Volumen intraday anómalo (señal de actividad institucional)
  ✅ Score de momentum mejorado con volatility-adjusted returns
  ✅ Cache multinivel: universo 24h, earnings 6h, scan completo 15min
  ✅ Universo ilimitado: S&P500 + Finviz movers + seed curado (~750 tickers)
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

# ── Cache multinivel ──────────────────────────────────────────────────────────
_cache: dict = {}

CACHE_TTL = {
    "sp500_universe": 24 * 3600,   # S&P 500 lista: 24 horas
    "finviz_movers":  30 * 60,     # Finviz movers: 30 min
    "earnings_week":  6 * 3600,    # Earnings calendar: 6 horas
    "edgar_8k":       30 * 60,     # SEC EDGAR 8-K: 30 min
    "full_scan":      15 * 60,     # Scan completo: 15 min
}


def _cached(key: str) -> Optional[object]:
    entry = _cache.get(key)
    if not entry:
        return None
    category = key.split(":")[0]
    ttl = CACHE_TTL.get(category, 900)
    if time.time() - entry["ts"] < ttl:
        return entry["data"]
    return None


def _set_cache(key: str, data: object) -> None:
    _cache[key] = {"data": data, "ts": time.time()}


# ── Seed universe (fallback si falla Wikipedia) ───────────────────────────────
_SEED_UNIVERSE = [
    # Mega cap tech
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","ORCL","AMD",
    "ADBE","CRM","NOW","INTU","CSCO","IBM","ANET","CDNS","SNPS","ANSS",
    # Financials
    "JPM","BAC","GS","MS","WFC","BLK","C","AXP","SCHW","COF","V","MA",
    "SPGI","MCO","ICE","CME","CBOE","NDAQ","BX","KKR","APO","ARES",
    # Healthcare / Biotech
    "UNH","JNJ","LLY","ABBV","MRK","TMO","ABT","AMGN","GILD","REGN",
    "MRNA","BIIB","VRTX","PFE","AZN","NVO","BMY","JAZZ","ALNY","SRPT",
    "BMRN","SGEN","RARE","ARWR","RCUS","RVMD","KYMR","ACAD","AXSM","RGEN",
    "INCY","EXAS","HALO","BEAM","EDIT","CRSP","NTLA","FATE","BLUE",
    # Energy
    "XOM","CVX","COP","SLB","OXY","MPC","PSX","VLO","EOG","DVN",
    # Consumer
    "HD","MCD","NKE","SBUX","TGT","COST","WMT","LOW","TJX","AMZN",
    "PG","KO","PEP","PM","MO","CL","EL","CHD","CLX",
    # Industrials
    "CAT","DE","GE","HON","RTX","LMT","BA","NOC","UPS","FDX",
    "LIN","APD","SHW","ECL","PPG",
    # Semis
    "INTC","QCOM","TXN","MRVL","AMAT","LRCX","KLAC","ASML","SMCI","ARM",
    "MCHP","MPWR","ON","SWKS","QRVO",
    # Growth / SaaS
    "SNOW","PLTR","COIN","SHOP","UBER","LYFT","DASH","ABNB",
    "ZM","DDOG","NET","MDB","OKTA","CRWD","PANW","ZS","FTNT","CYBR",
    "GTLB","HUBS","SMAR","BILL","BRZE","TOST","IOT","APPN",
    # Media / Telecom
    "NFLX","DIS","CMCSA","T","VZ","TMUS","SPOT","RBLX","WBD","PARA",
    # REITs / Utilities
    "AMT","PLD","EQIX","NEE","DUK","SO","CCI","WELL","DLR","O",
    # ETFs como proxy de mercado
    "SPY","QQQ","IWM","XLK","XLF","XLV","XLE","XLI","XLY","ARKK",
    "SMH","IBB","XBI","GDX","GDXJ",
    # Small/mid cap high-beta
    "IONQ","RKLB","ACHR","JOBY","ASTS","SOUN","MSTR","HOOD","SOFI",
    "AFRM","UPST","LC","OPEN","OFFERPAD","SMTC","KULR","BBAI",
    # Healthcare services
    "CVS","CI","HUM","MOH","CNC","ELV","HCA","THC","UHS",
    "DHR","STE","EW","ISRG","MDT","BSX","ZBH","SYK","BDX",
]


# ── Layer 0: S&P 500 dinámico via Wikipedia ───────────────────────────────────

async def _fetch_sp500_wikipedia(client: httpx.AsyncClient) -> list[str]:
    """
    Obtiene la lista actual del S&P 500 desde Wikipedia.
    Cacheado 24h — la lista cambia muy raramente.
    """
    cached = _cached("sp500_universe")
    if cached:
        return cached  # type: ignore

    tickers: list[str] = []
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = await client.get(url, timeout=20, headers={
            "User-Agent": "StockLens research@stocklens.app"
        })
        html = resp.text

        # Parse ticker symbols from the wikitable
        # Format: <td><a href="/wiki/AAPL" ...>AAPL</a></td>
        matches = re.findall(
            r'<td><a[^>]+href="/wiki/[^"]+_\(.*?\)"[^>]*>([A-Z]{1,5})</a></td>',
            html
        )
        if len(matches) < 400:
            # Fallback: try simpler pattern
            matches = re.findall(r'<td style="text-align:left">([A-Z]{1,5})</td>', html)

        if len(matches) < 400:
            # Another fallback pattern for Wikipedia table format
            matches = re.findall(r'<td><a[^>]*>([A-Z]{1,5})</a></td>', html)
            matches = [m for m in matches if 1 < len(m) <= 5]

        # Deduplicate
        seen: set = set()
        for t in matches:
            if t not in seen and t.isalpha():
                seen.add(t)
                tickers.append(t)

        logger.info("S&P 500 Wikipedia: %d tickers fetched", len(tickers))

        if len(tickers) < 400:
            logger.warning("Wikipedia S&P 500 only returned %d tickers, using seed", len(tickers))
            tickers = []

    except Exception as e:
        logger.warning("Wikipedia S&P 500 fetch failed: %s", e)

    _set_cache("sp500_universe", tickers)
    return tickers


# ── Layer 0b: Earnings Calendar esta semana ───────────────────────────────────

async def _fetch_earnings_this_week(client: httpx.AsyncClient) -> dict[str, dict]:
    """
    Obtiene el calendario de earnings de esta semana via Yahoo Finance.
    Los earnings son el catalizador más potente para movimientos de precio.
    Retorna: { ticker: { date, eps_estimate, is_today, days_until } }
    """
    cached = _cached("earnings_week")
    if cached:
        return cached  # type: ignore

    earnings_map: dict[str, dict] = {}

    try:
        # Yahoo Finance earnings calendar endpoint
        today = datetime.now(timezone.utc)
        # Scan next 7 days
        for day_offset in range(8):
            check_date = today + timedelta(days=day_offset)
            date_str   = check_date.strftime("%Y-%m-%d")

            url = f"https://finance.yahoo.com/calendar/earnings?day={date_str}"
            resp = await client.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
            })

            html = resp.text

            # Extract tickers from earnings page
            # Format in Yahoo: data-symbol="AAPL"
            ticker_matches = re.findall(r'data-symbol="([A-Z]{1,5})"', html)
            for ticker in set(ticker_matches):
                if ticker not in earnings_map:
                    is_today    = day_offset == 0
                    days_until  = day_offset
                    earnings_map[ticker] = {
                        "date":       date_str,
                        "is_today":   is_today,
                        "days_until": days_until,
                        "is_this_week": True,
                    }

            # Don't hammer Yahoo
            await asyncio.sleep(0.5)

        logger.info("Earnings calendar: %d tickers reporting this week", len(earnings_map))

    except Exception as e:
        logger.warning("Earnings calendar fetch failed: %s", e)

    _set_cache("earnings_week", earnings_map)
    return earnings_map


# ── Layer 1: Finviz movers ────────────────────────────────────────────────────

async def _fetch_finviz_movers(client: httpx.AsyncClient) -> list[str]:
    """Top movers por volumen anómalo hoy en Finviz."""
    cached = _cached("finviz_movers")
    if cached:
        return cached  # type: ignore

    tickers: list[str] = []
    try:
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
        }
        resp = await client.get(url, headers=headers, timeout=20, follow_redirects=True)
        html = resp.text

        matches = re.findall(r'quote\.ashx\?t=([A-Z]{1,5})"', html)
        seen: set = set()
        for t in matches:
            if t not in seen:
                seen.add(t)
                tickers.append(t)
                if len(tickers) >= 100:
                    break

        logger.info("Finviz discovery: %d tickers", len(tickers))
    except Exception as e:
        logger.warning("Finviz failed: %s", e)

    _set_cache("finviz_movers", tickers)
    return tickers


# ── Build full universe ───────────────────────────────────────────────────────

def _build_universe(sp500: list[str], finviz: list[str]) -> list[str]:
    """
    Combina S&P500 + Finviz movers + seed curado.
    Finviz movers van primero (mayor prioridad — tienen actividad HOY).
    Límite: 750 tickers para mantener tiempo de scan razonable.
    """
    combined: list[str] = []
    seen: set = set()

    # Finviz movers first (highest signal today)
    for t in finviz:
        if t not in seen:
            seen.add(t); combined.append(t)

    # S&P 500 completo
    for t in sp500:
        if t not in seen:
            seen.add(t); combined.append(t)

    # Seed curado (tickers extra no en S&P 500)
    for t in _SEED_UNIVERSE:
        if t not in seen:
            seen.add(t); combined.append(t)

    result = combined[:750]
    logger.info("Universe built: %d tickers (finviz=%d, sp500=%d, seed=%d)",
                len(result), len(finviz), len(sp500), len(_SEED_UNIVERSE))
    return result


# ── SEC EDGAR 8-K mejorado ────────────────────────────────────────────────────

# CIK map: CIK number → ticker (top 500 companies)
# Pre-built para las empresas más relevantes
_CIK_TO_TICKER: dict[str, str] = {
    "0000320193": "AAPL", "0000789019": "MSFT", "0001045810": "NVDA",
    "0001652044": "GOOGL","0001018724": "AMZN", "0001326801": "META",
    "0001318605": "TSLA", "0001730168": "AVGO", "0001341439": "ORCL",
    "0000002488": "AMD",  "0000019617": "JPM",  "0000070858": "BAC",
    "0000886158": "GS",   "0000895421": "MS",   "0000072971": "WFC",
    "0001364742": "BLK",  "0000831001": "C",    "0000004962": "AXP",
    "0000316206": "SCHW", "0000927628": "V",    "0001141391": "MA",
    "0000731788": "UNH",  "0000200406": "JNJ",  "0001792789": "LLY",
    "0001551152": "ABBV", "0000310158": "MRK",  "0000097476": "TMO",
    "0000001800": "ABT",  "0000014272": "AMGN", "0000882095": "GILD",
    "0000874716": "REGN", "0001682852": "MRNA", "0000875320": "BIIB",
    "0000875374": "VRTX", "0000078003": "PFE",  "0001692819": "AZN",
    "0000092380": "XOM",  "0000093410": "CVX",  "0000028823": "COP",
    "0000086312": "HD",   "0000063908": "MCD",  "0000320187": "NKE",
    "0000829224": "SBUX", "0000916365": "TGT",  "0000723254": "COST",
    "0000104169": "WMT",  "0000060667": "LOW",  "0001116132": "NFLX",
    "0001744489": "DIS",  "0001713683": "UBER", "0001645590": "COIN",
    "0001383312": "PLTR", "0001594686": "SHOP",
    "0001841125": "CRWD", "0001327567": "PANW", "0001376789": "NET",
    "0001834175": "SNOW", "0001786248": "DDOG",
}

async def _fetch_edgar_8k(client: httpx.AsyncClient) -> dict[str, list[str]]:
    """SEC EDGAR 8-K filings de las últimas 24h con CIK lookup mejorado."""
    cached = _cached("edgar_8k")
    if cached:
        return cached  # type: ignore

    ticker_filings: dict[str, list[str]] = {}
    try:
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            "?action=getcurrent&type=8-K&dateb=&owner=include"
            "&count=100&search_text=&output=atom"
        )
        headers = {
            "User-Agent": "StockLens research@stocklens.app",
            "Accept": "application/atom+xml",
        }
        resp = await client.get(url, headers=headers, timeout=15)
        root = ET.fromstring(resp.text)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        for entry in root.findall("atom:entry", ns):
            title_el   = entry.find("atom:title", ns)
            updated_el = entry.find("atom:updated", ns)
            link_el    = entry.find("atom:link", ns)

            if title_el is None or updated_el is None:
                continue

            try:
                updated = datetime.fromisoformat(
                    updated_el.text.replace("Z", "+00:00"))
            except Exception:
                continue
            if updated < cutoff:
                continue

            title = title_el.text or ""

            # Method 1: Extract CIK from link and map to ticker
            link_href = link_el.get("href", "") if link_el is not None else ""
            cik_match = re.search(r'CIK=(\d+)', link_href, re.IGNORECASE)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                ticker = _CIK_TO_TICKER.get(cik)
                if ticker:
                    ticker_filings.setdefault(ticker, []).append(title)
                    continue

            # Method 2: Extract from title via company name map
            m = re.match(r"8-K\s*-\s*(.+?)\s*\(", title)
            if m:
                company = m.group(1).strip().upper()
                for ticker, names in _COMPANY_NAME_MAP.items():
                    if any(n in company for n in names):
                        ticker_filings.setdefault(ticker, []).append(title)
                        break

        logger.info("EDGAR 8-K: %d tickers with filings", len(ticker_filings))
    except Exception as e:
        logger.warning("EDGAR fetch failed: %s", e)

    _set_cache("edgar_8k", ticker_filings)
    return ticker_filings


_COMPANY_NAME_MAP: dict[str, list[str]] = {
    "AAPL": ["APPLE"], "MSFT": ["MICROSOFT"], "NVDA": ["NVIDIA"],
    "GOOGL": ["ALPHABET","GOOGLE"], "AMZN": ["AMAZON"],
    "META": ["META PLATFORMS","FACEBOOK"], "TSLA": ["TESLA"],
    "AVGO": ["BROADCOM"], "ORCL": ["ORACLE"], "AMD": ["ADVANCED MICRO"],
    "JPM": ["JPMORGAN","JP MORGAN"], "BAC": ["BANK OF AMERICA"],
    "GS": ["GOLDMAN SACHS"], "MS": ["MORGAN STANLEY"],
    "JNJ": ["JOHNSON"], "LLY": ["ELI LILLY"], "ABBV": ["ABBVIE"],
    "MRK": ["MERCK"], "AMGN": ["AMGEN"], "GILD": ["GILEAD"],
    "MRNA": ["MODERNA"], "BIIB": ["BIOGEN"], "VRTX": ["VERTEX"],
    "PFE": ["PFIZER"], "AZN": ["ASTRAZENECA"],
    "XOM": ["EXXON"], "CVX": ["CHEVRON"], "COP": ["CONOCOPHILLIPS"],
    "NFLX": ["NETFLIX"], "DIS": ["DISNEY","WALT DISNEY"],
    "COIN": ["COINBASE"], "PLTR": ["PALANTIR"], "SHOP": ["SHOPIFY"],
    "CRWD": ["CROWDSTRIKE"], "PANW": ["PALO ALTO"], "SNOW": ["SNOWFLAKE"],
    "NOW": ["SERVICENOW"], "CRM": ["SALESFORCE"], "HOOD": ["ROBINHOOD"],
    "SOFI": ["SOFI TECHNOLOGIES"], "MSTR": ["MICROSTRATEGY"],
    "UBER": ["UBER TECHNOLOGIES"], "DASH": ["DOORDASH"],
    "ABNB": ["AIRBNB"], "DDOG": ["DATADOG"], "NET": ["CLOUDFLARE"],
}


# ── Yahoo RSS per ticker ──────────────────────────────────────────────────────

async def _fetch_yahoo_news(client: httpx.AsyncClient, ticker: str) -> list[dict]:
    try:
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline"
               f"?s={ticker}&region=US&lang=en-US")
        resp = await client.get(url, timeout=8, follow_redirects=True)
        if resp.status_code != 200:
            return []

        root  = ET.fromstring(resp.text)
        items = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=72)

        for item in root.findall(".//item")[:8]:
            title = item.findtext("title") or ""
            pub   = item.findtext("pubDate") or ""
            try:
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            except Exception:
                pass
            items.append({"title": title, "published": pub, "source": "yahoo_rss"})
            if len(items) >= 5:
                break
        return items
    except Exception:
        return []


# ── Catalyst NLP ──────────────────────────────────────────────────────────────

_CATALYST_PATTERNS = [
    (r"\b(beat|beats|topped|surpass|exceeded)\b.{0,40}\b(earn|eps|revenue|estimate)", "earnings_beat", 0.90, "bullish"),
    (r"\b(record|all.time).{0,20}\b(revenue|profit|sales|earnings)", "earnings_beat", 0.85, "bullish"),
    (r"\b(raised?|raise|raises|raising|upped?|increases?)\b.{0,30}\b(guidance|outlook|forecast)", "guidance_raise", 0.85, "bullish"),
    (r"\b(miss|misses|missed|below|disappointing)\b.{0,40}\b(earn|eps|revenue|estimate)", "earnings_miss", 0.80, "bearish"),
    (r"\b(cut|cuts|lowers?|reducing?|reduce|withdrew?)\b.{0,30}\b(guidance|outlook|forecast)", "guidance_cut", 0.80, "bearish"),
    (r"\bfda\b.{0,50}\b(approv|authoriz|clear|grant)", "fda_approval", 0.95, "bullish"),
    (r"\b(approv|authoriz|clear).{0,50}\bfda\b", "fda_approval", 0.95, "bullish"),
    (r"\bpdufa\b", "fda_catalyst", 0.80, "bullish"),
    (r"\b(nda|bla|anda)\b.{0,30}\b(approv|accept|grant)", "fda_approval", 0.90, "bullish"),
    (r"\bfda\b.{0,50}\b(reject|refus|complet.response|hold|delay)", "fda_rejection", 0.85, "bearish"),
    (r"\bclinical.trial.{0,30}\b(success|positive|met.{0,10}endpoint)", "clinical_trial", 0.80, "bullish"),
    (r"\bphase [23]\b.{0,50}\b(success|positive|met)", "clinical_trial", 0.80, "bullish"),
    (r"\b(awarded?|wins?|secures?|receives?)\b.{0,40}\b(contract|deal|agreement)\b.{0,30}\$([\d\.]+[bm])", "contract", 0.80, "bullish"),
    (r"\b(government|pentagon|dod|dhs|nasa|army|navy|air.force)\b.{0,50}\b(contract|deal|award)", "gov_contract", 0.85, "bullish"),
    (r"\b(department.of.defense|u\.s\. government|federal)\b.{0,50}\b(award|select|choose)", "gov_contract", 0.85, "bullish"),
    (r"\b(partnership|collaboration|joint.venture|licensing.agreement)\b", "partnership", 0.70, "bullish"),
    (r"\b(acqui|merger|buyout|takeover|bid)\b", "ma_activity", 0.75, "bullish"),
    (r"\b(upgrade|upgraded|rais.{0,5}target|rais.{0,5}price.target)\b", "analyst_upgrade", 0.65, "bullish"),
    (r"\b(buy|strong.buy|outperform|overweight)\b.{0,30}\b(initiat|reiterat|maintain)\b", "analyst_upgrade", 0.60, "bullish"),
    (r"\b(downgrade|lower.{0,5}target|cut.{0,5}target|underperform|underweight)\b", "analyst_downgrade", 0.60, "bearish"),
    (r"\b(fed|federal.reserve|fomc)\b.{0,50}\b(cut|lower|pivot|pause)\b.{0,30}\brate", "macro_positive", 0.70, "bullish"),
    (r"\b(cpi|inflation|pce)\b.{0,30}\b(cool|fell?|low|below.expect)", "macro_positive", 0.65, "bullish"),
    (r"\b(fed|federal.reserve)\b.{0,50}\b(hike|raise|higher)\b.{0,30}\brate", "macro_negative", 0.65, "bearish"),
    (r"\b(buyback|repurchas|share.repurchas)\b.{0,30}\$([\d\.]+[bm])", "buyback", 0.70, "bullish"),
    (r"\b(dividend|divid).{0,30}\b(increas|rais|special|extra)\b", "dividend_raise", 0.65, "bullish"),
    (r"\b(launch|release|unveil|announc).{0,40}\b(product|chip|model|platform|service|drug)", "product_launch", 0.65, "bullish"),
    (r"\b(generative.?ai|ai.chip|llm|gpu|accelerat)\b.{0,30}\b(partner|adopt|integrat|deploy)", "ai_catalyst", 0.70, "bullish"),
]

import re as _re
_COMPILED = [
    (_re.compile(pat, _re.IGNORECASE), ctype, score, sentiment)
    for pat, ctype, score, sentiment in _CATALYST_PATTERNS
]


def _classify_catalyst(text: str, source: str, pub_time_str: str) -> Optional[dict]:
    recency_hours = 24.0
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(pub_time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        recency_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        pass

    best_match = None
    for pattern, ctype, base_score, sentiment in _COMPILED:
        if pattern.search(text):
            recency_bonus = max(0, (48 - recency_hours) / 48) * 0.15
            score = min(1.0, base_score + recency_bonus)
            if best_match is None or score > best_match[2]:
                best_match = (ctype, sentiment, score)

    if best_match is None:
        return None

    ctype, sentiment, score = best_match
    return {
        "type":          ctype,
        "headline":      text[:120] + ("…" if len(text) > 120 else ""),
        "sentiment":     sentiment,
        "source":        source,
        "recency_hours": round(recency_hours, 1),
        "raw_score":     round(score, 3),
    }


def _catalyst_score(catalysts: list[dict]) -> int:
    if not catalysts:
        return 0
    bullish = [c for c in catalysts if c["sentiment"] == "bullish"]
    bearish = [c for c in catalysts if c["sentiment"] == "bearish"]
    bull = max((c["raw_score"] for c in bullish), default=0.0)
    bear = max((c["raw_score"] for c in bearish), default=0.0)
    if len(bullish) > 1:
        bull = min(1.0, bull + sum(c["raw_score"] * 0.2 for c in bullish[1:]))
    net = bull - bear * 0.8
    return max(0, min(100, int(net * 100)))


# ── Earnings catalyst score bonus ────────────────────────────────────────────

def _earnings_bonus(ticker: str, earnings_map: dict[str, dict]) -> tuple[int, Optional[dict]]:
    """
    Bonus de score si la empresa reporta earnings esta semana.
    Earnings son el catalizador más potente para movimiento de precio.
    """
    info = earnings_map.get(ticker)
    if not info:
        return 0, None

    days_until = info.get("days_until", 7)

    # Cuanto más cerca el earnings, mayor el bonus
    if days_until == 0:
        bonus = 35  # hoy
    elif days_until == 1:
        bonus = 25  # mañana
    elif days_until <= 3:
        bonus = 15  # esta semana
    else:
        bonus = 8   # la próxima semana

    catalyst = {
        "type":          "earnings_upcoming",
        "headline":      f"Earnings report en {days_until} día{'s' if days_until != 1 else ''} ({info['date']})",
        "sentiment":     "bullish",
        "source":        "yahoo_earnings_calendar",
        "recency_hours": 0,
        "raw_score":     bonus / 100,
    }
    return bonus, catalyst


# ── Technical scoring ─────────────────────────────────────────────────────────

def _technical_score(df: pd.DataFrame) -> tuple[int, list[str], dict]:
    import pandas_ta as ta
    if len(df) < 20:
        return 0, [], {}

    closes  = df["Close"]
    volumes = df["Volume"]
    score   = 0
    signals: list[str] = []
    metrics: dict = {}

    # RSI
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
    elif rsi < 50:
        score -= 5

    # Trend
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

    # MACD
    macd_bullish = False
    try:
        macd_df = ta.macd(closes, fast=12, slow=26, signal=9)
        if macd_df is not None and len(macd_df) >= 2:
            mcol = next((c for c in macd_df.columns if c.startswith("MACD_12")), None)
            scol = next((c for c in macd_df.columns if c.startswith("MACDs_")), None)
            if mcol and scol:
                m_now, m_prev = float(macd_df[mcol].iloc[-1]), float(macd_df[mcol].iloc[-2])
                s_now, s_prev = float(macd_df[scol].iloc[-1]), float(macd_df[scol].iloc[-2])
                if m_prev < s_prev and m_now > s_now:
                    macd_bullish = True; score += 15; signals.append("MACD cruce alcista")
                elif m_now > s_now:
                    score += 5
                elif m_prev > s_prev and m_now < s_now:
                    score -= 15
    except Exception:
        pass
    metrics["macd_bullish"] = macd_bullish

    # Price momentum (volatility-adjusted)
    change_5d  = float((closes.iloc[-1] - closes.iloc[-6]) / (closes.iloc[-6] + 1e-9) * 100) if len(closes) >= 6 else 0
    change_1d  = float((closes.iloc[-1] - closes.iloc[-2]) / (closes.iloc[-2] + 1e-9) * 100) if len(closes) >= 2 else 0
    vol_20d    = float(closes.pct_change().rolling(20).std().iloc[-1] * 100) if len(closes) >= 20 else 1.0
    # Volatility-adjusted return: how many standard deviations moved
    vol_adj_5d = change_5d / (vol_20d * np.sqrt(5) + 1e-9)
    metrics["change_1d"] = round(change_1d, 2)
    metrics["change_5d"] = round(change_5d, 2)
    metrics["vol_adj_5d"] = round(vol_adj_5d, 2)

    if change_5d > 0:
        score += min(int(change_5d * 3), 25)
        if change_5d > 3:
            signals.append(f"+{change_5d:.1f}% últimos 5 días")
    else:
        score += max(int(change_5d * 2), -20)

    # Volume anomaly (vs 20d average)
    avg_vol   = float(volumes.iloc[-20:].mean()) if len(volumes) >= 20 else float(volumes.mean())
    today_vol = float(volumes.iloc[-1])
    vol_ratio = today_vol / (avg_vol + 1)
    metrics["volume_ratio"] = round(vol_ratio, 2)

    if vol_ratio > 3.0:
        score += 25; signals.append(f"Volumen {vol_ratio:.1f}x (anomalía institucional)")
    elif vol_ratio > 2.0:
        score += 18; signals.append(f"Volumen {vol_ratio:.1f}x la media")
    elif vol_ratio > 1.5:
        score += 10
    elif vol_ratio < 0.5:
        score -= 5

    # Bollinger
    try:
        bb    = ta.bbands(closes, length=20, std=2)
        if bb is not None:
            bbu_col = next((c for c in bb.columns if c.startswith("BBU_")), None)
            if bbu_col:
                bbu = float(bb[bbu_col].iloc[-1])
                pct_from_upper = (float(closes.iloc[-1]) - bbu) / (bbu + 1e-9) * 100
                if -3 < pct_from_upper < 2:
                    score += 10; signals.append("Cerca de ruptura Bollinger")
                elif pct_from_upper > 5:
                    score -= 10
    except Exception:
        pass

    if trend == "bearish":
        score -= 25

    return max(0, min(100, score)), signals, metrics


def _momentum_score(metrics: dict) -> int:
    score   = 0
    c1      = metrics.get("change_1d", 0)
    c5      = metrics.get("change_5d", 0)
    vr      = metrics.get("volume_ratio", 1.0)
    vol_adj = metrics.get("vol_adj_5d", 0)

    # Price action
    if c1 > 0: score += min(int(c1 * 8), 30)
    else:       score += max(int(c1 * 5), -25)

    if c5 > 0: score += min(int(c5 * 3), 25)
    else:       score += max(int(c5 * 2), -20)

    # Volatility-adjusted (más fiable que retorno bruto)
    if vol_adj > 0.5:   score += 15
    elif vol_adj > 0.2: score += 8
    elif vol_adj < -0.5: score -= 15

    # Volume confirmation
    if vr > 1.5 and (c1 > 0 or c5 > 0):
        score += min(int((vr - 1) * 10), 25)

    return max(0, min(100, score))


# ── Main scan ─────────────────────────────────────────────────────────────────

async def run_opportunity_scan(max_results: int = 10) -> dict:
    cached = _cached("full_scan")
    if cached:
        logger.info("Returning cached scan")
        return cached  # type: ignore

    scan_start = time.time()
    logger.info("Starting full opportunity scan v2")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": "StockLens/2.0 research@stocklens.app"},
        follow_redirects=True,
    ) as client:

        # Layer 0: Universe building (parallel)
        sp500_task   = asyncio.create_task(_fetch_sp500_wikipedia(client))
        finviz_task  = asyncio.create_task(_fetch_finviz_movers(client))
        earnings_task = asyncio.create_task(_fetch_earnings_this_week(client))
        edgar_task   = asyncio.create_task(_fetch_edgar_8k(client))

        sp500, finviz, earnings_map, edgar_map = await asyncio.gather(
            sp500_task, finviz_task, earnings_task, edgar_task,
            return_exceptions=True
        )

        # Handle exceptions gracefully
        sp500       = sp500 if isinstance(sp500, list) else []
        finviz      = finviz if isinstance(finviz, list) else []
        earnings_map = earnings_map if isinstance(earnings_map, dict) else {}
        edgar_map   = edgar_map if isinstance(edgar_map, dict) else {}

        universe = _build_universe(sp500, finviz)
        logger.info("Universe: %d tickers | earnings this week: %d | edgar filings: %d",
                    len(universe), len(earnings_map), len(edgar_map))

        # Layer 1: Yahoo news for priority tickers
        # Priority: Finviz movers + earnings this week + EDGAR filings
        priority = list(set(finviz[:80]) | set(earnings_map.keys()) | set(edgar_map.keys()))
        priority = [t for t in priority if t in set(universe)][:120]

        sem = asyncio.Semaphore(12)
        async def fetch_news_limited(ticker: str):
            async with sem:
                return ticker, await _fetch_yahoo_news(client, ticker)

        news_results = await asyncio.gather(
            *[fetch_news_limited(t) for t in priority],
            return_exceptions=True,
        )
        news_map: dict[str, list[dict]] = {}
        for result in news_results:
            if isinstance(result, Exception):
                continue
            ticker, items = result
            if items:
                news_map[ticker] = items

    # Layer 2: Price data batch download
    logger.info("Downloading price data for %d tickers", len(universe))
    try:
        raw = yf.download(
            universe, period="3mo", interval="1d",
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
    except Exception as e:
        logger.error("yfinance batch download failed: %s", e)
        raw = None

    # Layer 3: Score each ticker
    candidates = []

    for ticker in universe:
        try:
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

            # Catalysts from news
            catalysts: list[dict] = []
            for item in news_map.get(ticker, []):
                cat = _classify_catalyst(item["title"], item["source"], item["published"])
                if cat:
                    catalysts.append(cat)
            for headline in edgar_map.get(ticker, []):
                cat = _classify_catalyst(headline, "sec_edgar", "")
                if cat:
                    catalysts.append(cat)

            # Earnings bonus
            earn_bonus, earn_catalyst = _earnings_bonus(ticker, earnings_map)
            if earn_catalyst:
                catalysts.append(earn_catalyst)

            # Scores
            cat_score  = _catalyst_score(catalysts) + earn_bonus
            cat_score  = min(100, cat_score)
            tech_score, tech_signals, metrics = _technical_score(df)
            mom_score  = _momentum_score(metrics)

            # Composite: catalyst-driven scanner
            composite = int(cat_score * 0.45 + tech_score * 0.30 + mom_score * 0.25)

            # Filter: must have catalyst OR exceptional technicals
            has_catalyst      = cat_score >= 30
            has_earnings      = ticker in earnings_map
            strong_technical  = tech_score >= 60 and mom_score >= 50

            if not (has_catalyst or has_earnings or strong_technical):
                continue

            candidates.append({
                "ticker":       ticker,
                "current_price": round(float(df["Close"].dropna().iloc[-1]), 2),
                "scores": {
                    "catalyst":  min(100, cat_score),
                    "technical": min(100, tech_score),
                    "momentum":  min(100, mom_score),
                    "composite": min(100, composite),
                },
                "catalysts":    catalysts,
                "technicals": {
                    "signals":        tech_signals,
                    "rsi":            metrics.get("rsi", 50.0),
                    "trend":          metrics.get("trend", "neutral"),
                    "macd_bullish":   metrics.get("macd_bullish", False),
                    "change_pct_1d":  metrics.get("change_1d", 0.0),
                    "change_pct_5d":  metrics.get("change_5d", 0.0),
                    "volume_ratio":   metrics.get("volume_ratio", 1.0),
                    "vol_adj_5d":     metrics.get("vol_adj_5d", 0.0),
                },
                "flags": {
                    "has_sec_filing":    ticker in edgar_map,
                    "has_earnings_week": ticker in earnings_map,
                    "earnings_days":     earnings_map.get(ticker, {}).get("days_until"),
                    "earnings_date":     earnings_map.get(ticker, {}).get("date"),
                },
            })

        except Exception as e:
            logger.debug("Error processing %s: %s", ticker, e)
            continue

    candidates.sort(key=lambda c: c["scores"]["composite"], reverse=True)
    top = candidates[:max_results]

    result = {
        "opportunities":     top,
        "scanned_at":        datetime.now(timezone.utc).isoformat(),
        "universe_size":     len(universe),
        "sp500_size":        len(sp500),
        "finviz_movers":     len(finviz),
        "earnings_this_week": len(earnings_map),
        "candidates_found":  len(candidates),
        "scan_duration_s":   round(time.time() - scan_start, 1),
        "sources":           ["wikipedia_sp500", "finviz_screener", "sec_edgar_8k",
                              "yahoo_earnings_calendar", "yahoo_rss", "yfinance_technical"],
        "cache_ttl_min":     CACHE_TTL["full_scan"] // 60,
        "disclaimer": (
            "Scanner multi-señal v2 — S&P500 dinámico, earnings calendar, "
            "SEC EDGAR 8-K, Yahoo RSS, análisis técnico. "
            "No constituye consejo de inversión."
        ),
    }

    _set_cache("full_scan", result)
    logger.info("Scan v2 complete: %d candidates from %d tickers in %.1fs",
                len(candidates), len(universe), time.time() - scan_start)
    return result