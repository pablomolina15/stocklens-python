"""
StockLens — Python Microservice
FastAPI + yfinance + pandas-ta + scikit-learn

Endpoints:
  GET  /                              → health check
  GET  /health                        → health check detallado
  GET  /analyze/technical/{ticker}    → precios + indicadores calculados
  GET  /analyze/fundamental/{ticker}  → métricas financieras + value score
  POST /predict/random-forest/{ticker}→ predicción ML (Fase 2)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routers import technical, fundamental, ml
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 StockLens Python Service starting up")
    logger.info(f"   CORS origins: {settings.cors_origins}")
    logger.info(f"   Supabase:     {'✓ configured' if settings.supabase_url else '✗ not configured'}")
    yield
    logger.info("🛑 StockLens Python Service shutting down")


app = FastAPI(
    title="StockLens API",
    description="Microservicio de análisis financiero — yfinance + pandas-ta + scikit-learn",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(technical.router,    prefix="/analyze",  tags=["Technical"])
app.include_router(fundamental.router,  prefix="/analyze",  tags=["Fundamental"])
app.include_router(ml.router,           prefix="/predict",  tags=["ML"])


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "stocklens-python",
        "version": "1.0.0",
        "supabase": "configured" if settings.supabase_url else "not configured",
    }
