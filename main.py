"""
StockLens — Python Microservice
Imports pesados (sklearn, pandas_ta) en lazy mode para arranque rápido
y que Railway healthcheck pase en < 30s.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_warmed_up = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _warmed_up
    logger.info("🚀 StockLens Python Service starting")
    logger.info("   CORS: %s", settings.cors_origins)
    _warmed_up = True
    yield
    logger.info("🛑 StockLens shutting down")


app = FastAPI(
    title="StockLens API",
    description="Análisis técnico, fundamental, ML y scanner multi-señal",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health():
    return {
        "status": "ok",
        "service": "stocklens-python",
        "version": "2.0.0",
        "warmed_up": _warmed_up,
        "supabase": "configured" if settings.supabase_url else "not configured",
    }


# Lazy imports after health endpoint
from app.routers import technical, fundamental, ml  # noqa: E402
from app.routers import scanner                      # noqa: E402

app.include_router(technical.router,   prefix="/analyze", tags=["Technical"])
app.include_router(fundamental.router, prefix="/analyze", tags=["Fundamental"])
app.include_router(ml.router,          prefix="/predict", tags=["ML"])
app.include_router(scanner.router,     prefix="/scan",    tags=["Scanner"])  # ← NEW
