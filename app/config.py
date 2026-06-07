from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # CORS — en Railway añade tu dominio Vercel aquí
    cors_origins: List[str] = [
        "http://localhost:3000",
        "https://localhost:3000",
        # Añade aquí tu dominio de Vercel, ej:
        # "https://investment-platform.vercel.app",
        # "https://tu-dominio.vercel.app",
    ]

    # Supabase (opcional — para caché desde Python)
    supabase_url: str = ""
    supabase_key: str = ""

    # Cache TTL en segundos
    cache_ttl_prices: int = 3600        # 1 hora
    cache_ttl_fundamentals: int = 86400  # 24 horas

    # yfinance rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
