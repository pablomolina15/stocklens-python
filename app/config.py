import json
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    # CORS — en Railway añade tu dominio Vercel aquí
    cors_origins: List[str] = [
        "http://localhost:3000",
        "https://localhost:3000",
        "https://investment-platform-lilac.vercel.app/"
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

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            # Try JSON first, then fall back to comma-separated
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            # Treat as comma-separated string
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
