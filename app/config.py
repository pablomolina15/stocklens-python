import json
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    cors_origins: List[str] = [
        "http://localhost:3000",
        "https://localhost:3000",
        "https://investment-platform-lilac.vercel.app",
    ]

    supabase_url: str = ""
    supabase_key: str = ""

    cache_ttl_prices: int = 3600
    cache_ttl_fundamentals: int = 86400
    max_retries: int = 3
    retry_delay: float = 1.0

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    # Strip trailing slashes from each origin
                    return [o.rstrip("/") for o in parsed]
            except (json.JSONDecodeError, TypeError):
                pass
            return [o.strip().rstrip("/") for o in v.split(",") if o.strip()]
        if isinstance(v, list):
            return [o.rstrip("/") for o in v]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
