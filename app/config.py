import json
import os
from typing import List


def _parse_cors(raw: str | None) -> List[str]:
    """Parse CORS_ORIGINS from env var — handles empty, JSON array, or comma-separated."""
    default = [
        "http://localhost:3000",
        "https://investment-platform-lilac.vercel.app",
    ]
    if not raw or not raw.strip():
        return default
    raw = raw.strip()
    # JSON array: ["url1","url2"]
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            result = [o.rstrip("/") for o in parsed if o and o.strip()]
            return result if result else default
        except json.JSONDecodeError:
            pass
    # Comma-separated: url1,url2
    result = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    return result if result else default


class Settings:
    """Simple settings class — avoids pydantic-settings JSON pre-parsing bug with List fields."""

    def __init__(self):
        self.cors_origins: List[str] = _parse_cors(os.environ.get("CORS_ORIGINS", ""))
        self.supabase_url: str = os.environ.get("SUPABASE_URL", "")
        self.supabase_key: str = os.environ.get("SUPABASE_KEY", "")
        self.cache_ttl_prices: int = int(os.environ.get("CACHE_TTL_PRICES", "3600"))
        self.cache_ttl_fundamentals: int = int(os.environ.get("CACHE_TTL_FUNDAMENTALS", "86400"))
        self.max_retries: int = int(os.environ.get("MAX_RETRIES", "3"))
        self.retry_delay: float = float(os.environ.get("RETRY_DELAY", "1.0"))


settings = Settings()
