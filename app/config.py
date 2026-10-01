from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "production"  # anything but "production" also serves /docs
    log_level: str = "INFO"
    log_format: str = "json"  # json for production log collectors, text for readable local output

    # Auth: comma separated list of accepted keys. Empty => auth disabled (local dev only).
    api_keys: str = ""
    cors_origins: str = ""  # comma separated, e.g. http://localhost:3000

    max_upload_mb: int = 15
    max_batch_files: int = 10
    max_pdf_pages: int = 10
    max_image_pixels: int = 60_000_000

    # OCR
    tesseract_lang: str = "eng"
    tesseract_psm: int = 6
    ocr_timeout_s: int = 60
    libreoffice_timeout_s: int = 60
    concurrency: int = 2  # documents processed in parallel per API process
    # Ambiguous dates like 04/05/2020: "MDY" (US, matches the dataset) or "DMY"
    date_order: str = "MDY"
    review_threshold: float = 0.80

    # LLM fallback (Gemini). Off unless a key is set. Quota defaults match the key's limits.
    gemini_api_key: str = ""
    gemini_base_url: str = ""  # tests only: point the SDK at a fake server
    llm_mode: str = "auto"  # auto = only when the rules result looks unreliable | always | off
    llm_model: str = "gemini-3.5-flash-lite"  # gemini-2.5-flash-lite is closed to new users (404) as of 2026-10
    llm_temperature: float = 0.0
    llm_timeout_s: int = 30
    llm_rpm: int = 10  # requests per minute (rolling 60 s window)
    llm_rpd: int = 20  # requests per day (resets at midnight Pacific, like Google's quota)
    llm_state_dir: str = "/tmp/invoice-llm"  # shared by all worker processes: quota counter + result cache

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
