from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "production"
    log_level: str = "INFO"
    log_format: str = "json"

    api_keys: str = ""  # comma separated; empty disables auth (local dev only)
    cors_origins: str = ""

    max_upload_mb: int = 15
    max_batch_files: int = 10
    max_pdf_pages: int = 10
    max_image_pixels: int = 60_000_000

    tesseract_lang: str = "eng"
    tesseract_psm: int = 6
    ocr_timeout_s: int = 60
    libreoffice_timeout_s: int = 60
    concurrency: int = 2
    date_order: str = "MDY"
    review_threshold: float = 0.80

    gemini_api_key: str = ""
    gemini_base_url: str = ""  # tests only: point the SDK at a fake server
    llm_mode: str = "auto"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.0
    llm_timeout_s: int = 30
    llm_rpm: int = 10
    llm_rpd: int = 20
    llm_state_dir: str = "/tmp/invoice-llm"

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
