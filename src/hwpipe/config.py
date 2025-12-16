"""Configuration helpers for the handwriting pipeline."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Pipeline configuration loaded from environment."""

    input_dir: Path = Field(default=Path("data/in"))
    normalized_dir: Path = Field(default=Path("data/normalized"))
    out_raw_dir: Path = Field(default=Path("data/out_raw"))
    out_flags_dir: Path = Field(default=Path("data/out_flags"))
    out_final_dir: Path = Field(default=Path("data/out_final"))
    manifest_dir: Path = Field(default=Path("data/manifests"))

    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def ensure_dirs(self) -> None:
        for folder in [
            self.input_dir,
            self.normalized_dir,
            self.out_raw_dir,
            self.out_flags_dir,
            self.out_final_dir,
            self.manifest_dir,
        ]:
            folder.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
