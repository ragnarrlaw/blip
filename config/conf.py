"""
config -- infrastructure settings read from .env once, typed and validated.

What belongs here: things that describe the *deployment* -- API keys, where files live, which
database to use, which frontends may call the API. What does NOT belong here: anything a user
chooses per grading run (model, RAG on/off, k, temperature, routing threshold, chunking). Those
arrive from the frontend as request parameters and are stored with the run; the values below
are only their defaults.

    from config.conf import get_config
    cfg = get_config()
    cfg.results_dir("SCS1303")      -> Path(".../out/scs1303/results")

The .env file is located relative to the project root, not the current working directory, so
`uvicorn`, `celery` and `pytest` started from any directory see the same settings. Override the
location with BLIP_ENV_FILE.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path(os.environ.get("BLIP_ENV_FILE", PROJECT_ROOT / ".env"))


class Config(BaseSettings):
    # env names map case-insensitively: GEMINI_API_KEY -> gemini_api_key, etc.
    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # --- model providers (secrets: never sent to or accepted from the frontend) ---
    gemini_api_key: str = ""
    gemini_model_name: str = ""
    xai_api_key: str = ""
    grok_model_name: str = ""
    deepseek_api_key: str = ""
    deepseek_model_name: str = ""

    # --- vle ---
    vle_username: str = ""
    vle_password: str = ""
    vle_base_url: str = ""
    vle_service: str = "moodle_mobile_app"
    vle_default_submissions_count: int = 10000

    # --- database ---
    # Any SQLAlchemy URL works.
    # Postgres: one-line change (postgresql+psycopg://user:pass@host/blip).
    blip_database_url: str = f"sqlite:///{PROJECT_ROOT / 'var' / '.db' / 'blip.db'}"
    # How long a writer waits for the lock before failing. Writers queue behind each other; with
    # the BEGIN IMMEDIATE handling in db.engine they never fail fast.
    blip_sqlite_busy_timeout_ms: int = 30_000
    # FULL survives power loss without losing committed transactions; NORMAL is faster and
    # survives application crashes but can lose the last few commits on power loss.
    blip_sqlite_synchronous: Literal["NORMAL", "FULL"] = "FULL"
    blip_sql_echo: bool = False

    # --- api ---
    blip_cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- blip-b defaults (the frontend can override every one of these per run) ---
    # Confidence is on [0, 1] everywhere in the pipeline. "70" is accepted and read as 0.70.
    blip_confidence_threshold: float = 0.70
    blip_course_content_upload_dir: str = str(PROJECT_ROOT / "var" / "out" / "{course_tag}" / "course_content")
    blip_student_submissions_upload_dir: str = str(PROJECT_ROOT / "var" / "out" / "{course_tag}" / "submissions")
    blip_output_dir: str = str(PROJECT_ROOT / "var" / "out")
    blip_results_dir: str = str(PROJECT_ROOT / "var" / "out" / "{course_tag}" / "results")
    blip_stat_dir: str = str(PROJECT_ROOT / "var" / "out" / "{course_tag}" / "stats")
    blip_chromadb_dir: str = str(PROJECT_ROOT / "var" / ".vector_store" / "{course_tag}")
    blip_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    blip_embed_max_tokens: int = 256
    blip_chunk_size: int = 1500
    blip_chunk_overlap: int = 150
    blip_chunker_separators: list[Literal["\n\n", "\n", " ", ""]] = ["\n\n", "\n", " ", ""]

    @field_validator("blip_confidence_threshold")
    @classmethod
    def _threshold_as_fraction(cls, value: float) -> float:
        if value > 1:
            value = value / 100.0
        if not 0 <= value <= 1:
            raise ValueError("blip_confidence_threshold must be in [0, 1] (or a percentage in [0, 100]).")
        return value

    @model_validator(mode="after")
    def _warn_on_truncated_chunks(self) -> "Config":
        # chunk_size counts characters; the embedder's limit counts tokens (~4 characters each for
        # English). Anything past the limit is silently dropped before embedding, so retrieval
        # only "sees" the start of each chunk while the grader is handed all of it.
        approx_tokens = self.blip_chunk_size / 4
        if approx_tokens > self.blip_embed_max_tokens:
            logger.warning(
                "BLIP_CHUNK_SIZE=%d characters is roughly %d tokens, above BLIP_EMBED_MAX_TOKENS=%d. "
                "The embedder truncates each chunk, so the tail of every chunk is invisible to retrieval.",
                self.blip_chunk_size, approx_tokens, self.blip_embed_max_tokens,
            )
        return self

    # ------------------------------------------------------------------ path templates
    @staticmethod
    def course_slug(course_tag: str) -> str:
        return course_tag.strip().replace(" ", "_").lower()

    @classmethod
    def _fill(cls, template: str, course_tag: str) -> Path:
        return Path(template.format(course_tag=cls.course_slug(course_tag)))

    def content_out(self, course_tag: str) -> Path:
        return self._fill(self.blip_output_dir, course_tag)

    def submissions_out(self, course_tag: str) -> Path:
        return self._fill(self.blip_student_submissions_upload_dir, course_tag)

    def content_upload_dir(self, course_tag: str) -> Path:
        return self._fill(self.blip_course_content_upload_dir, course_tag)

    def results_dir(self, course_tag: str) -> Path:
        return self._fill(self.blip_results_dir, course_tag)

    def stat_dir(self, course_tag: str) -> Path:
        return self._fill(self.blip_stat_dir, course_tag)

    def chromadb_dir(self, course_tag: str) -> Path:
        return self._fill(self.blip_chromadb_dir, course_tag)

    @classmethod
    def chroma_collection(cls, course_tag: str) -> str:
        return f"chroma_content_{cls.course_slug(course_tag)}"


@lru_cache
def get_config() -> Config:
    """The process-wide settings. Tests call get_config.cache_clear() after changing the env."""
    return Config()


# Kept for existing imports (`from config.conf import config`).
config = get_config()
