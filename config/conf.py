"""
config — reads .env once, typed and validated, and resolves the {course_tag}/{version}
path templates.

    from config import config
    settings.content_out("ABC1234", "v1")   -> Path("out/ABC1234/v1")

.env is not automatically loaded, should be done prior to the import
"""

from __future__ import annotations
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # env names map case-insensitively: GEMINI_API_KEY -> gemini_api_key, etc.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- model config ---
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
    vle_course_content_out: str = "out/{course_tag}/{version}"
    vle_student_submissions_out: str = "out/{course_tag}/{version}"
    vle_default_submissions_count: int = 10000

    # --- blip-b ---
    blip_confidence_threshold: int = 70
    blip_content_upload_dir: str = "out/{course_tag}/{version}"
    blip_student_responses_dir: str = "out/{course_tag}/{version}"
    blip_output_dir: str = "out"
    blip_results_dir: str = "out/results"
    blip_stat_dir: str = "out/stats"
    blip_chromadb_dir: str = ".db/chroma"
    blip_sqlite_dir: str = ".db/sqlite"

    @staticmethod
    def _fill(template: str, course_tag: str, version: str) -> Path:
        return Path(template.format(course_tag=course_tag, version=version))

    def content_out(self, course_tag: str, version: str = "v1") -> Path:
        return self._fill(self.vle_course_content_out, course_tag, version)

    def submissions_out(self, course_tag: str, version: str = "v1") -> Path:
        return self._fill(self.vle_student_submissions_out, course_tag, version)

    def content_upload_dir(self, course_tag: str, version: str = "v1") -> Path:
        return self._fill(self.blip_content_upload_dir, course_tag, version)

    def responses_dir(self, course_tag: str, version: str = "v1") -> Path:
        return self._fill(self.blip_student_responses_dir, course_tag, version)

    def chroma_collection(self, course_tag: str) -> str:
        # isolates each course's RAG corpus in its own collection.
        return f"content_{course_tag}".lower()


config = Config()