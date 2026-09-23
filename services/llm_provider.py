from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from config.conf import Config

logger = logging.getLogger("blip.llm_provider")

DEFAULT_REQUEST_TIMEOUT_S = 120.0

_BLOCKED_FINISH_REASONS = {
    "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII",
    "LANGUAGE", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION",
}

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMExecutionResult:
    parsed: Optional[BaseModel]
    raw_text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float


class LLMResponseError(Exception):
    """
    Raised when an API call succeeds at the transport layer,
    but cannot produce a usable grading or parsing result.
    """
    def __init__(self, message: str, *, category: str, retryable: bool, raw_text: str = ""):
        super().__init__(message)
        self.message = message
        self.category = category
        self.retryable = retryable
        self.raw_text = raw_text


class LLMProvider(ABC):
    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def generate_structured_response(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.0,
        model_override: Optional[str] = None,
    ) -> LLMExecutionResult:
        """Generates schema-validated JSON with telemetry extraction."""
        ...


class GeminiProvider(LLMProvider):
    def __init__(self, config: Config):
        super().__init__(config)
        if not self.config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiProvider.")

        self.request_timeout_s = float(
            getattr(self.config, "gemini_request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
        )
        self.default_model = self.config.gemini_model_name or "gemini-1.5-flash"

        self._client = genai.Client(
            api_key=self.config.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(self.request_timeout_s * 1000)),
        )

    def generate_structured_response(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: str = "",
        temperature: float = 0.0,
        model_override: Optional[str] = None,
    ) -> LLMExecutionResult:
        model = model_override or self.default_model
        start_time = time.perf_counter()

        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=response_schema,
        )
        if system_instruction:
            generation_config.system_instruction = system_instruction

        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=generation_config,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if response.parsed is None:
            raise self._classify_unusable_response(response)

        usage = getattr(response, "usage_metadata", None)
        in_tokens = getattr(usage, "prompt_token_count", 0) or 0
        out_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or (in_tokens + out_tokens)

        return LLMExecutionResult(
            parsed=response.parsed,
            raw_text=response.text or "",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _classify_unusable_response(response) -> LLMResponseError:
        try:
            raw_text = response.text or ""
        except Exception:
            raw_text = ""

        # Check prompt-level moderation block
        prompt_feedback = getattr(response, "prompt_feedback", None)
        block_reason = getattr(prompt_feedback, "block_reason", None)
        if block_reason and str(block_reason) != "BLOCKED_REASON_UNSPECIFIED":
            return LLMResponseError(
                f"Prompt blocked by API policy ({block_reason}).",
                category="blocked",
                retryable=False,
                raw_text=raw_text,
            )

        candidates = getattr(response, "candidates", None) or []
        finish_reason = None
        if candidates:
            finish_reason = getattr(candidates[0], "finish_reason", None)
            finish_reason = getattr(finish_reason, "name", str(finish_reason))

        if finish_reason in _BLOCKED_FINISH_REASONS:
            return LLMResponseError(
                f"Response withheld by safety filters (finish reason: {finish_reason}).",
                category="blocked",
                retryable=False,
                raw_text=raw_text,
            )

        if finish_reason == "MAX_TOKENS":
            return LLMResponseError(
                "Response cut off at output token limit.",
                category="truncated_output",
                retryable=False,
                raw_text=raw_text,
            )

        if not raw_text.strip():
            return LLMResponseError(
                f"API returned empty payload (finish reason: {finish_reason}).",
                category="empty_output",
                retryable=True,
                raw_text=raw_text,
            )

        return LLMResponseError(
            "Response could not be parsed into the expected JSON schema.",
            category="malformed_output",
            retryable=True,
            raw_text=raw_text,
        )