"""
Unified LLM client supporting any OpenAI-compatible API endpoint
(OpenAI, Together, DeepSeek, local vLLM, etc.).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    retry_delay: float = 2.0


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completions."""

    def __init__(self, config: LLMConfig):
        self.cfg = config
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )
        self.total_calls: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Call the LLM and return the generated text.
        Retries up to `max_retries` times on failure.
        """
        t = temperature if temperature is not None else self.cfg.temperature
        n = max_tokens if max_tokens is not None else self.cfg.max_tokens

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=t,
                    max_tokens=n,
                )
                self.total_calls += 1
                if resp.usage:
                    self.total_tokens_in += resp.usage.prompt_tokens
                    self.total_tokens_out += resp.usage.completion_tokens
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_exc = exc
                wait = self.cfg.retry_delay * (2 ** attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    self.cfg.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} attempts") from last_exc

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def chat_n(
        self,
        messages: list[dict],
        n: int,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> list[str]:
        """
        Generate `n` independent completions for the same prompt.

        Uses the API's native `n` parameter (one round-trip) when the server
        supports it; falls back to `n` sequential calls otherwise.  All `n`
        completions use the same `temperature`.
        """
        tok = max_tokens if max_tokens is not None else self.cfg.max_tokens

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=tok,
                    n=n,
                )
                self.total_calls += 1
                if resp.usage:
                    self.total_tokens_in  += resp.usage.prompt_tokens
                    self.total_tokens_out += resp.usage.completion_tokens
                return [c.message.content or "" for c in resp.choices]
            except Exception as exc:
                last_exc = exc
                wait = self.cfg.retry_delay * (2 ** attempt)
                logger.warning(
                    "chat_n failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"chat_n failed after {self.cfg.max_retries} attempts") from last_exc

    def system_user(self, system: str, user: str, **kwargs) -> str:
        """Shortcut: single system + user turn."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.chat(messages, **kwargs)

    def stats(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
        }
