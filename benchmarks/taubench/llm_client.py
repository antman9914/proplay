"""
LLM client for τ-Bench: extends the shared LLMClient with tool-calling support.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# memory_planning_demo deps moved into proplay package

from proplay.llm import LLMClient, LLMConfig  # noqa: F401 (re-exported)
from openai import OpenAI
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TauBenchLLMClient(LLMClient):
    """
    Extends LLMClient with chat_with_tools() for OpenAI-format function calling.
    All other methods (chat, stats, etc.) are inherited unchanged.
    """

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int | None = None,
    ) -> tuple[str | None, list[dict]]:
        """
        Call the LLM with tool definitions.

        Returns (content, tool_calls) where:
          content    — str or None (text response when no tools called)
          tool_calls — list of {"id": str, "name": str, "arguments": dict}
        """
        n = max_tokens or self.cfg.max_tokens

        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    tools=tools or [],
                    tool_choice="auto" if tools else "none",
                    temperature=self.cfg.temperature,
                    max_tokens=n,
                )
                self.total_calls += 1
                if resp.usage:
                    self.total_tokens_in += resp.usage.prompt_tokens
                    self.total_tokens_out += resp.usage.completion_tokens

                msg = resp.choices[0].message
                content = msg.content  # may be None when tool_calls present

                tool_calls: list[dict] = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": args,
                        })
                return content, tool_calls

            except Exception as exc:
                last_exc = exc
                wait = self.cfg.retry_delay * (2 ** attempt)
                logger.warning(
                    "chat_with_tools attempt %d/%d failed: %s — retrying in %.1fs",
                    attempt + 1, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"chat_with_tools failed after {self.cfg.max_retries} attempts"
        ) from last_exc
