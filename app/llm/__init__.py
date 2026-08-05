"""LLM backends: Gemini (cloud) → local OpenAI-compatible → offline heuristics."""

from .router import LLMRouter, LLMResponse

__all__ = ["LLMRouter", "LLMResponse"]
