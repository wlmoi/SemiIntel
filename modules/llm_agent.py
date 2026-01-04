"""
LangChain-powered chat agent for SEMIINTEL.
Supports OpenAI-compatible providers (OpenAI, Qwen via DashScope) with streaming.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

try:  # Optional dependency for Gemini
    from google import genai  # type: ignore
except Exception:  # pragma: no cover - optional
    genai = None


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    model: str
    api_key_env: str
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None


class LLMChatAgent:
    """Simple streaming chat agent using LangChain ChatOpenAI or Gemini."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.config = self._provider_config(provider, model)
        self.system_prompt = system_prompt or (
            "You are SEMIINTEL's LLM assistant. Be concise, cite tools or datasets when helpful, "
            "and keep answers factual."
        )
        self.temperature = temperature

        if self.config.name == "gemini":
            if genai is None:
                raise RuntimeError("google-genai is not installed. Run `pip install -r requirements.txt`.")
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY for Gemini provider.")
            self.gemini_client = genai.Client(api_key=api_key)
            self.llm = None
        else:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key and self.config.name == "local":
                api_key = "nokey"  # Some OpenAI-compatible servers accept any non-empty string
            if not api_key:
                raise RuntimeError(
                    f"Missing API key for provider '{self.config.name}'. "
                    f"Set {self.config.api_key_env} or provide a compatible local endpoint."
                )

            resolved_base_url = base_url
            if not resolved_base_url and self.config.base_url_env:
                resolved_base_url = os.getenv(self.config.base_url_env, self.config.default_base_url)
            if self.config.name == "local" and not resolved_base_url:
                raise RuntimeError("Local provider requires a base URL (OpenAI-compatible).")

            self.llm = ChatOpenAI(
                api_key=api_key,
                model=self.config.model,
                base_url=resolved_base_url,
                temperature=temperature,
            )
            self.gemini_client = None

    @staticmethod
    def _provider_config(provider: str, override_model: Optional[str]) -> LLMProviderConfig:
        provider = provider.lower()
        if provider == "qwen":
            return LLMProviderConfig(
                name="qwen",
                model=override_model or "qwen2.5-14b-instruct",
                api_key_env="QWEN_API_KEY",
                base_url_env="QWEN_API_BASE",
                default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        if provider == "gemini":
            return LLMProviderConfig(
                name="gemini",
                model=override_model or "gemini-2.5-flash",
                api_key_env="GEMINI_API_KEY",
                base_url_env=None,
                default_base_url=None,
            )
        if provider == "local":
            return LLMProviderConfig(
                name="local",
                model=override_model or "local-model",
                api_key_env="LOCAL_LLM_API_KEY",
                base_url_env="LOCAL_LLM_BASE_URL",
                default_base_url=None,
            )
        # Default: OpenAI
        return LLMProviderConfig(
            name="openai",
            model=override_model or "gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            base_url_env=None,
            default_base_url=None,
        )

    def _build_messages(self, history: Sequence[dict], user_input: str):
        """Convert simple history dicts into LangChain message objects."""
        messages: List[object] = [SystemMessage(content=self.system_prompt)]
        for item in history:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=user_input))
        return messages

    def stream_response(self, user_input: str, history: Sequence[dict]) -> Generator[str, None, None]:
        """Yield streamed tokens for a user input given chat history."""
        if self.config.name == "gemini":
            if self.gemini_client is None:
                raise RuntimeError("Gemini client not initialized.")
            contents = []
            for item in history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    contents.append({"role": "user", "parts": [content]})
                elif role == "assistant":
                    contents.append({"role": "model", "parts": [content]})
            contents.append({"role": "user", "parts": [user_input]})

            stream = self.gemini_client.models.generate_content(
                model=self.config.model,
                contents=contents,
                system_instruction=self.system_prompt,
                stream=True,
                generation_config={"temperature": self.temperature},
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
            return

        messages = self._build_messages(history, user_input)
        for chunk in self.llm.stream(messages):
            text = chunk.content
            if text:
                yield text


__all__ = ["LLMChatAgent", "LLMProviderConfig"]
