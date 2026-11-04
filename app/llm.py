"""Thin wrapper over any OpenAI-compatible chat endpoint (DeepSeek, Qwen, Ollama...)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI

from app.config import get_settings

_THINK = re.compile(r"<think>.*?</think>", re.S)
_JSON = re.compile(r"\{.*\}", re.S)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature

    def chat(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        # Reasoning models (DeepSeek-R1, QwQ) may inline their chain of thought.
        return _THINK.sub("", text).strip()

    def chat_json(self, system: str, user: str) -> dict:
        text = self.chat(system + "\n只输出一个 JSON 对象，不要输出其他内容。", user)
        m = _JSON.search(text)
        if not m:
            raise ValueError(f"LLM did not return JSON: {text[:200]}")
        return json.loads(m.group(0))


@lru_cache
def get_llm() -> LLMClient | None:
    s = get_settings()
    if not s.llm_enabled:
        return None
    return LLMClient(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_temperature, s.llm_timeout)
