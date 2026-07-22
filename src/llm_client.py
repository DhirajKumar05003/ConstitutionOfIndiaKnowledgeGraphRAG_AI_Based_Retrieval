"""
llm_client.py
==============
Thin wrapper around the Groq Cloud chat-completions API.

Kept separate from the agent logic so the rest of the codebase (and the
tests) depend on a small, mockable interface rather than the Groq SDK
directly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from groq import Groq

# Groq deprecated the llama-3.3-70b-versatile chat models; gpt-oss-120b is the
# current recommended general-purpose / reasoning model on GroqCloud.
# Override via the GROQ_MODEL env var if you want a different one
# (see https://console.groq.com/docs/models for the current catalogue).
DEFAULT_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict | None = None


class GroqLLMClient:
    """Minimal chat-completion client used by every agent in the workflow."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "No Groq API key found. Set GROQ_API_KEY in your environment "
                "or pass api_key= explicitly (get one at https://console.groq.com/keys)."
            )
        self._client = Groq(api_key=key)
        self.model = model

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResponse:
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0].message.content or ""
        usage = getattr(completion, "usage", None)
        usage_dict = usage.model_dump() if hasattr(usage, "model_dump") else None
        return LLMResponse(text=choice.strip(), model=self.model, usage=usage_dict)

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        default: dict,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> dict:
        """
        Chat call that expects a JSON object back (used by the Query
        Analyzer and Faithfulness Verifier stages). Falls back to `default`
        on any parse failure rather than raising — a malformed analyzer
        response should degrade the pipeline gracefully, not crash it.
        """
        try:
            resp = self.chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens, json_mode=True)
            return _safe_json_parse(resp.text, default)
        except Exception:
            return dict(default)


def _safe_json_parse(text: str, default: dict) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            merged = dict(default)
            merged.update(parsed)
            return merged
    except json.JSONDecodeError:
        pass
    return dict(default)
