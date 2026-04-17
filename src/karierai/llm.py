from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from .config import get_settings

logger = logging.getLogger(__name__)

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None


_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


@dataclass(frozen=True)
class LLMResult:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw_response: Any = None


def llm_is_available() -> bool:
    settings = get_settings()
    return bool(settings.openai_api_key and ChatOpenAI is not None)


def require_llm() -> None:
    if llm_is_available():
        return
    raise RuntimeError("LLM utama belum dikonfigurasi. Set OPENAI_API_KEY terlebih dahulu.")


def build_chat_model(*, temperature: float = 0.2):
    require_llm()
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=temperature,
    )


def _extract_token_usage(response: Any) -> tuple[int, int]:
    metadata = getattr(response, "response_metadata", None) or {}
    usage = metadata.get("token_usage") or metadata.get("usage_metadata") or {}
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    return input_tokens, output_tokens


def normalize_llm_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def invoke_text(prompt: str, *, temperature: float = 0.2) -> LLMResult:
    response = build_chat_model(temperature=temperature).invoke(prompt)
    content = normalize_llm_content(getattr(response, "content", response))
    input_tokens, output_tokens = _extract_token_usage(response)
    return LLMResult(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw_response=response,
    )


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception as exc:
        logger.debug("Failed to parse JSON block from LLM output: %s", exc)
        return None
    return payload if isinstance(payload, dict) else None


def invoke_json(prompt: str, *, temperature: float = 0.0) -> tuple[dict[str, Any], LLMResult]:
    result = invoke_text(prompt, temperature=temperature)
    payload = extract_json_object(result.content)
    if payload is None:
        raise RuntimeError("Output LLM tidak valid karena bukan JSON object.")
    return payload, result
