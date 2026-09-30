"""OpenAI answer generation. Keys come from an override, .env / environment, or Streamlit Secrets."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from utils.prompts import SYSTEM_PROMPT, build_user_prompt

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
ROOT = Path(__file__).resolve().parent.parent


class LLMError(Exception):
    """User-facing LLM failure."""


class MissingAPIKeyError(LLMError):
    pass


def resolve_api_key(override: Optional[str] = None) -> Optional[str]:
    """Priority: sidebar override > environment/.env > Streamlit Secrets. Never hard-coded."""
    if override and override.strip():
        return override.strip()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        import streamlit as st

        secret = st.secrets.get("OPENAI_API_KEY")
    except Exception:  # no secrets file / not running under Streamlit
        secret = None
    return secret.strip() if isinstance(secret, str) and secret.strip() else None


def get_model_name(api_key: Optional[str] = None) -> str:
    load_dotenv(ROOT / ".env")
    name = os.getenv("OPENAI_MODEL") or os.getenv("GROQ_MODEL")
    if not name:
        try:
            import streamlit as st

            name = st.secrets.get("OPENAI_MODEL") or st.secrets.get("GROQ_MODEL")
        except Exception:
            name = None
    if name and name.strip():
        return name.strip()
    key = resolve_api_key(api_key)
    if key and key.startswith("gsk_"):
        return DEFAULT_GROQ_MODEL
    return DEFAULT_MODEL


def generate_answer(
    question: str,
    evidence: Sequence,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    client=None,
) -> str:
    """Answer from evidence only. `client` can be injected (tests)."""
    key = resolve_api_key(api_key)
    is_groq = bool(key and key.startswith("gsk_"))

    if client is None:
        if not key:
            raise MissingAPIKeyError(
                "No API key found. Add OPENAI_API_KEY or GROQ_API_KEY to .env (local) or Streamlit Secrets "
                "(deployment), or paste a key in the sidebar."
            )
        base_url = "https://api.groq.com/openai/v1" if is_groq else None
        client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=2)

    effective_model = model or get_model_name(api_key)
    if is_groq and (not model or model == DEFAULT_MODEL):
        effective_model = DEFAULT_GROQ_MODEL

    kwargs = dict(
        model=effective_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, evidence)},
        ],
        temperature=0,
    )
    provider_name = "Groq" if is_groq else "OpenAI"
    try:
        try:
            response = client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "temperature" not in str(exc).lower():
                raise
            kwargs.pop("temperature")  # some models only accept the default temperature
            response = client.chat.completions.create(**kwargs)
    except AuthenticationError as exc:
        raise LLMError(f"{provider_name} rejected the API key. Check that it is valid and active.") from exc
    except RateLimitError as exc:
        raise LLMError(f"{provider_name} rate limit or quota reached. Wait a moment or check billing.") from exc
    except APIConnectionError as exc:
        raise LLMError(f"Could not reach the {provider_name} API. Check your network connection.") from exc
    except APIError as exc:
        raise LLMError(f"{provider_name} API error: {exc}") from exc

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise LLMError("The model returned an empty answer. Please try again.")
    return text
