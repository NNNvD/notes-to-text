"""Vision model transcription utilities."""
from __future__ import annotations

import base64
import hashlib
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import get_settings


SYSTEM_PROMPT = "You are a transcription engine. Transcribe handwritten text verbatim."
RULES_PROMPT = (
    "Preserve line breaks as seen.\n"
    "Do not correct spelling or grammar.\n"
    "If uncertain, write [unclear] (do not guess).\n"
    "If a token is partly legible, write the legible part + [unclear] (e.g., psycho[unclear]).\n"
    "Output only the transcription, no commentary."
)


def _load_image_base64(path: Path) -> str:
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/{path.suffix.lstrip('.').lower()};base64,{encoded}"


def prompt_hash() -> str:
    hasher = hashlib.sha256()
    hasher.update(SYSTEM_PROMPT.encode("utf-8"))
    hasher.update(RULES_PROMPT.encode("utf-8"))
    return hasher.hexdigest()


def _get_client(api_key: Optional[str]) -> OpenAI:
    return OpenAI(api_key=api_key)


def transcribe_image(path_normalized: Path, model: Optional[str] = None, api_key: Optional[str] = None) -> str:
    settings = get_settings()
    model_name = model or settings.openai_model
    api_key = api_key or settings.openai_api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    client = _get_client(api_key)
    image_url = _load_image_base64(path_normalized)

    content = [
        {"type": "text", "text": RULES_PROMPT},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0,
                max_tokens=4096,
            )
            message = response.choices[0].message.content or ""
            return message.strip()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_for = 2**attempt
            time.sleep(sleep_for)
    raise RuntimeError(f"Transcription failed after retries: {last_error}")
