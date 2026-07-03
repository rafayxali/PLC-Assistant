import os
import base64
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("aegis.guardrails.moderation")

MODERATION_MODEL = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")

_client = None
_client_init_attempted = False


def _get_client():
    """
    Lazily builds the OpenAI client. Returns None (and logs once) if
    OPENAI_API_KEY isn't set, so the app can still boot in environments
    where moderation is intentionally disabled — callers must treat None
    as "moderation unavailable, fail open" rather than crash.
    """
    global _client, _client_init_attempted
    if _client is not None:
        return _client
    if _client_init_attempted:
        return None
    _client_init_attempted = True

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set — moderation checks are DISABLED. "
            "Set OPENAI_API_KEY before deploying to production; the Moderation "
            "endpoint is free to use but still requires an OpenAI API key."
        )
        return None

    try:
        from openai import OpenAI
        _client = OpenAI(api_key=api_key)
        return _client
    except Exception:
        logger.exception("Failed to initialize OpenAI client for moderation.")
        return None


class ModerationResult:
    def __init__(self, flagged: bool, categories: List[str], raw: Optional[dict] = None):
        self.flagged = flagged
        self.categories = categories  # names of flagged categories, e.g. ["violence", "harassment"]
        self.raw = raw

    def __repr__(self):
        return f"ModerationResult(flagged={self.flagged}, categories={self.categories})"


def _image_to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/jpeg"
    data = image_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def check_moderation(text: Optional[str] = None, image_path: Optional[Path] = None) -> ModerationResult:
    """
    Runs OpenAI's Moderation endpoint against the incoming query text and/or
    an attached diagnostic image (a single multi-modal call when both are
    present, since omni-moderation-latest supports mixed text+image input).

    Fails OPEN on any error (missing API key, network failure, API error):
    logs the problem and returns flagged=False rather than blocking every
    request when moderation itself is broken. Flip to fail-closed here if
    your deployment needs stricter enforcement over availability.
    """
    if not text and image_path is None:
        return ModerationResult(flagged=False, categories=[])

    client = _get_client()
    if client is None:
        return ModerationResult(flagged=False, categories=[])

    content = []
    if text:
        content.append({"type": "text", "text": text})
    if image_path is not None:
        try:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(image_path)}})
        except Exception:
            logger.exception("Failed to read image for moderation check: %s", image_path)

    if not content:
        return ModerationResult(flagged=False, categories=[])

    try:
        response = client.moderations.create(model=MODERATION_MODEL, input=content)
        result = response.results[0]
        categories_dict = result.categories.model_dump()
        flagged_categories = [name for name, is_flagged in categories_dict.items() if is_flagged]
        return ModerationResult(
            flagged=result.flagged,
            categories=flagged_categories,
            raw=result.model_dump(),
        )
    except Exception:
        logger.exception("Moderation API call failed; failing open for this request.")
        return ModerationResult(flagged=False, categories=[])