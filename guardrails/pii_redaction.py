import re
import logging
from typing import Dict, Iterable, Optional, Set, Tuple

logger = logging.getLogger("aegis.guardrails.pii")

# ======================================================
# Patterns
# ======================================================
_EMAIL_RE = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+')
_PHONE_US_RE = re.compile(r'(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)')
_SSN_RE = re.compile(r'(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)')
_CC_CANDIDATE_RE = re.compile(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)')
_IP_RE = re.compile(r'(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?!\d)')

_SIMPLE_PATTERNS = {
    "EMAIL": (_EMAIL_RE, "[REDACTED_EMAIL]"),
    "PHONE_US": (_PHONE_US_RE, "[REDACTED_PHONE]"),
    "SSN": (_SSN_RE, "[REDACTED_SSN]"),
    "IP_ADDRESS": (_IP_RE, "[REDACTED_IP]"),
}

# IP_ADDRESS is intentionally OFF by default. In this PLC/industrial
# automation domain, device IP addresses are routine, legitimate technical
# content technicians need to discuss (e.g. "the PLC at 192.168.0.5 isn't
# responding on the rack"). Enable it explicitly via enabled_categories if
# your deployment's threat model needs it scrubbed anyway.
DEFAULT_ENABLED_CATEGORIES: Set[str] = {"EMAIL", "PHONE_US", "SSN", "CREDIT_CARD"}
ALL_CATEGORIES: Set[str] = set(_SIMPLE_PATTERNS.keys()) | {"CREDIT_CARD"}


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact_credit_cards(text: str) -> Tuple[str, int]:
    """
    Card numbers are Luhn-validated before redaction. This matters a lot in
    this domain specifically: a naive "13-19 digit sequence" regex would
    happily clobber PLC serial numbers, fault codes, and rack/slot addressing
    strings that are the whole point of the conversation. Luhn validation
    filters almost all of those out since they won't checksum correctly.
    """
    count = 0

    def _sub(match: "re.Match") -> str:
        nonlocal count
        raw = match.group(0)
        digits = re.sub(r'[ -]', '', raw)
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            count += 1
            return "[REDACTED_CC]"
        return raw

    redacted = _CC_CANDIDATE_RE.sub(_sub, text)
    return redacted, count


def redact_pii(text: str, enabled_categories: Optional[Iterable[str]] = None) -> Tuple[str, Dict[str, int]]:
    """
    Scrubs common PII patterns from text.

    Returns (redacted_text, counts) where counts maps category name -> number
    of redactions made for that category (empty dict if nothing was found).
    Never logs or returns the actual matched PII values — only counts and
    category names, so logs stay clean even when this fires.

    Pass enabled_categories to override DEFAULT_ENABLED_CATEGORIES, e.g.
    redact_pii(text, enabled_categories={"EMAIL", "IP_ADDRESS"}).
    """
    if not text:
        return text, {}

    categories = set(enabled_categories) if enabled_categories is not None else DEFAULT_ENABLED_CATEGORIES
    counts: Dict[str, int] = {}
    result = text

    for name, (pattern, replacement) in _SIMPLE_PATTERNS.items():
        if name not in categories:
            continue
        result, n = pattern.subn(replacement, result)
        if n:
            counts[name] = n

    if "CREDIT_CARD" in categories:
        result, n = _redact_credit_cards(result)
        if n:
            counts["CREDIT_CARD"] = n

    return result, counts