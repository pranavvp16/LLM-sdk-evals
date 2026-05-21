"""Regex patterns and constants used by the guardrails filters."""

from __future__ import annotations

import re

MAX_INPUT_CHARS = 16_000

# Detect classic prompt-injection / jailbreak phrasings. Patterns are
# case-insensitive and match across whitespace. They're intentionally narrow:
# false positives hurt more than missed catches because the L2 safety judge
# is the real backstop.
PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(the\s+)?(system|previous)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s*(mode)?", re.IGNORECASE),
    re.compile(r"\b(do\s+anything\s+now|DAN\s+mode)\b", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions)", re.IGNORECASE),
    re.compile(r"print\s+(your\s+)?(system\s+)?(prompt|instructions)\s+verbatim", re.IGNORECASE),
)

# Patterns we never want to send back to the user — usually a sign the model
# leaked chat-template tokens or echoed a destructive instruction.
BANNED_OUTPUT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"),
    re.compile(r"<\|begin_of_text\|>|<\|eot_id\|>"),
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
)

# Tool-argument tightening. The Tool schemas themselves are JSON Schema
# draft-07 (validated by sdk/normalize.py); these are extra checks above
# what the schema enforces.
ISO_8601_PATTERN: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+\-]\d{2}:?\d{2})?$"
)
CONTROL_CHAR_PATTERN: re.Pattern[str] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
