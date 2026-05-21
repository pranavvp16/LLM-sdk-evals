"""
PII redaction for inference log previews.

Strips common PII patterns before writing to InferenceLog.input_preview
and InferenceLog.output_preview.

Patterns covered:
  - Email addresses
  - US/international phone numbers
  - US Social Security Numbers
  - Credit card numbers (16-digit)
  - IP addresses

Inspired by real production practice:
  "Run PII detection inline on log entries before writing to persistent storage."
"""

from __future__ import annotations
import re

# ── Redaction patterns ─────────────────────────────────────────────────────────

_PATTERNS = [
    # Email (before phone — @ disambiguates)
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # Credit card: exactly 16 contiguous digits (run BEFORE phone to avoid partial match)
    (re.compile(r"\b\d{16}\b"), "[CC]"),
    # US SSN: 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # US phone: (123) 456-7890 / 123-456-7890 / +1 123 456 7890
    (re.compile(r"(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"), "[PHONE]"),
    # IPv4
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
]


def redact(text: str) -> str:
    """Apply all PII patterns to `text` and return the redacted string."""
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
