"""Governance — PII detection + redaction (mandatory)"""
from __future__ import annotations
from ..contracts import *  # noqa

import re

_BN_TO_ASCII_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
 
 
def _digit_normalized(text: str) -> str:
    return text.translate(_BN_TO_ASCII_DIGITS)
 
 
# Ordered by priority: earlier patterns claim their span first, later
# (broader/numeric) patterns are skipped if they'd overlap an earlier match.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("URL", re.compile(r"https?://[^\s)>\]]+")),
    # Bangladeshi mobile: optional +880/880 country code, then 01[3-9]xxxxxxxx (11 digits)
    ("PHONE_BD", re.compile(r"(?:\+?88)?01[3-9]\d{8}\b")),
    # General international phone: + followed by 7-15 digits, optional separators
    ("PHONE_INTL", re.compile(r"\+\d{1,3}[-\s]?\d{2,4}[-\s]?\d{2,4}[-\s]?\d{2,6}\b")),
    ("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
    )),
    # Bangladesh NID: old (13-digit), new smart card (10 or 17-digit)
    ("NID_BD", re.compile(r"\b\d{17}\b|\b\d{13}\b|\b\d{10}\b")),
    # Generic card number: 13-16 digits, optionally space/dash grouped
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
]



def detect(text: str) -> list[tuple[int,int,str]]:
    """Return (start,end,type) PII spans. IMPLEMENT."""
    # raise NotImplementedError("Governance: PII detect")
    if not text:
        return []
 
    normalized = _digit_normalized(text)
    claimed: list[tuple[int, int]] = []
    spans: list[tuple[int, int, str]] = []
 
    for label, pattern in _PATTERNS:
        for m in pattern.finditer(normalized):
            start, end = m.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue  # overlaps a higher-priority match already found
            claimed.append((start, end))
            spans.append((start, end, label))
 
    spans.sort(key=lambda s: s[0])
    return spans


def redact(text: str) -> str:
    # raise NotImplementedError("Governance: PII redact")
    spans = detect(text)
    if not spans:
        return text
 
    parts = []
    last = 0
    for start, end, label in spans:
        parts.append(text[last:start])
        parts.append(f"[REDACTED:{label}]")
        last = end
    parts.append(text[last:])
    return "".join(parts)

def _scrub_value(value):
    """Recursively redact every string found in a ctx value, regardless of
    which key it's nested under -- robust to AFTER_OCR / BEFORE_ANSWER /
    ON_LOG passing differently-shaped ctx dicts."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _scrub_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        scrubbed = [_scrub_value(v) for v in value]
        return type(value)(scrubbed)
    return value


def register(hooks) -> None:
    """Wire PII redaction into the pipeline. IMPLEMENT the handler (call redact())."""
    def _scrub(ctx: dict) -> dict:
        # raise NotImplementedError("PII: redact text/answer/log in ctx")
        return _scrub_value(ctx)
    
    hooks.register(hooks.AFTER_OCR, _scrub)       # scrub extracted text before indexing
    hooks.register(hooks.BEFORE_ANSWER, _scrub)   # scrub the outgoing answer
    hooks.register(hooks.ON_LOG, _scrub)          # scrub logs
