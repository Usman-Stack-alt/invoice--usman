"""Parsing of messy OCR strings into clean numbers and ISO dates."""

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from dateutil import parser as dateparser

_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP"}
_CENT = Decimal("0.01")


def parse_number(raw: str) -> Decimal | None:
    """'1 394,67' -> 1394.67, '$5,640.17' -> 5640.17, '(12.00)' -> -12.00, '10%' -> 10."""
    s = raw.strip()
    if not s or re.search(r"[A-Za-z]", s) or not re.search(r"\d", s):
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = re.sub(r"[^\d.,]", "", s)
    if not s:
        return None
    sep_pos = max(s.rfind(","), s.rfind("."))
    if sep_pos == -1:
        digits = s
    else:
        frac = s[sep_pos + 1 :]
        if len(frac) == 3:  # 1,000 / 1.000 -> thousands separator
            digits = re.sub(r"[.,]", "", s)
        else:
            digits = re.sub(r"[.,]", "", s[:sep_pos]) + "." + frac
    try:
        d = Decimal(digits)
    except InvalidOperation:
        return None
    return -d if neg else d


def is_number_token(tok: str) -> bool:
    return bool(re.fullmatch(r"\(?-?[$€£]?\d[\d.,]*%?\)?", tok)) and parse_number(tok) is not None


_ISO = {
    "USD", "EUR", "GBP", "CHF", "JPY", "CNY", "INR", "PKR", "AED", "SAR", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK",
    "PLN", "CZK", "HUF", "RON", "TRY", "ZAR", "BRL", "MXN", "SGD", "HKD", "KRW",
}  # fmt: skip


_PREFIXED = {"CA$": "CAD", "C$": "CAD", "A$": "AUD", "AU$": "AUD", "US$": "USD", "NZ$": "NZD", "HK$": "HKD", "S$": "SGD"}


def currency_of(text: str) -> str | None:
    """'Currency EUR' > '(EUR)' header > 'CA$' style prefix > ISO code next to an amount > symbol."""
    if (m := re.search(r"(?i)\bcurrency\W{0,3}([A-Z]{3})\b", text)) and m[1].upper() in _ISO:
        return m[1].upper()
    if m := next((m for m in re.finditer(r"\(([A-Z]{3})\)", text) if m[1] in _ISO), None):
        return m[1]
    for pre, code in _PREFIXED.items():
        if re.search(rf"(?<![A-Za-z]){re.escape(pre)}\s?\d", text):
            return code
    # a bare code counts only when it sits beside a number: "$0.00 CAD", "EUR 100"
    if m := next((m for m in re.finditer(r"\d\s*([A-Z]{3})\b|\b([A-Z]{3})\s*[$€£]?\d", text) if (m[1] or m[2]) in _ISO), None):
        return m[1] or m[2]
    if "$" in text:  # a bare dollar sign is ambiguous: use country hints, else USD
        if re.search(r"(?i)\bABN\b|australia", text):
            return "AUD"
        if re.search(r"(?i)canada|\bGST/HST\b", text):
            return "CAD"
    for sym, code in _CURRENCY.items():
        if sym in text:
            return code
    return None


def money(d: Decimal | None) -> float | None:
    return None if d is None else float(d.quantize(_CENT))


def number(d: Decimal | None) -> float | None:
    """Quantities / percentages: no forced 2dp."""
    if d is None:
        return None
    return float(d.normalize()) if d != d.to_integral() else float(d.to_integral())


def parse_date(raw: str | None, order: str = "MDY") -> str | None:
    """Return ISO yyyy-mm-dd or None."""
    if not raw:
        return None
    s = raw.strip().strip(".,:;")
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    try:
        if m:
            return date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s)
        if m:
            a, b, y = int(m[1]), int(m[2]), int(m[3])
            y += 2000 if y < 100 else 0
            if a > 12:
                day, mon = a, b
            elif b > 12:
                day, mon = b, a
            else:
                day, mon = (b, a) if order == "MDY" else (a, b)
            return date(y, mon, day).isoformat()
        if re.search(r"[A-Za-z]{3}", s):
            return dateparser.parse(s, fuzzy=True, dayfirst=order == "DMY").date().isoformat()
    except (ValueError, OverflowError):
        return None
    return None
