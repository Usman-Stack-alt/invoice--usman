"""Merge the rules result with the LLM answer, field by field (policy: README, "LLM fallback")."""

import math
import re
from copy import deepcopy

from app.pipeline.llm import LLMInvoice, LLMItem
from app.pipeline.validate import validate
from app.schemas import ExtractionMeta, InvoiceData, LineItem, Party, Payment, Summary

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STUB_META = ExtractionMeta(status="ok", confidence=0, engine="", processing_ms=0)


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, abs_tol=0.005)
    return re.sub(r"\s+", " ", str(a)).strip().casefold() == re.sub(r"\s+", " ", str(b)).strip().casefold()


def _pick(rules, llm, trusted: bool):
    """Returns (value, taken_from_llm)."""
    if llm in (None, ""):
        return rules, False
    if rules in (None, ""):
        return llm, True
    if _same(rules, llm):
        return rules, False
    return (rules, False) if trusted else (llm, True)


def _items(llm_items: list[LLMItem]) -> list[LineItem]:
    r2 = lambda v: None if v is None else round(v, 2)  # noqa: E731
    return [
        LineItem(
            line=i,
            description=(it.description or "").strip(),
            quantity=it.quantity,
            unit=it.unit,
            unit_price=r2(it.unit_price),
            vat_percent=it.vat_percent,
            net_amount=r2(it.net_amount),
            gross_amount=r2(it.gross_amount),
        )  # fmt: skip
        for i, it in enumerate(llm_items, start=1)
    ]


def _problems(items: list[LineItem], summary: Summary) -> int:
    bad, _ = validate(InvoiceData(items=deepcopy(items), summary=deepcopy(summary), meta=_STUB_META))
    return len(bad)


_SUMMARY_FIELDS = ("subtotal", "tax", "tax_rate", "discount", "total", "amount_paid", "amount_due")


def merge(rules: InvoiceData, llm: LLMInvoice, trusted: bool) -> tuple[InvoiceData, list[str]]:
    out = rules.model_copy(deep=True)
    taken: list[str] = []

    def take(path: str, from_llm: bool) -> None:
        if from_llm:
            taken.append(path)

    m = out.invoice
    for f in ("invoice_number", "date", "due_date", "currency"):
        cand = getattr(llm, f)
        if f in ("date", "due_date") and cand and not _ISO_DATE.match(cand):
            cand = None
        if f == "currency":
            cand = cand.strip().upper() if cand and re.fullmatch(r"[A-Za-z]{3}", cand.strip()) else None
        val, got = _pick(getattr(m, f), cand, trusted)
        setattr(m, f, val)
        take(f"invoice.{f}", got)

    for role in ("seller", "client"):
        theirs, mine = getattr(llm, role), getattr(out, role)
        if theirs is None:
            continue
        party_trusted = trusted and not (role == "seller" and "seller_from_letterhead" in rules.meta.warnings)
        for f in ("name", "address", "tax_id", "email"):
            val, got = _pick(getattr(mine, f), getattr(theirs, f), party_trusted)
            setattr(mine, f, val)
            take(f"{role}.{f}", got)
        setattr(out, role, Party(**mine.model_dump()))

    if llm.payment:
        for f in Payment.model_fields:
            val, got = _pick(getattr(out.payment, f), getattr(llm.payment, f, None), trusted)
            setattr(out.payment, f, val)
            take(f"payment.{f}", got)

    llm_items = _items(llm.items)
    llm_sum = Summary(**{f: getattr(llm, f) for f in _SUMMARY_FIELDS})
    candidates = [("rules", out.items, out.summary), ("llm", llm_items, llm_sum)]
    if not out.items:
        candidates = candidates[1:]
    elif not llm_items:
        candidates = candidates[:1]
    if trusted:
        order = [("rules", "rules"), ("llm", "llm"), ("rules", "llm"), ("llm", "rules")]
    else:
        order = [("llm", "llm"), ("rules", "rules"), ("llm", "rules"), ("rules", "llm")]
    pool = {n: (i, s) for n, i, s in candidates}
    combos = [(a, b) for a, b in order if a in pool and b in pool]
    best = min(combos, key=lambda c: _problems(pool[c[0]][0], pool[c[1]][1]))
    out.items = [it.model_copy(update={"line": n}) for n, it in enumerate(pool[best[0]][0], start=1)]
    if best[0] == "llm" and llm_items:
        taken.append("items")
    chosen, other = (out.summary, llm_sum) if best[1] == "rules" else (llm_sum, out.summary)
    merged_sum = chosen.model_copy(deep=True)
    for f in _SUMMARY_FIELDS:
        if getattr(merged_sum, f) is None:
            setattr(merged_sum, f, getattr(other, f))
    merged_sum.other_charges = rules.summary.other_charges  # the LLM schema has no charges list
    for f in _SUMMARY_FIELDS:
        rv, mv = getattr(rules.summary, f), getattr(merged_sum, f)
        take(f"summary.{f}", mv is not None and (rv is None or not _same(rv, mv)))
    out.summary = merged_sum
    return out, list(dict.fromkeys(taken))
