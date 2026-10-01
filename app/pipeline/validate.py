"""Arithmetic / completeness checks. These drive the confidence score and review flag."""

from app.schemas import InvoiceData


def _close(a: float, b: float, rel: float = 0.003, abs_tol: float = 0.03) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b)))


def validate(data: InvoiceData) -> tuple[list[str], list[str]]:
    """Returns (inconsistencies, missing). Also fills a missing line net from qty*price."""
    bad: list[str] = []
    missing: list[str] = []

    if not data.invoice.invoice_number:
        missing.append("missing:invoice_number")
    if not data.invoice.date:
        missing.append("missing:date")
    if not data.items:
        bad.append("no_line_items")

    for it in data.items:
        q, p, net, gross, vat = it.quantity, it.unit_price, it.net_amount, it.gross_amount, it.vat_percent
        # '1.350' kg may have been read as 1350: trust the arithmetic when it fits better
        if q and p and net and q >= 1000 and not _close(q * p, net) and _close(q / 1000 * p, net):
            it.quantity = q = q / 1000
        # exactly one of qty / price / amount missing: the other two determine it
        if net is None and q is not None and p is not None:
            it.net_amount = net = round(q * p, 2)
        elif p is None and q and net is not None:
            it.unit_price = p = round(net / q, 2)
            missing.append(f"item_{it.line}:unit_price_derived")
        if q is not None and p is not None and net is not None and not _close(q * p, net):
            bad.append(f"item_{it.line}:qty_x_price_ne_net")
        if net is not None and gross is not None and vat is not None and not _close(net * (1 + vat / 100), gross):
            bad.append(f"item_{it.line}:net_plus_vat_ne_gross")
        if net is None and gross is None:
            bad.append(f"item_{it.line}:no_amount")

    s = data.summary
    nets = [i.net_amount for i in data.items if i.net_amount is not None]
    grosses = [i.gross_amount for i in data.items if i.gross_amount is not None]
    if s.subtotal is not None and nets and len(nets) == len(data.items) and not _close(sum(nets), s.subtotal):
        bad.append("items_sum_ne_subtotal")
    if s.total is not None:
        if s.subtotal is not None:
            expect = s.subtotal + (s.tax or 0) + sum(c.amount for c in s.other_charges) - (s.discount or 0)
            if not _close(expect, s.total):
                bad.append("subtotal_plus_tax_ne_total")
        elif grosses and len(grosses) == len(data.items) and not _close(sum(grosses), s.total):
            bad.append("items_gross_sum_ne_total")
    else:
        missing.append("missing:total")
    return bad, missing
