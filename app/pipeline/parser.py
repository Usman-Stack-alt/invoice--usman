"""Layout-aware invoice parser: finds the table, parties and totals from word geometry and labels."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from itertools import pairwise
from statistics import median

from app.pipeline.layout import Line, group_lines, merge_numbers
from app.pipeline.normalize import (
    currency_of,
    is_number_token,
    money,
    number,
    parse_date,
    parse_number,
)
from app.pipeline.ocr import Word


@dataclass
class Cell:
    label: str
    x0: int
    x1: int
    canon: str | None = None

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class PageParse:
    number: str | None = None
    date: str | None = None
    due_date: str | None = None
    currency: str | None = None
    seller: dict = field(default_factory=dict)
    client: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    payment: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    table_found: bool = False


_INV_NO = re.compile(
    r"(?i)\b(?:invoice|inv|receipt|bill)\s*(?:no\.?|number|num\.?|nr\.?|#)(?![A-Za-z])\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9\-/]*)"
)
_DATE = re.compile(r"(?i)\b(?:invoice\s*)?date(?:\s*of\s*issue)?\b\s*[:.]?\s*(.+)$")
_DUE = re.compile(r"(?i)\b(?:payment\s*)?due(?:\s*date)?\b\s*[:.]?\s*(.+)$")
_NOT_ISSUE_DATE = re.compile(r"(?i)\bdue\b|date\s*paid|paid\s*date|date\s*of\s*payment|service\s*date")


def _candidates(ln: Line) -> list[str]:
    """Each cell alone, then each cell joined with its right neighbour (keeps other columns out of the value)."""
    texts = [" ".join(w.text for w in c) for c in _cells(ln)]
    return texts + [f"{a} {b}" for a, b in pairwise(texts)]


def _header_fields(lines: list[Line], out: PageParse, order: str) -> None:
    for ln in lines:
        for t in _candidates(ln):
            if out.number is None and (m := _INV_NO.search(t)):
                out.number = m[1].strip(".:")
            if out.due_date is None and (m := _DUE.search(t)):
                out.due_date = parse_date(m[1], order)
            elif out.date is None and not _NOT_ISSUE_DATE.search(t) and (m := _DATE.search(t)):
                out.date = parse_date(m[1], order)


_SELLER_ROLE = {"seller", "vendor", "supplier"}
_SELLER_PRE = {"from", "sold by", "issued by"}
# "Seller:", "From (Seller)", "To (Buyer)", "Bill To", "Client" ... label cell with nothing else in it
_LABEL = re.compile(
    r"(?i)^(?:(?P<pre>from|to|bill(?:ed)?\s*to|sold\s*by|issued\s*by|issued\s*to|invoice\s*to|ship\s*to)\s*)?"
    r"(?:\(?\s*(?P<role>seller|vendor|supplier|buyer|client|customer)\s*\)?)?\s*:?$"
)
_FIELD_LINE = re.compile(
    r"(?i)^(invoice\s*(no|number|nr|#|date)|date|due\s*date|service\s*period|period|currency|po\s*(no|number)|order)\b"
)
_ID_A = re.compile(
    r"(?i)^(?:(?:vat|tax|gst|ust)[\s-]*(?:id|nr|no|number|reg\w*)\b|tin\b|ntn\b|gstin\b|abn\b|acn\b|trn\b|ein\b)\.?\s*[:#]?\s*(.*)$"
)
_ID_B = re.compile(r"(?i)^(?:vat|tax)\s*:\s*(.*)$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _cells(ln: Line) -> list[list[Word]]:
    """Split a line into cells wherever the horizontal gap is large (separate columns)."""
    cells: list[list[Word]] = []
    for w in ln.words:
        if cells and w.x0 - cells[-1][-1].x1 <= 3 * ln.h:
            cells[-1].append(w)
        else:
            cells.append([w])
    return cells


def _party_labels(lines: list[Line]) -> list[tuple[str, Word]]:
    found: list[tuple[str, Word]] = []
    for ln in lines:
        for cell in _cells(ln):
            m = _LABEL.match(" ".join(w.text for w in cell).strip())
            if not m or not (m["pre"] or m["role"]):
                continue
            if m["role"]:
                role = "seller" if m["role"].lower() in _SELLER_ROLE else "client"
            else:
                role = "seller" if re.sub(r"\s+", " ", m["pre"].lower()) in _SELLER_PRE else "client"
            if all(r != role for r, _ in found):
                found.append((role, cell[0]))
    return found


def _parse_party(col_lines: list[str]) -> dict:
    p: dict = {"name": None, "address": None, "tax_id": None, "iban": None, "email": None}
    addr: list[str] = []
    seen_ids = False
    for i, t in enumerate(col_lines):
        if m := _EMAIL.search(t):
            p["email"] = m[0]
        elif m := _ID_A.match(t) or _ID_B.match(t):
            p["tax_id"], seen_ids = m[1].strip() or None, True
        elif m := re.match(r"(?i)^iban\W*(.*)$", t):
            p["iban"], seen_ids = re.sub(r"\s+", "", m[1]) or None, True
        elif i == 0:
            p["name"] = t
        elif not seen_ids:
            addr.append(t)
    p["address"] = "\n".join(addr) or None
    return p


def _parties(lines: list[Line], width: int, stop_y: float, out: PageParse) -> None:
    labels = _party_labels(lines)
    tol = 0.02 * width
    indent = 0.06 * width
    for role, lw in labels:
        right = [o.x0 for _, o in labels if o is not lw and o.x0 > lw.x0 and abs(o.yc - lw.yc) < 2 * lw.h]
        x_lo, x_hi = lw.x0 - tol, (min(right) - tol if right else float("inf"))
        col: list[str] = []
        prev_y, pitch = lw.yc, None
        for ln in lines:
            if ln.y <= lw.yc + 0.5 * lw.h:
                continue
            if x_hi != float("inf"):  # a neighbouring label bounds this column
                txt = " ".join(w.text for w in ln.words if x_lo <= w.x0 < x_hi)
            else:  # last column: the run of words starting under the label, cut at the first big gap (drops e.g. JOB SITE)
                run: list[Word] = []
                for w in (w for w in ln.words if w.x0 >= x_lo):
                    if run and w.x0 - run[-1].x1 > 3 * ln.h:
                        break
                    run.append(w)
                txt = " ".join(w.text for w in run) if run and run[0].x0 < lw.x0 + indent else ""
            if ln.y >= stop_y or re.match(r"(?i)^items?\W*$", ln.text) or (txt and _FIELD_LINE.match(txt)):
                break
            if not txt:
                continue
            step = ln.y - prev_y
            # a vertical gap much larger than the block's own line pitch ends the block
            if pitch and step > 2.5 * pitch:
                break
            if col:  # the label->first-line distance is not a line pitch
                pitch = min(pitch or step, step)
            prev_y = ln.y
            col.append(txt)
        setattr(out, role, _parse_party(col))


_TITLE = re.compile(r"(?i)^(?:tax\s+)?(?:invoice|receipt|credit\s*note|bill)\b")
_PHONE = re.compile(r"(?i)^(?:tel|phone|ph|mob|mobile|fax|call)\b|^[+\d][\d\s()+.-]{7,}$")


def _ends_letterhead(ln: Line) -> bool:
    """Document title or the first key/value header line: the issuer block is over (tolerates OCR slips in the title)."""
    t = ln.text
    return bool(
        _FIELD_LINE.match(t) or _INV_NO.search(t) or (len(ln.words) <= 4 and re.search(r"(?i)\b(invoice|receipt)\b|\bINVOI", t))
    )


def _letterhead(lines: list[Line], stop_y: float, out: PageParse) -> bool:
    """No seller label: take the issuer from the first block above the document title."""
    name, addr, tax, email = None, [], None, None
    for ln in lines:
        if ln.y >= stop_y or _TITLE.match(ln.text) or _party_labels([ln]) or _ends_letterhead(ln):
            break
        cell = next((c for c in _cells(ln) if len(" ".join(w.text for w in c)) > 2), None)  # skips a logo letter
        if not cell:
            continue
        for seg in re.split(r"\s*[|·•]\s*", " ".join(w.text for w in cell)):
            seg = seg.strip()
            if not seg or _PHONE.match(seg) or re.match(r"(?i)^lic", seg):
                continue
            if m := _EMAIL.search(seg):
                email = email or m[0]
            elif m := _ID_A.match(seg):
                tax = tax or (m[1].strip() or None)
            elif name is None:
                name = seg
            else:
                addr.append(seg)
        if len(addr) > 4:
            break
    if not name:
        return False
    out.seller = {"name": name, "address": "\n".join(addr) or None, "tax_id": tax, "iban": None, "email": email}
    out.warnings.append("seller_from_letterhead")
    return True


_PAY_KEYS = [
    (r"(?:payment\s*)?reference|ref\.?", "reference"),
    (r"beneficiary|account\s*(?:holder|name)", "beneficiary"),
    (r"bank(?:\s*name)?", "bank"),
    (r"iban", "iban"),
    (r"(?:bic|swift)(?:\s*/\s*(?:bic|swift))?(?:\s*code)?", "bic"),
    (r"account\s*(?:number|no\.?)", "account_number"),
]


def _payment(lines: list[Line], out: PageParse) -> None:
    for ln in lines:
        cells = _cells(ln)
        if len(cells) >= 2:
            label, value = " ".join(w.text for w in cells[0]), " ".join(w.text for c in cells[1:] for w in c)
        elif m := re.match(r"^([A-Za-z /.]+?)\s*:\s*(.+)$", ln.text):
            label, value = m[1], m[2]
        else:
            continue
        for pat, key in _PAY_KEYS:
            if re.fullmatch(pat, label.strip(" :"), re.I) and key not in out.payment:
                out.payment[key] = re.sub(r"\s+", "", value) if key in ("iban", "bic") else value.strip()
                break


_HDR_KEYS = re.compile(r"(?i)^(qty|quantity|price|total|amount|net|gross|vat|tax|rate|um|uom|unit|units)")


def _canon(label: str) -> str | None:
    text = label.lower().strip()
    if re.match(r"^(no\.?|#|pos\.?|nr\.?|sl\.?|sr\.?|item\s*(no\.?|#|code)|sku|code)$", text):
        return "index"
    if text.startswith("descr") or text in (
        "item", "item name", "items", "product", "product name", "article", "particulars", "service", "details",
    ):  # fmt: skip
        return "description"
    if re.match(r"^(qty|quantity|qnty|units?\s*sold|hours)", text):
        return "quantity"
    if re.match(r"^(um|uom|unit|units)$", text):
        return "unit"
    if "gross" in text:
        return "gross_amount"
    if re.match(r"^(net\s*)?(unit\s*)?(price|rate|cost)", text):
        return "unit_price"
    if re.match(r"^(vat|tax|gst)", text):
        return "vat_percent"
    if re.match(r"^(net\s*(worth|amount|total)|amount|total|line\s*total|subtotal)", text):
        return "net_amount"
    return None


def _find_header(lines: list[Line]) -> int | None:
    for i, ln in enumerate(lines):
        if (
            any(re.match(r"(?i)^(descr|item|product|article|particulars|service)", w.text) for w in ln.words)
            and sum(1 for w in ln.words if _HDR_KEYS.match(w.text)) >= 1
        ):
            return i
    return None


def _header_cells(lines: list[Line], i: int) -> tuple[list[Cell], float]:
    ln = lines[i]
    cells: list[Cell] = []
    for w in ln.words:
        if cells and w.x0 - cells[-1].x1 < 1.0 * ln.h:
            c = cells[-1]
            c.label, c.x1 = f"{c.label} {w.text}", w.x1
        else:
            cells.append(Cell(w.text, w.x0, w.x1))
    bottom = max(w.y1 for w in ln.words)
    # wrapped header text ("Gross" / "worth" on two lines)
    if i + 1 < len(lines):
        nxt = lines[i + 1]
        if nxt.y - ln.y < 2.0 * ln.h and all(re.fullmatch(r"[A-Za-z%\[\]./]+", w.text) for w in nxt.words):
            for w in nxt.words:
                for c in cells:
                    if c.x0 - ln.h <= w.xc <= c.x1 + ln.h:
                        c.label = f"{c.label} {w.text}"
                        c.x0, c.x1 = min(c.x0, w.x0), max(c.x1, w.x1)
                        bottom = max(bottom, w.y1)
                        break
    for c in cells:
        c.canon = _canon(c.label)
    return cells, bottom


_END = re.compile(
    r"(?i)^(summary|sub\s?-?total|total|sales\s*tax|tax|vat|shipping|discount|balance|amount\s*due|items?\s*:|no\.?\s*of\s*items)"
)


def _is_table_end(ln: Line, num_zone_x: float, n_slots: int) -> bool:
    """A summary keyword ends the table if it is right-aligned, says SUMMARY, or has fewer amounts than a data row."""
    if not _END.match(ln.text):
        return False
    first = ln.words[0]
    if first.text.lower().startswith("summary") or first.x0 >= num_zone_x - 1:
        return True
    amounts = [w for w in merge_numbers(ln.words) if is_number_token(w.text) and not w.text.endswith("%")]
    return len(amounts) < min(2, n_slots)


def _blocks(desc_lines: list[tuple[float, str, float]], n_anchors: int) -> list[list[tuple[float, str, float]]] | None:
    """Group description lines into rows, trying gap thresholds until blocks match the number of numeric rows."""
    if not desc_lines:
        return []
    h = median(h for _, _, h in desc_lines)
    for k in (1.7, 1.5, 1.9, 1.35, 2.1, 1.25, 2.4):
        blocks, cur = [], [desc_lines[0]]
        for prev, nxt in pairwise(desc_lines):
            if nxt[0] - prev[0] > k * h:
                blocks.append(cur)
                cur = []
            cur.append(nxt)
        blocks.append(cur)
        if len(blocks) == n_anchors:
            return blocks
    return None


Reread = Callable[[Word], "str | None"]


def _needs_reread(w: Word) -> bool:
    # Only isolated glyphs and non-numeric garbage. Re-reading a plausible multi-digit
    # number is worse than trusting the block-level pass (crops pull in neighbour digits).
    return len(w.text) == 1 or not is_number_token(w.text)


def _fix_numeric_words(body: list[Line], slots: list[Cell], num_zone_x: float, reread: Reread, out: PageParse) -> None:
    for ln in body:
        for k, w in enumerate(ln.words):
            if w.x0 < num_zone_x or not _needs_reread(w):
                continue
            slot = min(slots, key=lambda c: min(abs(w.xc - c.xc), abs(w.x1 - c.x1)))
            if slot.canon == "unit" or slot.canon is None:
                continue
            txt = reread(w)
            if txt and is_number_token(txt) and txt != w.text:
                ln.words[k] = Word(txt, w.x0, w.y0, w.x1, w.y1, 95.0)


def _parse_table(lines: list[Line], hdr: int, width: int, out: PageParse, reread: Reread | None) -> int:
    """Fills out.items, returns index of first line after the table."""
    cells, hdr_bottom = _header_cells(lines, hdr)
    desc = next((c for c in cells if c.canon == "description"), None)
    cols = [c for c in cells if desc and c.x0 > desc.x1 - 1 and c.canon not in (None, "description", "index")]
    if desc is None or not cols:
        out.warnings.append("table_header_unrecognised")
        return hdr + 1
    out.table_found = True
    # every cell right of description keeps its slot, even if unmapped, so positional matching works
    slots = [c for c in cells if c.x0 > desc.x1 - 1]
    num_zone_x = slots[0].x0 - 0.05 * width
    desc_x0 = desc.x0 - 0.02 * width

    end = len(lines)
    body: list[Line] = []
    for j in range(hdr + 1, len(lines)):
        ln = lines[j]
        if ln.y <= hdr_bottom + 0.3 * ln.h:
            continue
        if _is_table_end(ln, num_zone_x, len(slots)):
            end = j
            break
        body.append(ln)

    if reread:
        _fix_numeric_words(body, slots, num_zone_x, reread, out)

    anchors: list[tuple[float, list[Word]]] = []
    desc_lines: list[tuple[float, str, float]] = []
    for ln in body:
        nums = [w for w in ln.words if w.x0 >= num_zone_x]
        text_desc = " ".join(w.text for w in ln.words if desc_x0 <= w.x0 < num_zone_x)
        n_numeric = sum(1 for w in merge_numbers(nums) if is_number_token(w.text))
        if n_numeric >= min(2, len(slots)):
            anchors.append((ln.y, nums))
        if re.search(r"[A-Za-z0-9]", text_desc):  # ignore '-----' / '=====' rules
            desc_lines.append((ln.y, text_desc, ln.h))

    blocks = _blocks(desc_lines, len(anchors))
    if blocks is None:
        out.warnings.append("row_grouping_fallback")
        blocks = [[] for _ in anchors]
        for dl in desc_lines:
            k = min(range(len(anchors)), key=lambda a: abs(anchors[a][0] - dl[0])) if anchors else None
            if k is not None:
                blocks[k].append(dl)

    for n, ((_, nums), block) in enumerate(zip(anchors, blocks, strict=False), start=1):
        item = {"line": n, "description": " ".join(t for _, t, _ in block)}
        toks = merge_numbers(nums)
        assign: dict[int, Word] = {}
        if len(toks) == len(slots):
            assign = dict(enumerate(toks))
        else:
            for tk in toks:
                k = min(range(len(slots)), key=lambda s: min(abs(tk.xc - slots[s].xc), abs(tk.x1 - slots[s].x1)))
                if k in assign:
                    out.warnings.append(f"item_{n}_column_conflict")
                    continue
                assign[k] = tk
        for k, tk in assign.items():
            canon = slots[k].canon
            if canon == "unit":
                item["unit"] = tk.text
                continue
            val = parse_number(tk.text) if is_number_token(tk.text) else None
            if val is None or canon is None:
                continue
            item[canon] = {"quantity": number, "vat_percent": number}.get(canon, money)(val)
        out.items.append(item)
    if len(anchors) != len(blocks):
        out.warnings.append("row_count_mismatch")
    return end


_TAIL_OK = re.compile(r"^[$€£]$|^[A-Z]{3}$")


def _summary(lines: list[Line], out: PageParse) -> None:
    s = out.summary
    taxes: list[Decimal] = []
    charges: list[dict] = []
    for ln in lines:
        toks = [w for w in merge_numbers(ln.words) if re.search(r"\w", w.text)]  # drop OCR noise like '|' '©'
        first = next((i for i, w in enumerate(toks) if is_number_token(w.text) and not w.text.endswith("%")), None)
        if first is None:
            continue
        # a summary row ends with its amounts (optionally a currency code); prose and ID lines do not
        if not all(is_number_token(w.text) or _TAIL_OK.match(w.text) for w in toks[first:]):
            continue
        label = " ".join(w.text for w in toks[:first]).lower().strip(" :")
        amounts = [parse_number(w.text) for w in toks[first:] if is_number_token(w.text) and not w.text.endswith("%")]
        pct = next((parse_number(w.text.strip("()")) for w in toks if w.text.strip("()").endswith("%")), None)
        if not label:
            if len(amounts) >= 3 and pct is not None:
                s["tax_rate"] = number(pct)
            continue
        if re.match(r"^total\b", label) and len(amounts) >= 3:  # net | vat | gross table
            s["subtotal"], s["tax"], s["total"] = (money(a) for a in amounts[-3:])
            continue
        amt = amounts[-1]
        if re.match(r"^sub\s?-?total", label):
            s["subtotal"] = money(amt)
        elif re.match(r"^(sales\s*)?(tax|vat|gst|pst|hst|qst)\b", label):
            taxes.append(amt)
            if pct is not None and len(taxes) == 1:
                s["tax_rate"] = number(pct)
        elif label.startswith("discount"):
            s["discount"] = money(amt)
        elif re.match(r"^(shipping|handling|freight|delivery|fee|surcharge)", label):
            charges.append({"label": " ".join(w.text for w in toks[:first]).strip(" :"), "amount": money(amt)})
        elif re.match(r"^(amount|balance)\s*due|^balance\b", label):
            s["amount_due"] = money(amt)
        elif re.match(r"^(amount\s*)?paid\b|^payment\s*received|^deposit", label):
            s["amount_paid"] = money(amt)
        elif re.match(r"^(grand\s*)?total\b", label):
            s["total"] = money(amt)
    if taxes:
        s["tax"] = money(sum(taxes))
        if len(taxes) > 1:
            s["tax_rate"] = None  # several taxes: a single rate would mislead
    if charges:
        s["other_charges"] = charges
    if s.get("total") is None and s.get("amount_due") is not None and s.get("amount_paid") is None:
        s["total"] = s["amount_due"]  # unpaid invoice that only states the amount due


def parse_page(words: list[Word], width: int, date_order: str = "MDY", reread: Reread | None = None) -> PageParse:
    out = PageParse()
    lines = group_lines(words)
    out.currency = currency_of(" ".join(w.text for w in words))

    hdr = _find_header(lines)
    if hdr is None:
        out.warnings.append("no_table_found")
        _header_fields(lines, out, date_order)
        _summary(lines, out)
        return out

    end = _parse_table(lines, hdr, width, out, reread)
    _header_fields(lines[:hdr] + lines[end:], out, date_order)
    stop_y = lines[hdr].y - 0.5 * lines[hdr].h
    _parties(lines[:hdr], width, stop_y, out)
    if not out.seller:
        _letterhead(lines[:hdr], stop_y, out)
    _summary(lines[end:], out)
    _payment(lines[end:], out)
    return out
