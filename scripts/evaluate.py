"""Field-level accuracy of the pipeline against the dataset's ground truth.

    python scripts/evaluate.py --n 100 [--cache /tmp/inv_eval]

Downloads samples.json + N annotated images from the Hugging Face dataset repo.
"""

import argparse
import json
import random
import re
import statistics
import urllib.request
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from app.pipeline import extract

BASE = "https://huggingface.co/datasets/Voxel51/high-quality-invoice-images-for-ocr/resolve/main/"


def fetch(rel: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BASE + rel, dest)
    return dest


def num(s) -> float | None:
    if s in (None, ""):
        return None
    s = re.sub(r"[^\d.,]", "", str(s))
    if not s:
        return None
    if "," in s and ("." not in s or s.rfind(",") > s.rfind(".")):
        s = s.replace(".", "").replace(",", ".")
    return float(s.replace(",", ""))


def norm(s) -> str:
    return re.sub(r"\W+", " ", (s or "")).strip().lower()


def iso(d: str) -> str | None:
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", d or "")
    return f"{m[3]}-{m[1]}-{m[2]}" if m else None


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def close(a: float | None, b: float | None, tol: float = 0.015) -> bool:
    return a is not None and b is not None and abs(a - b) < tol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--cache", default="/tmp/inv_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default="off", choices=["off", "auto", "always"], help="default off: never spends LLM quota")
    a = ap.parse_args()
    cache = Path(a.cache)
    samples = json.loads(fetch("samples.json", cache / "samples.json").read_text())["samples"]
    ann = [s for s in samples if s.get("json_annotation")]
    random.Random(a.seed).shuffle(ann)

    hits: dict[str, list[bool]] = defaultdict(list)
    times: list[int] = []
    flagged = 0
    wrong_but_ok: list[str] = []  # extraction judged "ok" while a key field is wrong
    failures: list[tuple[str, str]] = []
    for s in ann[: a.n]:
        gt = json.loads(s["json_annotation"])
        img = fetch(s["filepath"], cache / s["filepath"])
        try:
            r = extract(img.read_bytes(), a.llm)
        except Exception as e:
            failures.append((s["filepath"], repr(e)))
            continue
        times.append(r.meta.processing_ms)
        flagged += r.meta.status == "needs_review"
        inv, sm = gt["invoice"], gt["subtotal"]
        row: dict[str, bool] = {
            "invoice_number": r.invoice.invoice_number == inv["invoice_number"],
            "invoice_date": r.invoice.date == iso(inv["invoice_date"]),
            "seller_name": norm(r.seller.name) == norm(inv["seller_name"]),
            "client_name": norm(r.client.name) == norm(inv["client_name"]),
            "seller_address": norm(r.seller.address) == norm(inv["seller_address"]),
            "client_address": norm(r.client.address) == norm(inv["client_address"]),
            "tax_total": close(r.summary.tax, num(sm["tax"])),
            "grand_total": close(r.summary.total, num(sm["total"])),
            "item_count": len(r.items) == len(gt["items"]),
        }
        for k, v in row.items():
            hits[k].append(v)
        if len(r.items) == len(gt["items"]):
            for it, g in zip(r.items, gt["items"], strict=False):
                hits["item_quantity"].append(close(it.quantity, num(g["quantity"]), 1e-6))
                # ground-truth 'total_price' is net or gross depending on the invoice: accept either
                tp = num(g["total_price"])
                hits["item_amount"].append(any(close(v, tp) for v in (it.net_amount, it.gross_amount)))
                hits["item_description (sim>=0.9)"].append(similar(it.description, g["description"]) >= 0.9)
        if r.meta.status == "ok" and not all(row[k] for k in ("grand_total", "item_count", "invoice_number")):
            wrong_but_ok.append(s["filepath"])

    n = len(times)
    print(
        f"\nEvaluated {n} invoices ({len(failures)} crashed); mean {statistics.mean(times):.0f} ms, "
        f"p95 {sorted(times)[max(int(n * .95) - 1, 0)]} ms"
    )
    print(f"Flagged needs_review: {flagged}/{n}")
    print(f"Silent errors (status ok but number/total/item-count wrong): {len(wrong_but_ok)}  {wrong_but_ok[:5]}\n")
    for k, v in hits.items():
        print(f"{k:30s} {sum(v) / len(v):6.1%}  ({sum(v)}/{len(v)})")
    for f in failures[:5]:
        print("FAIL", f)


if __name__ == "__main__":
    main()
