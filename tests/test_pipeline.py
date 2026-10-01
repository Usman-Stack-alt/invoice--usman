import pytest

from app.pipeline import ExtractionError, extract


def test_template_a_seller_client_layout(template_a):
    r = extract(template_a)
    assert r.invoice.invoice_number == "51109338"
    assert r.invoice.date == "2013-04-13"
    assert r.seller.name == "Andrews, Kirby and Valdez"
    assert r.seller.iban == "GB75MCRL06841367619257"
    assert r.client.name == "Becker Ltd"
    assert r.client.address == "8012 Stewart Summit Apt. 455\nNorth Douglas, AZ 95355"
    assert len(r.items) == 7
    assert (r.items[3].quantity, r.items[3].unit_price, r.items[3].net_amount, r.items[3].gross_amount) == (
        3,
        464.89,
        1394.67,
        1534.14,
    )
    assert (r.summary.subtotal, r.summary.tax, r.summary.total) == (5640.17, 564.02, 6204.19)
    assert r.meta.status == "ok" and r.meta.warnings == []


def test_template_b_freeform_layout(template_b):
    r = extract(template_b)
    assert (r.invoice.invoice_number, r.invoice.date, r.invoice.due_date) == ("9362", "2022-03-13", "2022-04-05")
    assert [i.quantity for i in r.items] == [2, 4, 5, 4, 5]
    assert [i.net_amount for i in r.items] == [24, 32, 35, 36, 15]
    assert r.summary.subtotal == 142 and r.summary.tax == 12.48 and r.summary.total == 164.48
    assert r.summary.other_charges[0].amount == 10
    # no Seller/Client labels on this layout: the issuer is inferred from the letterhead, and says so
    assert "seller_from_letterhead" in r.meta.warnings


def test_unreadable_input_is_a_permanent_error():
    with pytest.raises(ExtractionError):
        extract(b"not an image")


def test_blank_image_is_rejected():
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (1600, 2200), "white").save(buf, "PNG")
    with pytest.raises(ExtractionError):
        extract(buf.getvalue())


def test_scanned_pdf_goes_through_ocr(scanned_pdf):
    r = extract(scanned_pdf)
    assert "tesseract" in r.meta.engine and "pdf-text-layer" not in r.meta.engine
    assert r.invoice.invoice_number == "51109338" and r.summary.total == 6204.19 and len(r.items) == 7


def _check_digital_invoice(r):
    assert r.invoice.invoice_number == "INV-1001"
    assert r.invoice.date == "2021-05-06" and r.invoice.due_date == "2021-06-05"
    assert [(i.quantity, i.unit_price, i.net_amount) for i in r.items] == [(2, 10, 20), (3, 10, 30)]
    assert "wraps" in r.items[1].description
    assert (r.summary.subtotal, r.summary.tax, r.summary.tax_rate, r.summary.total) == (50, 5, 10, 55)
    assert r.meta.status == "ok"


@pytest.mark.skipif(not __import__("shutil").which("soffice"), reason="LibreOffice not installed")
def test_docx_is_converted_and_uses_text_layer(docx_invoice):
    r = extract(docx_invoice)
    assert r.meta.engine == "pdf-text-layer"  # no OCR needed for born-digital documents
    _check_digital_invoice(r)


@pytest.mark.skipif(not __import__("shutil").which("soffice"), reason="LibreOffice not installed")
def test_digital_pdf_uses_text_layer(docx_invoice):
    from app.pipeline.convert import docx_to_pdf

    r = extract(docx_to_pdf(docx_invoice))
    assert r.meta.engine == "pdf-text-layer"
    _check_digital_invoice(r)


def test_non_word_zip_is_rejected():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("xl/workbook.xml", "<x/>")
    with pytest.raises(ExtractionError):
        extract(buf.getvalue())


def test_from_seller_to_buyer_layout_with_payment_details(from_to_pdf):
    r = extract(from_to_pdf)
    assert r.meta.engine == "pdf-text-layer"
    assert r.invoice.invoice_number == "INV-2026-08"
    assert (r.invoice.date, r.invoice.due_date, r.invoice.currency) == ("2026-08-28", "2026-09-07", "EUR")
    # parties are cut by column and stop before the invoice-number block
    assert r.seller.name == "Jane Example"
    assert r.seller.address == "12 Example Road, Block 4\nSpringfield, Exampleland"
    assert r.seller.email == "jane@example.com"
    assert r.client.name == "Acme Buyer GmbH"
    assert r.client.address == "Sample Gasse 1\n1010 Sampletown\nAustria"
    assert r.client.tax_id == "ATU00000000"
    assert "missing:parties" not in r.meta.warnings
    assert [(i.quantity, i.unit_price) for i in r.items] == [(1, 625)] and r.summary.total == 625
    assert r.payment.model_dump() == {
        "beneficiary": "Jane Example", "bank": "Example Bank Ltd", "iban": "PK00EXMP0000000000000000",
        "bic": "EXMPPKKA000", "account_number": None, "reference": "INV-2026-08",
    }  # fmt: skip


@pytest.mark.parametrize(
    "text,code",
    [
        ("Currency EUR", "EUR"),
        ("Price (GBP) Qty", "GBP"),
        ("Total $ 5 640,17", "USD"),
        ("Total € 10", "EUR"),
        ("nothing", None),
        ("Our CAD team Currency CHF", "CHF"),
    ],
)
def test_currency_detection(text, code):
    from app.pipeline.normalize import currency_of

    assert currency_of(text) == code


def test_photo_on_a_desk_with_stamp_is_flattened_and_read():
    """Tilted, perspective-distorted photo with a PAID stamp and handwriting."""
    from pathlib import Path

    r = extract(Path(__file__).with_name("fixtures").joinpath("photo_invoice.jpg").read_bytes())
    assert (r.invoice.invoice_number, r.invoice.date, r.invoice.due_date) == ("00742", "2026-09-09", "2026-09-23")
    assert r.invoice.currency == "AUD"
    assert r.seller.name == "Coastline Plumbing & Gas Pty Ltd" and r.client.name == "Sarah & Tom Whitfield"
    assert [(i.quantity, i.net_amount) for i in r.items] == [(1, 95), (3.5, 420), (1, 1689), (1, 148.5), (1, 60)]
    assert (r.summary.subtotal, r.summary.tax, r.summary.total, r.summary.amount_due) == (2412.5, 241.25, 2653.75, 0)
    assert r.meta.status == "ok"


def test_narrow_thermal_receipt_with_description_above_numbers():
    from pathlib import Path

    r = extract(Path(__file__).with_name("fixtures").joinpath("receipt.jpg").read_bytes())
    assert (r.invoice.invoice_number, r.invoice.date, r.invoice.currency) == ("TI-00218734", "2026-09-27", "AED")
    assert r.seller.name == "AL NOOR HYPERMARKET LLC" and r.seller.tax_id == "100234567800003"
    assert len(r.items) == 8 and r.items[0].description == "BASMATI RICE 5KG"
    assert (r.summary.subtotal, r.summary.tax, r.summary.total) == (157.93, 7.9, 165.83)
    assert r.items[2].quantity == 1.35  # '1.350' kg must not become 1350


def test_modern_saas_invoice_separates_total_paid_and_due():
    from pathlib import Path

    r = extract(Path(__file__).with_name("fixtures").joinpath("saas_invoice.png").read_bytes())
    assert (r.invoice.invoice_number, r.invoice.date, r.invoice.currency) == ("INV-7F3A-20260901", "2026-09-01", "CAD")
    assert r.seller.name == "Cloudpine Software Inc." and r.client.name == "Tidewater Analytics Ltd."
    assert r.client.email == "finance@tidewater.example"
    assert [(i.quantity, i.net_amount) for i in r.items] == [(12, 348), (1, 15), (1, 49)]
    # two taxes add up; the footer tax-registration line is not mistaken for a tax row
    assert (r.summary.subtotal, r.summary.tax, r.summary.total) == (412, 49.44, 461.44)
    assert (r.summary.amount_paid, r.summary.amount_due) == (461.44, 0)
