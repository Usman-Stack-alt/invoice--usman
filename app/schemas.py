from typing import Literal

from pydantic import BaseModel, Field


class InvoiceMeta(BaseModel):
    invoice_number: str | None = None
    date: str | None = Field(None, description="ISO yyyy-mm-dd")
    due_date: str | None = None
    currency: str | None = Field(None, description="ISO 4217 when a symbol was detected")


class Party(BaseModel):
    name: str | None = None
    address: str | None = None
    tax_id: str | None = None
    iban: str | None = None
    email: str | None = None


class Payment(BaseModel):
    beneficiary: str | None = None
    bank: str | None = None
    iban: str | None = None
    bic: str | None = None
    account_number: str | None = None
    reference: str | None = None


class LineItem(BaseModel):
    line: int
    description: str = ""
    quantity: float | None = None
    unit: str | None = None
    unit_price: float | None = None
    vat_percent: float | None = None
    net_amount: float | None = Field(None, description="Line amount before tax")
    gross_amount: float | None = Field(None, description="Line amount including tax, when the invoice shows it")


class Charge(BaseModel):
    label: str
    amount: float


class Summary(BaseModel):
    subtotal: float | None = Field(None, description="Net total before tax")
    tax: float | None = None
    tax_rate: float | None = None
    discount: float | None = None
    other_charges: list[Charge] = []
    total: float | None = Field(None, description="Grand total of the invoice")
    amount_paid: float | None = None
    amount_due: float | None = Field(None, description="Outstanding balance (0 when already paid)")


class ExtractionMeta(BaseModel):
    status: Literal["ok", "needs_review"]
    confidence: float = Field(description="0..1, OCR confidence reduced by failed consistency checks")
    warnings: list[str] = []
    pages: int = 1
    engine: str = Field(description="tesseract-<version> and/or pdf-text-layer, plus the LLM model when it was used")
    llm_used: bool = Field(False, description="True when the LLM contributed to this result")
    llm_fields: list[str] = Field([], description='Fields taken from the LLM, e.g. ["seller.name", "items"]')
    processing_ms: int


class InvoiceData(BaseModel):
    """The clean JSON the frontend consumes."""

    invoice: InvoiceMeta = InvoiceMeta()
    seller: Party = Party()
    client: Party = Party()
    items: list[LineItem] = []
    summary: Summary = Summary()
    payment: Payment = Payment()
    meta: ExtractionMeta


class BatchItem(BaseModel):
    filename: str
    ok: bool
    data: InvoiceData | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    results: list[BatchItem]


class ErrorResponse(BaseModel):
    detail: str
