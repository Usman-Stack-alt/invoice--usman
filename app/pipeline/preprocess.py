"""Decode uploads (image, PDF, DOCX) into pages: a raster for OCR, or a PDF's own text layer."""

import io
import zipfile
from dataclasses import dataclass

import cv2
import fitz  # pymupdf
import numpy as np
from PIL import Image, ImageOps

from app.config import get_settings
from app.pipeline.convert import docx_to_pdf
from app.pipeline.errors import ExtractionError
from app.pipeline.ocr import Word

MIN_WIDTH = 1500  # below this the image is upscaled to UPSCALE_TO: Tesseract wants ~25px+ text height
UPSCALE_TO = 1500
MAX_SIDE = 4200


PDF_DPI = 200
MIN_TEXT_WORDS = 15  # fewer embedded words than this => treat the page as a scan


@dataclass
class Page:
    """Exactly one of image / words is set."""

    image: Image.Image | None = None
    words: list[Word] | None = None  # native text layer, in pixels of `width`
    width: int = 0


def sniff(data: bytes) -> str:
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                names = z.namelist()
                if "word/document.xml" in names and "[Content_Types].xml" in names:
                    if sum(i.file_size for i in z.infolist()) > 200 * 1024 * 1024:
                        raise ExtractionError("DOCX is too large when decompressed")
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise ExtractionError("Unsupported file type (expected JPEG, PNG, TIFF, WEBP, PDF or DOCX)")
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise ExtractionError("Unsupported file type (expected JPEG, PNG, TIFF, WEBP, PDF or DOCX)")


def _pdf_pages(data: bytes) -> list[Page]:
    s = get_settings()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:
        raise ExtractionError(f"Corrupt PDF: {e}") from e
    if doc.is_encrypted:
        raise ExtractionError("Encrypted PDFs are not supported")
    if doc.page_count == 0:
        raise ExtractionError("PDF has no pages")
    zoom = PDF_DPI / 72
    pages = []
    for page in list(doc)[: s.max_pdf_pages]:
        raw = page.get_text("words")
        if len(raw) >= MIN_TEXT_WORDS:
            rot = page.rotation_matrix
            words = []
            for x0, y0, x1, y1, text, *_ in raw:
                r = fitz.Rect(x0, y0, x1, y1) * rot
                words.append(Word(text, int(r.x0 * zoom), int(r.y0 * zoom), int(r.x1 * zoom), int(r.y1 * zoom), 100.0))
            pages.append(Page(words=words, width=int(page.rect.width * zoom)))
        else:
            pix = page.get_pixmap(dpi=PDF_DPI, colorspace=fitz.csGRAY)
            pages.append(Page(image=Image.frombytes("L", (pix.width, pix.height), pix.samples)))
    return pages


def normalize_source(data: bytes) -> bytes:
    """DOCX -> PDF bytes (once); everything else unchanged. Lets the OCR path and the LLM share one conversion."""
    return docx_to_pdf(data) if sniff(data) == "docx" else data


def load_pages(data: bytes) -> list[Page]:
    s = get_settings()
    data = normalize_source(data)
    kind = sniff(data)
    if kind == "pdf":
        return _pdf_pages(data)

    Image.MAX_IMAGE_PIXELS = s.max_image_pixels
    try:
        img = Image.open(io.BytesIO(data))
        n_frames = min(getattr(img, "n_frames", 1), s.max_pdf_pages)
        pages = []
        for i in range(n_frames):
            img.seek(i)
            pages.append(Page(image=ImageOps.exif_transpose(img.copy())))
        return pages
    except (OSError, Image.DecompressionBombError, ValueError) as e:
        raise ExtractionError(f"Could not decode image: {e}") from e


def _deskew(gray: np.ndarray) -> np.ndarray:
    # Only dark pixels (text); ignores light-gray decorations such as side bars.
    ys, xs = np.where(gray < 110)
    if len(xs) < 500:
        return gray
    pts = np.column_stack([xs, ys]).astype(np.float32)
    angle = cv2.minAreaRect(pts)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    if not 0.3 < abs(angle) < 12:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Corners as top-left, top-right, bottom-right, bottom-left."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _flatten_page(gray: np.ndarray) -> np.ndarray:
    """Photo on a desk: find the paper outline and warp it flat. No-op for full-frame scans."""
    h, w = gray.shape
    k = 800 / max(h, w) if max(h, w) > 800 else 1.0
    small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k != 1.0 else gray
    edges = cv2.Canny(cv2.GaussianBlur(small, (7, 7), 0), 40, 120)
    edges = cv2.morphologyEx(cv2.dilate(edges, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray
    c = max(cnts, key=cv2.contourArea)
    frac = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not 0.25 < frac < 0.94:
        return gray
    approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
    quad = approx.reshape(4, 2) if len(approx) == 4 and cv2.isContourConvex(approx) else cv2.boxPoints(cv2.minAreaRect(c))
    tl, tr, br, bl = _order_quad(quad.astype(np.float32) / k)
    width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    if width < 300 or height < 300:
        return gray
    m = cv2.getPerspectiveTransform(
        np.array([tl, tr, br, bl], dtype=np.float32),
        np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32),
    )
    return cv2.warpPerspective(gray, m, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def prepare(page: Image.Image) -> np.ndarray:
    """RGB/any -> page-flattened, deskewed, contrast-normalised, size-normalised grayscale array."""
    if page.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", page.size, "white")
        bg.paste(page.convert("RGBA"), mask=page.convert("RGBA").split()[-1])
        page = bg
    rgb = page.convert("RGB")
    # Brightest channel: black text stays dark, but red/blue stamps and highlights fade out.
    # Grayscale inputs (R=G=B) are unaffected.
    gray = np.array(ImageOps.autocontrast(Image.fromarray(np.array(rgb).max(axis=2)), cutoff=1))
    gray = _flatten_page(gray)
    h, w = gray.shape
    scale = 1.0
    if w < MIN_WIDTH:
        scale = min(UPSCALE_TO / w, 3.0)
    elif max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    return _deskew(gray)
