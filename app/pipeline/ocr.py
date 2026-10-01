"""Thin wrapper around Tesseract 5: returns word boxes with confidences."""

from dataclasses import dataclass

import numpy as np
import pytesseract
from pytesseract import Output, TesseractError

from app.config import get_settings
from app.pipeline.errors import ExtractionError


@dataclass(slots=True)
class Word:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def h(self) -> int:
        return max(self.y1 - self.y0, 1)


def tesseract_version() -> str:
    return str(pytesseract.get_tesseract_version())


def read_words(gray: np.ndarray) -> list[Word]:
    s = get_settings()
    cfg = f"--oem 1 --psm {s.tesseract_psm} -c preserve_interword_spaces=1"
    try:
        d = pytesseract.image_to_data(gray, lang=s.tesseract_lang, config=cfg, output_type=Output.DICT, timeout=s.ocr_timeout_s)
    except RuntimeError as e:  # pytesseract raises RuntimeError on timeout
        raise TimeoutError(f"OCR timed out: {e}") from e  # transient -> retried by worker
    except TesseractError as e:
        raise ExtractionError(f"Tesseract failed: {e}") from e

    words: list[Word] = []
    for i, text in enumerate(d["text"]):
        text = text.strip()
        conf = float(d["conf"][i])
        if not text or conf < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(Word(text, x, y, x + w, y + h, conf))
    return words


def reread_number(gray: np.ndarray, w: Word) -> str | None:
    """Second pass on one numeric cell: crop, upscale, digits-only whitelist, single line.
    Fixes isolated glyphs that block-level OCR misreads ('2' -> '9', '5' -> 'a')."""
    import cv2

    pad_x, pad_y = 6, 6
    h, wd = gray.shape
    crop = gray[max(w.y0 - pad_y, 0) : min(w.y1 + pad_y, h), max(w.x0 - pad_x, 0) : min(w.x1 + pad_x, wd)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    cfg = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,$%-"
    try:
        txt = pytesseract.image_to_string(crop, lang=get_settings().tesseract_lang, config=cfg, timeout=10)
    except (RuntimeError, TesseractError):
        return None
    txt = txt.strip().replace(" ", "")
    return txt or None
