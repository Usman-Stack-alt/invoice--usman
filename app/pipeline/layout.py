"""Geometry helpers: group OCR words into visual lines and merge split numbers."""

import re
from dataclasses import dataclass, field
from statistics import median

from app.pipeline.normalize import is_number_token
from app.pipeline.ocr import Word


@dataclass
class Line:
    words: list[Word] = field(default_factory=list)

    @property
    def y(self) -> float:
        return sum(w.yc for w in self.words) / len(self.words)

    @property
    def h(self) -> float:
        return median(w.h for w in self.words)

    @property
    def x0(self) -> int:
        return min(w.x0 for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def group_lines(words: list[Word]) -> list[Line]:
    """Cluster words by vertical centre; words are then ordered left-to-right."""
    if not words:
        return []
    tol = 0.5 * median(w.h for w in words)  # global: thin glyphs ('-') must not shrink it
    lines: list[Line] = []
    for w in sorted(words, key=lambda w: w.yc):
        for ln in reversed(lines[-3:]):
            if abs(w.yc - ln.y) <= tol:
                ln.words.append(w)
                break
        else:
            lines.append(Line([w]))
    for ln in lines:
        ln.words.sort(key=lambda w: w.x0)
    return sorted(lines, key=lambda ln: ln.y)


_CUR_ONLY = re.compile(r"^[$€£]$")


def merge_numbers(words: list[Word]) -> list[Word]:
    """Re-join thousands groups that OCR split ('1' + '394,67'), drop lone currency signs."""
    out: list[Word] = []
    for w in sorted(words, key=lambda w: w.x0):
        if _CUR_ONLY.match(w.text):
            continue
        if out:
            p = out[-1]
            gap = w.x0 - p.x1
            if (
                gap < 1.4 * max(p.h, w.h)
                and re.fullmatch(r"[$€£]?\d{1,3}", p.text)
                and re.fullmatch(r"\d{3}([.,]\d{1,2})?", w.text)
            ):
                out[-1] = Word(p.text + w.text, p.x0, min(p.y0, w.y0), w.x1, max(p.y1, w.y1), min(p.conf, w.conf))
                continue
        out.append(w)
    return out


def number_words(words: list[Word]) -> list[Word]:
    return [w for w in merge_numbers(words) if is_number_token(w.text)]
