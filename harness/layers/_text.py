"""So khớp văn bản dùng chung cho `critic` và `citation_checker`.

Chuẩn hoá giống hệt `arena.scorer._norm` (NFC, casefold, gộp khoảng trắng)
để layer đánh giá claim đúng như bộ chấm. Chỉ dùng để SO SÁNH — không bao
giờ ghi kết quả chuẩn hoá ngược vào `claim["text"]`.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

#: Ký tự mô hình thật hay bọc quanh một trích dẫn. Cắt chúng ở HAI ĐẦU vẫn
#: là cắt bớt (substring), nên không mất provenance.
_EDGE_CHARS = " \t\r\n.,;:!?\"'“”‘’*`_-–—()[]"


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def doc_lines(doc) -> tuple:
    """Các dòng đã chuẩn hoá của một tài liệu (một trích dẫn nằm trên một dòng)."""
    return tuple(line for line in (norm(raw) for raw in doc.body.splitlines()) if line)


def on_one_line(lines, normalised: str) -> bool:
    return bool(normalised) and any(normalised in line for line in lines)


def fetched_whole(doc, seen: str) -> bool:
    """Thân tài liệu về NGUYÊN VẸN từ một lần fetch sạch."""
    return norm(doc.body) in seen


def retrieved(doc, seen: str) -> bool:
    """Agent đã thấy tài liệu: fetch nguyên vẹn, hoặc doc_id có trong kết quả search."""
    return fetched_whole(doc, seen) or doc.doc_id.casefold() in seen


def trimmed(text: str) -> str:
    """Bỏ dấu câu / định dạng ở hai đầu — một substring của chính `text`."""
    return text.strip(_EDGE_CHARS)
