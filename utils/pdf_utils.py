import fitz  # pymupdf
import base64
from typing import Dict, List


def extract_pages_as_text(pdf_bytes: bytes) -> Dict[int, str]:
    """Extract text from each page. Returns {page_num: text}"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    for i, page in enumerate(doc):
        pages[i] = page.get_text()
    doc.close()
    return pages


def extract_pages_as_images(pdf_bytes: bytes, dpi: int = 150) -> Dict[int, str]:
    """Extract each page as base64 PNG. Returns {page_num: base64_string}"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = {}
    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        pages[i] = base64.b64encode(img_bytes).decode("utf-8")
    doc.close()
    return pages


def extract_selected_pages_bytes(pdf_bytes: bytes, page_nums: List[int]) -> bytes:
    """Extract specific pages into a new PDF bytes object."""
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    dst = fitz.open()
    for p in page_nums:
        if p < len(src):
            dst.insert_pdf(src, from_page=p, to_page=p)
    result = dst.tobytes()
    src.close()
    dst.close()
    return result