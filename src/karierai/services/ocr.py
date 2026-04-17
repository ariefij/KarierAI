from __future__ import annotations

import re
import shutil
from io import BytesIO
from typing import Any

try:
    import fitz
except Exception:
    fitz = None

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception:
    Image = None
    ImageFilter = None
    ImageOps = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    import pytesseract
except Exception:
    pytesseract = None

_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_MAX_IMAGE_PIXELS = 20_000_000
_MAX_OCR_PDF_PAGES = 6


def _get_ocr_languages() -> str:
    preferred = ["ind", "eng"]
    if pytesseract is None:
        raise RuntimeError("pytesseract belum terpasang.")

    available: set[str] = set()
    try:
        result = pytesseract.get_languages(config="")
        available = {item.strip() for item in result if item.strip()}
    except Exception:
        pass

    selected = [lang for lang in preferred if lang in available]
    if not selected:
        selected = ["eng"] if "eng" in available or not available else [next(iter(available))]
    return "+".join(selected)


def is_ocr_ready() -> tuple[bool, str]:
    if pytesseract is None:
        return False, "pytesseract belum terpasang"
    if not shutil.which("tesseract"):
        return False, "binary tesseract tidak ditemukan di PATH"
    return True, "ready"


def _load_image(image_bytes: bytes):
    if Image is None or ImageOps is None:
        raise RuntimeError("Pillow belum terpasang.")

    image = Image.open(BytesIO(image_bytes))
    image.load()
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > _MAX_IMAGE_PIXELS:
        raise ValueError("Ukuran gambar CV terlalu besar untuk diproses dengan aman.")
    return image


def _prepare_images_for_ocr(image) -> list[Any]:
    if ImageFilter is None or ImageOps is None:
        raise RuntimeError("Pillow belum terpasang.")

    variants = []
    base = image.convert("L")
    autocontrast = ImageOps.autocontrast(base)
    if min(autocontrast.size) < 1600:
        scale = max(2, int(1800 / max(1, min(autocontrast.size))))
        autocontrast = autocontrast.resize((autocontrast.width * scale, autocontrast.height * scale))
    variants.append(autocontrast)
    sharpened = autocontrast.filter(ImageFilter.SHARPEN)
    variants.append(sharpened)
    binary = sharpened.point(lambda px: 255 if px > 170 else 0)
    variants.append(binary)
    return variants


def _normalize_ocr_text(text: str) -> str:
    normalized = " ".join(text.split())
    replacements = {
        r"\bsol\b": "sql",
        r"\b5ql\b": "sql",
        r"\bsqi\b": "sql",
        r"\bpyth0n\b": "python",
        r"\btabieau\b": "tableau",
        r"\bpowerbi\b": "power bi",
        r"\bmach1ne learning\b": "machine learning",
    }
    for pattern, repl in replacements.items():
        normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
    return normalized


def _ocr_single_image(image) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract belum terpasang.")

    ready, reason = is_ocr_ready()
    if not ready:
        raise RuntimeError(reason)

    languages = _get_ocr_languages()
    candidates: list[str] = []
    configs = ["--psm 6", "--psm 3", "--psm 11"]
    for prepared in _prepare_images_for_ocr(image):
        for config in configs:
            text = pytesseract.image_to_string(prepared, lang=languages, config=config) or ""
            cleaned = " ".join(text.split())
            if cleaned:
                candidates.append(cleaned)
    best = max(candidates, key=len) if candidates else ""
    return _normalize_ocr_text(best)


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    image = _load_image(image_bytes)
    text = _ocr_single_image(image)
    if not text:
        raise ValueError("Gambar CV tidak berhasil dibaca. Pastikan teks terlihat jelas dan resolusinya cukup.")
    return text


def _render_pdf_to_images(pdf_bytes: bytes) -> list[Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF belum terpasang.")
    if Image is None:
        raise RuntimeError("Pillow belum terpasang.")

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[Any] = []
    try:
        for page_index, page in enumerate(document):
            if page_index >= _MAX_OCR_PDF_PAGES:
                break
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            pages.append(Image.open(BytesIO(pix.tobytes("png"))))
    finally:
        document.close()
    return pages


def _extract_text_pdf_native(pdf_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf belum terpasang.")

    reader = PdfReader(BytesIO(pdf_bytes))
    page_texts: list[str] = []
    for page in reader.pages[:_MAX_OCR_PDF_PAGES]:
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            page_texts.append(cleaned)
    return "\n".join(page_texts).strip()


def _looks_like_useful_text(text: str) -> bool:
    compact = " ".join(text.split())
    return len(compact) >= 20 and sum(ch.isalpha() for ch in compact) >= 10


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    native_text = _extract_text_pdf_native(pdf_bytes)
    if _looks_like_useful_text(native_text):
        return native_text

    ocr_pages: list[str] = []
    for image in _render_pdf_to_images(pdf_bytes):
        text = _ocr_single_image(image)
        cleaned = " ".join(text.split())
        if cleaned:
            ocr_pages.append(cleaned)

    combined_ocr = "\n".join(ocr_pages).strip()
    if _looks_like_useful_text(combined_ocr):
        return combined_ocr
    if native_text:
        return native_text
    raise ValueError("PDF tidak berhasil dibaca. Pastikan file tidak rusak dan teks pada CV scan terlihat jelas.")


def extract_text_from_upload_bytes(file_name: str, content_type: str | None, raw_bytes: bytes) -> str:
    lowered_name = (file_name or "").lower()
    lowered_type = (content_type or "").lower()
    if lowered_name.endswith(".pdf") or lowered_type in {"application/pdf", "application/x-pdf"}:
        return extract_text_from_pdf_bytes(raw_bytes)
    if any(lowered_name.endswith(ext) for ext in _SUPPORTED_IMAGE_EXTENSIONS) or lowered_type.startswith("image/"):
        return extract_text_from_image_bytes(raw_bytes)
    raise ValueError("Format file belum didukung. Upload CV dalam bentuk PDF, PNG, JPG, JPEG, WEBP, BMP, atau TIFF.")
